"""
myCobot 320 부드러운 직선경로 이동 - 전면 재설계 버전
======================================================

이전 버전의 근본 문제:
    직선 검증이 실패하면 파이썬에서 웨이포인트를 하나씩 send_angles로
    끊어 보냈고, 펌웨어는 각 명령마다 가속-감속을 반복하므로 멈칫거림이
    발생했다. 실제 로봇들이 부드러운 이유는 "경로 보간을 펌웨어가 통째로"
    하기 때문이며, 파이썬에서 웨이포인트를 스트리밍하는 방식 자체가 잘못.

새 설계 원칙:
    1. 실제 로봇 이동은 오직 send_coords(target, speed, mode=1) 만 사용.
       (펌웨어가 내부적으로 직선 보간 -> 항상 부드러움)
    2. 파이썬 쪽 웨이포인트 계산은 "안전 검증"에만 사용하고,
       절대 실행 명령으로 흘려보내지 않음.
    3. 직선 A->B가 위험하면: 경유점 C를 탐색해서
       A->C 직선 + C->B 직선 (부드러운 이동 2번, C에서 자연스러운 정지 1번)
       으로 우회. 이것도 실패하면 이동 자체를 거부하고 이유+추천좌표 출력.

사전 설치:
    pip3 install ikpy pybullet numpy scipy
"""

import os
import re
import glob
import time
import math
import numpy as np
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R, Slerp
import pybullet as p
from pymycobot import MyCobot320
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

PLOT_TRAJECTORY = True      # 이동 후 실제 경로를 XY/YZ/ZX 평면 그래프로 저장/표시할지
TRAJ_SAMPLE_SEC = 0.08      # 이동 중 좌표 샘플링 간격(초)
SESSION_DIR = None          # 이번 실행의 그래프 저장 폴더 (main()에서 설정됨)


def create_session_dir():
    """path_test_N_날짜_시간 형식의 새 폴더를 만들어 이번 실행의 그래프를 모아 저장"""
    existing = glob.glob("path_test_*")
    nums = []
    for d in existing:
        m = re.match(r"path_test_(\d+)_", os.path.basename(d))
        if m:
            nums.append(int(m.group(1)))
    n = max(nums, default=0) + 1
    folder = f"path_test_{n}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(folder, exist_ok=True)
    return folder

# ---------------------------------------------------------
# 0. 설정
# ---------------------------------------------------------
URDF_PATH = "/home/kms/colcon_ws/src/mycobot_ros2/mycobot_description/urdf/mycobot_320_m5_2022/mycobot_320_m5_2022.urdf"
PACKAGE_ROOT = "/home/kms/colcon_ws/src/mycobot_ros2/mycobot_description"
GRID_FILE_PATH = "/home/kms/바탕화면/python_code/mycobot_reachability_grid.npz"

import glob

PORT = None  # None이면 자동 감지 (아래 find_robot_port 사용)
BAUD_RATE = 115200


def find_robot_port():
    """/dev/ttyACM*, /dev/ttyUSB* 를 순회하며 get_angles()에 정상 응답하는 포트를 찾음"""
    candidates = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
    if not candidates:
        return None, None
    for port in candidates:
        try:
            mc_try = MyCobot320(port, BAUD_RATE)
            time.sleep(0.8)
            angles = mc_try.get_angles()
            if isinstance(angles, list) and len(angles) == 6:
                print(f"🔌 로봇 발견: {port}")
                return mc_try, port
            # 응답 없는 포트는 닫고 다음 후보로
            try:
                mc_try._serial_port.close()
            except Exception:
                pass
        except Exception:
            continue
    return None, None
DESK_SAFETY_MARGIN_MM = 20
ALIGN_ANGLES = [0, -30, 30, 0, 30, 0]  # 준비 자세: 전부 0도(특이점)를 피해 팔꿈치를 굽힌 시작 자세
JOINT_LIMITS_DEG = [(-165, 165), (-165, 165), (-165, 165), (-165, 165), (-165, 165), (-175, 175)]
JOINT_LIMITS_RAD = [(math.radians(lo), math.radians(hi)) for lo, hi in JOINT_LIMITS_DEG]

SPEED = 50                  # 관절 이동(send_angles) 속도
LINEAR_SPEED = 80           # 직선 이동(send_coords mode=1) 전용 속도 - 펌웨어가 직선 모드 속도를 다르게(느리게) 해석하는 경향이 있어 별도로 높게 설정 (1~100)
CHECK_STEP_MM = 15          # 검증용 웨이포인트 간격 (실행에는 사용 안 함)
POS_TOL_MM = 15             # 검증 IK 수렴 허용 오차 (검증용일 뿐, 실제 이동 정밀도는 로봇 펌웨어가 담당)
ORIENT_TOL_DEG = 5
IK_POLISH_TRIES = 8         # 웨이포인트당 IK 다듬기 최대 횟수
CONDITION_NUMBER_MAX = 90   # 특이점 근접 판정 (조건수)
SETTLE_DELAY_SEC = 0.3
MOVE_TIMEOUT_SEC = 30

# 경유점 탐색: 중간점에서 이만큼 벗어난 후보들을 시도 (mm)
VIA_OFFSETS_MM = [60, 100, 140]

# ---------------------------------------------------------
# 1. ikpy 체인 로드
# ---------------------------------------------------------
chain = Chain.from_urdf_file(URDF_PATH, base_elements=["base"])
active_indices = [i for i, l in enumerate(chain.links) if l.joint_type != "fixed"]
joint_names_ikpy = [chain.links[i].name for i in active_indices]
n_active = len(active_indices)

# ---------------------------------------------------------
# 2. PyBullet 초기화 (자가충돌 검사용)
# ---------------------------------------------------------
with open(URDF_PATH, "r") as f:
    urdf_text = f.read()

def _resolve_package_uri(match):
    rel_path = match.group(1)
    parts = rel_path.split("/", 1)
    if len(parts) == 2:
        return f'filename="{PACKAGE_ROOT}/{parts[1]}"'
    return match.group(0)

fixed_urdf_text = re.sub(r'filename="package://([^"]+)"', _resolve_package_uri, urdf_text)
TEMP_URDF_PATH = "/tmp/mycobot_320_m5_2022_fixed_paths.urdf"
with open(TEMP_URDF_PATH, "w") as f:
    f.write(fixed_urdf_text)

p.connect(p.DIRECT)
robot_id = p.loadURDF(
    TEMP_URDF_PATH, useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
)
name_to_pb_index = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    if info[2] != p.JOINT_FIXED:
        name_to_pb_index[info[1].decode("utf-8")] = j
pb_indices_in_order = [name_to_pb_index[name] for name in joint_names_ikpy]


def check_self_collision(full_q):
    for pb_idx, angle in zip(pb_indices_in_order, [full_q[i] for i in active_indices]):
        p.resetJointState(robot_id, pb_idx, angle)
    p.performCollisionDetection()
    return len(p.getContactPoints(bodyA=robot_id, bodyB=robot_id)) > 0


def within_joint_limits(full_q):
    for i, idx in enumerate(active_indices):
        lo, hi = JOINT_LIMITS_RAD[i]
        if not (lo - 1e-6 <= full_q[idx] <= hi + 1e-6):
            return False, i
    return True, None


# ---------------------------------------------------------
# 3. 자세 변환 (실측 검증 완료: scipy 'xyz' 외재적 = pymycobot rx,ry,rz)
# ---------------------------------------------------------
def matrix_to_rxryrz(rotation_matrix):
    return R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)


# ---------------------------------------------------------
# 4. 야코비안/특이점
# ---------------------------------------------------------
def numerical_jacobian(full_q, delta=1e-4):
    base_pos = chain.forward_kinematics(full_q)[:3, 3]
    J = np.zeros((3, n_active))
    for i, idx in enumerate(active_indices):
        q_perturbed = list(full_q)
        q_perturbed[idx] += delta
        pos = chain.forward_kinematics(q_perturbed)[:3, 3]
        J[:, i] = (pos - base_pos) / delta
    return J


def numerical_jacobian_6d(full_q, delta=1e-4):
    """위치(3) + 자세(3, 회전벡터) 를 모두 포함한 6xN 야코비안"""
    fk0 = chain.forward_kinematics(full_q)
    p0 = fk0[:3, 3]
    R0 = fk0[:3, :3]
    J = np.zeros((6, n_active))
    for i, idx in enumerate(active_indices):
        qp = list(full_q)
        qp[idx] += delta
        fk = chain.forward_kinematics(qp)
        J[:3, i] = (fk[:3, 3] - p0) / delta
        dR = fk[:3, :3] @ R0.T
        J[3:, i] = R.from_matrix(dR).as_rotvec() / delta
    return J


def solve_pose_ik(q_seed_full, target_mm, target_R,
                  pos_tol_mm=POS_TOL_MM, orient_tol_deg=ORIENT_TOL_DEG,
                  max_iters=40, damping=0.05):
    """
    자체 구현 6자유도 IK: 감쇠최소자승(damped least squares) 반복.
    ikpy의 orientation_mode="all"이 자세만 우선시하고 위치를 포기하는
    고질적 문제가 있어, 위치+자세를 균형있게 동시에 수렴시키는 방식으로 대체.
    반환: (full_q 또는 None, 최종 위치오차mm, 최종 자세오차deg)
    """
    q = list(q_seed_full)
    target_p_m = np.array(target_mm) / 1000.0
    last_pos_err = last_ori_err = float("inf")

    for _ in range(max_iters):
        fk = chain.forward_kinematics(q)
        p = fk[:3, 3]
        Rc = fk[:3, :3]
        e_pos = target_p_m - p                                # 미터 단위
        e_rot = R.from_matrix(target_R @ Rc.T).as_rotvec()    # 라디안 단위

        last_pos_err = np.linalg.norm(e_pos) * 1000.0
        last_ori_err = math.degrees(np.linalg.norm(e_rot))
        if last_pos_err <= pos_tol_mm and last_ori_err <= orient_tol_deg:
            return q, last_pos_err, last_ori_err

        err = np.concatenate([e_pos, e_rot])
        J = numerical_jacobian_6d(q)
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(6), err)
        for i, idx in enumerate(active_indices):
            q[idx] += dq[i] * 0.6  # 안정성을 위해 한 번에 60%만 이동

    return None, last_pos_err, last_ori_err


def jacobian_condition_number(J):
    sv = np.linalg.svd(J, compute_uv=False)
    if sv[-1] < 1e-9:
        return float("inf")
    return sv[0] / sv[-1]


# ---------------------------------------------------------
# 5. 데드존 파일 로드 (1차 필터 + 검증된 자세 시드 + 추천좌표)
# ---------------------------------------------------------
try:
    _g = np.load(GRID_FILE_PATH)
    _reachable_grid = _g["reachable_grid"]
    _collision_only_grid = _g["collision_only_grid"]
    _best_angles_2d = _g["best_angles_2d"]
    _x_min, _y_min, _z_min = float(_g["x_min"]), float(_g["y_min"]), float(_g["z_min"])
    _r_min = float(_g["r_min"])
    _step = float(_g["step"])
    _nx, _ny, _nz = int(_g["nx"]), int(_g["ny"]), int(_g["nz"])
    _nr = int(_g["nr"])
    _grid_loaded = True

    _safe_mask = _reachable_grid & (~_collision_only_grid)
    _z_coords = _z_min + np.arange(_nz) * _step
    _safe_mask[:, :, _z_coords < DESK_SAFETY_MARGIN_MM] = False
    _safe_indices = np.argwhere(_safe_mask)
except Exception as e:
    print(f"⚠️ 데드존 파일 로드 실패 (1차 필터 없이 진행): {e}")
    _grid_loaded = False


def quick_prefilter(x, y, z):
    if not _grid_loaded:
        return True, "필터 없음"
    ix = int(round((x - _x_min) / _step))
    iy = int(round((y - _y_min) / _step))
    iz = int(round((z - _z_min) / _step))
    if not (0 <= ix < _nx and 0 <= iy < _ny and 0 <= iz < _nz):
        return False, "격자 범위 밖 (명백히 도달 불가)"
    if not _reachable_grid[ix, iy, iz]:
        return False, "명백히 도달 불가능 지점"
    if z < DESK_SAFETY_MARGIN_MM:
        return False, "책상 아래 (데드존)"
    return True, "1차 필터 통과"


def suggest_nearest_safe(x, y, z):
    if not _grid_loaded or len(_safe_indices) == 0:
        return None
    t = np.array([int(round((x - _x_min) / _step)),
                  int(round((y - _y_min) / _step)),
                  int(round((z - _z_min) / _step))])
    d = np.linalg.norm(_safe_indices - t, axis=1)
    n = _safe_indices[np.argmin(d)]
    return (_x_min + n[0] * _step, _y_min + n[1] * _step, _z_min + n[2] * _step)


# ---------------------------------------------------------
# 6. 직선경로 안전 검증 (검증 전용 - 실행에는 절대 사용 안 함)
# ---------------------------------------------------------
def validate_line(q_start_full, R_start, R_end, start_mm, end_mm):
    """
    start->end 직선을, 자세는 R_start에서 R_end로 서서히 보간(slerp)하며 검증.
    (실제 펌웨어의 직선 이동도 자세를 고정하는 게 아니라 목표 자세로
     회전하면서 이동하므로, 이 방식이 실제 동작과 일치함)
    반환: (안전여부, 실패사유)
    """
    start = np.array(start_mm, dtype=float)
    end = np.array(end_mm, dtype=float)
    dist = np.linalg.norm(end - start)
    if dist < 1e-6:
        return True, "이동거리 0"
    n_wp = max(1, int(round(dist / CHECK_STEP_MM)))

    slerp = Slerp([0.0, 1.0], R.from_matrix(np.stack([R_start, R_end])))

    q_prev = list(q_start_full)

    for wi in range(1, n_wp + 1):
        t = wi / n_wp
        target_mm = start + (end - start) * t
        target_R = slerp([t]).as_matrix()[0]

        sol, pos_err, orient_err = solve_pose_ik(q_prev, target_mm, target_R)

        if sol is None:
            return False, (f"웨이포인트 {wi}/{n_wp} ({target_mm.round(0).tolist()}) 수렴 실패 "
                            f"(위치 {pos_err:.1f}mm / 자세 {orient_err:.1f}도)")

        ok, bad = within_joint_limits(sol)
        if not ok:
            return False, f"웨이포인트 {wi}/{n_wp}에서 관절 {bad+1} 한계 초과"

        if jacobian_condition_number(numerical_jacobian(sol)) > CONDITION_NUMBER_MAX:
            return False, f"웨이포인트 {wi}/{n_wp} ({target_mm.round(0).tolist()}) 특이점 근접"

        if check_self_collision(sol):
            return False, f"웨이포인트 {wi}/{n_wp} ({target_mm.round(0).tolist()}) 자가충돌"

        if target_mm[2] < DESK_SAFETY_MARGIN_MM:
            return False, f"웨이포인트 {wi}/{n_wp} 책상 충돌 위험 (z={target_mm[2]:.0f}mm)"

        q_prev = sol

    return True, "직선 안전"


def ik_at(target_mm, orientation_matrix, q_seed):
    """특정 지점의 자세고정 IK 해 하나를 구함 (경유점 검증 이어가기용)"""
    sol, _, _ = solve_pose_ik(q_seed, np.array(target_mm, dtype=float), orientation_matrix)
    return sol


# ---------------------------------------------------------
# 7. 경유점(via-point) 탐색 - 직선이 안 될 때의 '부드러운' 우회
# ---------------------------------------------------------
def grid_seed_pose(target_mm):
    """데드존 파일에 저장된 '검증된 자세'를 J1만 방향 보정해서 시드로 반환"""
    if not _grid_loaded:
        return None
    x, y, z = target_mm
    r = math.sqrt(x**2 + y**2)
    ir = int(round((r - _r_min) / _step))
    iz = int(round((z - _z_min) / _step))
    if not (0 <= ir < _nr and 0 <= iz < _nz):
        return None
    base_deg = _best_angles_2d[ir, iz, :]
    if np.any(np.isnan(base_deg)):
        return None
    theta_deg = math.degrees(math.atan2(y, x))
    final_deg = base_deg.copy()
    final_deg[0] = ((final_deg[0] + theta_deg + 180) % 360) - 180
    full_q = [0.0] * len(chain.links)
    for i, idx in enumerate(active_indices):
        full_q[idx] = math.radians(final_deg[i])
    return full_q


def natural_pose_at(target_mm, q_seed):
    """목표점에서 '자연스럽게 도달 가능한' 자세를 찾음.
    여러 시드(현재자세 -> 데드존 저장자세)를 순서대로 시도해서 성공률을 높임.
    반환: (full_q, R_target) 또는 (None, None)"""
    seeds = [q_seed]
    gs = grid_seed_pose(target_mm)
    if gs is not None:
        seeds.append(gs)

    for seed in seeds:
        # 시드 자체가 이미 목표에 맞고 안전하면 그대로 사용 (grid seed의 경우)
        fk = chain.forward_kinematics(seed)
        if np.linalg.norm(fk[:3, 3] * 1000.0 - np.array(target_mm)) <= POS_TOL_MM:
            ok, _ = within_joint_limits(seed)
            if ok and not check_self_collision(seed):
                return list(seed), fk[:3, :3]

        # 시드에서 위치 IK 수행
        try:
            sol = chain.inverse_kinematics(
                target_position=np.array(target_mm) / 1000.0,
                initial_position=seed,
            )
        except Exception:
            continue
        fk = chain.forward_kinematics(sol)
        if np.linalg.norm(fk[:3, 3] * 1000.0 - np.array(target_mm)) > POS_TOL_MM:
            continue
        ok, _ = within_joint_limits(sol)
        if not ok or check_self_collision(sol):
            continue
        return sol, fk[:3, :3]

    return None, None


def find_via_point(q_start_full, R_start, start_mm, end_mm, R_end):
    """
    A->B 직선이 위험할 때, A->C, C->B 두 직선이 모두 안전한 경유점 C를 찾는다.
    """
    start = np.array(start_mm, dtype=float)
    end = np.array(end_mm, dtype=float)
    mid = (start + end) / 2.0

    direction = end - start
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return None
    direction /= dist

    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(direction, up)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    else:
        side /= np.linalg.norm(side)

    candidate_dirs = [up, side, -side, (up + side) / np.linalg.norm(up + side),
                      (up - side) / np.linalg.norm(up - side)]

    for offset in VIA_OFFSETS_MM:
        for d in candidate_dirs:
            via = mid + d * offset
            if via[2] < DESK_SAFETY_MARGIN_MM + 10:
                continue
            ok, _ = quick_prefilter(*via)
            if not ok:
                continue

            # C 지점의 자연스러운 자세 확보
            q_via, R_via = natural_pose_at(via.tolist(), q_start_full)
            if q_via is None:
                continue

            # A->C 검증 (자세: R_start -> R_via 보간)
            ok1, _ = validate_line(q_start_full, R_start, R_via, start_mm, via.tolist())
            if not ok1:
                continue
            # C->B 검증 (자세: R_via -> R_end 보간)
            ok2, _ = validate_line(q_via, R_via, R_end, via.tolist(), end_mm)
            if ok2:
                return via.tolist(), R_via

    return None, None


# ---------------------------------------------------------
# 8. 이동 실행 - 오직 send_coords(mode=1)만 사용
# ---------------------------------------------------------
def move_linear(mc, target_mm, rx, ry, rz, speed=LINEAR_SPEED):
    """펌웨어 직선보간 이동. 실제 도달 여부까지 확인해서 (성공여부, 샘플경로)를 반환.
    이동 중 실제 좌표를 주기적으로 샘플링해 경로 시각화에 사용."""
    pose = [target_mm[0], target_mm[1], target_mm[2], rx, ry, rz]
    mc.send_coords(pose, speed, 1)
    time.sleep(SETTLE_DELAY_SEC)

    samples = []
    t0 = time.time()
    while mc.is_moving():
        if time.time() - t0 > MOVE_TIMEOUT_SEC:
            break
        c = mc.get_coords()
        if isinstance(c, list) and len(c) >= 3:
            samples.append(c[:3])
        time.sleep(TRAJ_SAMPLE_SEC)

    # 실측 검증: 실제로 목표 근처에 도달했는가
    time.sleep(0.3)
    coords = mc.get_coords()
    if isinstance(coords, list) and len(coords) >= 3:
        samples.append(coords[:3])
        arrival_err = math.dist(coords[:3], list(target_mm[:3]))
        if arrival_err <= 25:  # 도달 판정 여유
            return True, samples

    # 도달 실패 - 펌웨어 에러 확인 및 클리어
    err = mc.get_error_information()
    if err not in (0, -1):
        print(f"   (펌웨어 에러 {err} 감지 - 직선 이동이 펌웨어에서 거부됨. 에러 클리어)")
        mc.clear_error_information()
        time.sleep(0.3)
    return False, samples


def move_joint(mc, q_full, speed=SPEED):
    """관절 이동 (직선 아님). 이동 중 실제 좌표를 샘플링해서 반환."""
    angles_deg = [math.degrees(q_full[i]) for i in active_indices]
    mc.send_angles(angles_deg, speed)
    time.sleep(SETTLE_DELAY_SEC)
    samples = []
    t0 = time.time()
    while mc.is_moving():
        if time.time() - t0 > MOVE_TIMEOUT_SEC:
            break
        c = mc.get_coords()
        if isinstance(c, list) and len(c) >= 3:
            samples.append(c[:3])
        time.sleep(TRAJ_SAMPLE_SEC)
    c = mc.get_coords()
    if isinstance(c, list) and len(c) >= 3:
        samples.append(c[:3])
    return samples


def plot_path(samples, start_mm, target_mm, title_tag):
    """실제 경로를 XY / YZ / ZX 세 평면 + '직선으로부터의 이탈거리' 그래프로 저장/표시.
    빨간 점선 = 이상적인 직선, 파란 실선 = 실제 측정 경로."""
    if not PLOT_TRAJECTORY or len(samples) < 2:
        return
    arr = np.array(samples, dtype=float)
    s = np.array(start_mm, dtype=float)
    t = np.array(target_mm, dtype=float)

    # --- 각 샘플점에서 이상적인 직선(s->t)까지의 수직 거리 계산 ---
    direction = t - s
    dir_norm = np.linalg.norm(direction)
    if dir_norm > 1e-6:
        u = direction / dir_norm
        deviations = np.linalg.norm(np.cross(arr - s, u), axis=1)
    else:
        deviations = np.linalg.norm(arr - s, axis=1)
    max_dev = deviations.max()
    mean_dev = deviations.mean()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    planes = [(0, 1, "X", "Y"), (1, 2, "Y", "Z"), (0, 2, "X", "Z")]
    for ax, (i, j, ni, nj) in zip(axes[:3], planes):
        ax.plot([s[i], t[i]], [s[j], t[j]], "r--", linewidth=1.5, label="ideal line")
        ax.plot(arr[:, i], arr[:, j], "b.-", markersize=4, linewidth=1, label="actual path")
        ax.plot(s[i], s[j], "go", markersize=8, label="start")
        ax.plot(t[i], t[j], "r*", markersize=12, label="target")
        ax.set_xlabel(f"{ni} (mm)")
        ax.set_ylabel(f"{nj} (mm)")
        ax.set_title(f"{ni}{nj} plane")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(loc="best", fontsize=8)

    # --- 4번째 패널: 직선으로부터의 이탈거리(mm) ---
    ax4 = axes[3]
    ax4.plot(range(len(deviations)), deviations, "m.-", markersize=4, linewidth=1)
    ax4.axhline(max_dev, color="orange", linestyle="--", linewidth=1,
                label=f"max = {max_dev:.2f}mm")
    ax4.axhline(mean_dev, color="gray", linestyle=":", linewidth=1,
                label=f"mean = {mean_dev:.2f}mm")
    ax4.set_xlabel("sample index")
    ax4.set_ylabel("deviation from ideal line (mm)")
    ax4.set_title("Deviation from straight line")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best", fontsize=8)
    ax4.set_ylim(bottom=0)

    fig.suptitle(f"Path check: {title_tag}  (max dev {max_dev:.2f}mm / mean dev {mean_dev:.2f}mm)")
    fig.tight_layout()

    folder = SESSION_DIR if SESSION_DIR else "."
    fname = os.path.join(folder, f"path_{datetime.now().strftime('%H%M%S')}_{title_tag}.png")
    plt.savefig(fname, dpi=120)
    print(f"📈 경로 그래프 저장됨: {fname}  (최대 이탈 {max_dev:.2f}mm / 평균 이탈 {mean_dev:.2f}mm)")
    plt.show(block=False)
    plt.pause(0.5)


def get_current_q_full(mc):
    angles_deg = mc.get_angles()
    full_q = [0.0] * len(chain.links)
    for i, idx in enumerate(active_indices):
        full_q[idx] = math.radians(angles_deg[i])
    return full_q


# ---------------------------------------------------------
# 9. 메인 루프
# ---------------------------------------------------------
def main():
    global SESSION_DIR
    if PLOT_TRAJECTORY:
        SESSION_DIR = create_session_dir()
        print(f"📁 이번 실행의 경로 그래프는 여기 저장됩니다: {SESSION_DIR}/")

    mc, port = find_robot_port()
    if mc is None:
        print("❌ 응답하는 로봇 포트를 찾지 못했습니다. USB 연결과 로봇 전원, "
              "화면의 Transponder(USB) 모드 진입 여부를 확인해주세요.")
        return
    mc.power_on()          # 재부팅 후 서보가 비활성 상태면 이동 명령이 조용히 무시됨 - 반드시 활성화
    time.sleep(2)
    mc.clear_error_information()  # 이전 세션에서 남은 에러(33 등)가 있으면 명령이 거부되므로 클리어
    time.sleep(0.5)
    mc.set_fresh_mode(0)  # 보간 모드 (이전 버전과 동일하게 유지)

    print("🤖 로봇 시작: 홈 자세 이동")
    mc.send_angles(ALIGN_ANGLES, SPEED)
    time.sleep(SETTLE_DELAY_SEC)
    t0 = time.time()
    while mc.is_moving():
        if time.time() - t0 > MOVE_TIMEOUT_SEC:
            break
        time.sleep(0.1)
    print("✅ 홈 자세 도착.")

    try:
        last_target_mm = None
        first_move_done = False  # 첫 이동은 직선 검증 없이 관절 이동으로 처리
        while True:
            # --- 목표좌표 입력 및 검증 ---
            target_mm = None
            while target_mm is None:
                if last_target_mm is not None:
                    print(f"(이전 좌표: {last_target_mm[0]:.0f} {last_target_mm[1]:.0f} {last_target_mm[2]:.0f})")
                raw = input("이동할 좌표 입력 (X Y Z, q:종료): ")
                if raw.lower() == "q":
                    return
                try:
                    cand = list(map(float, raw.split()))
                    if len(cand) != 3:
                        raise ValueError
                except ValueError:
                    print("⚠️ 'X Y Z' 형식으로 숫자 3개를 입력해주세요.")
                    continue

                ok, reason = quick_prefilter(*cand)
                if not ok:
                    print(f"❌ 좌표 불가: {reason}")
                    s = suggest_nearest_safe(*cand)
                    if s:
                        print(f"   💡 추천 좌표: {s[0]:.0f} {s[1]:.0f} {s[2]:.0f}")
                    continue
                target_mm = cand

            # --- 현재 상태 파악 ---
            q_cur = get_current_q_full(mc)
            fk = chain.forward_kinematics(q_cur)
            cur_mm = (fk[:3, 3] * 1000.0).tolist()
            ori_matrix = fk[:3, :3]
            rx, ry, rz = matrix_to_rxryrz(ori_matrix)

            # --- 첫 이동: 직선 검증 생략, 안전한 관절 이동으로 작업영역 진입 ---
            if not first_move_done:
                print("➡️ 첫 이동은 관절 이동으로 작업영역에 진입합니다 (직선 검증 생략)...")
                q_first = ik_at(target_mm, ori_matrix, q_cur)
                if q_first is None:
                    # 자세 유지가 안 되면 자세 자유로 재시도
                    try:
                        q_first = chain.inverse_kinematics(
                            target_position=np.array(target_mm) / 1000.0,
                            initial_position=q_cur,
                        )
                    except Exception:
                        q_first = None
                if q_first is None or check_self_collision(q_first) or not within_joint_limits(q_first)[0]:
                    print("❌ 첫 이동 목표의 안전한 자세를 찾지 못했습니다. 다른 좌표를 시도해주세요.")
                    continue
                samples_first = move_joint(mc, q_first)
                print("✅ 첫 이동 완료 (관절 이동). 다음부터는 직선 이동을 시도합니다.")
                plot_path(samples_first, cur_mm, target_mm, "first_joint")
                first_move_done = True
                last_target_mm = target_mm
                continue

            # --- 목표점의 자연스러운 자세 찾기 (직선 검증의 끝 자세로 사용) ---
            q_end, R_end = natural_pose_at(target_mm, q_cur)
            if q_end is None:
                print("❌ 이 좌표의 도달 자세를 계산하지 못했습니다 (격자상 도달 가능하지만 IK 실패).")
                print("   다른 좌표를 시도하거나, 현재 위치에서 더 가까운 중간 지점을 거쳐 이동해보세요.")
                continue
            end_rx, end_ry, end_rz = matrix_to_rxryrz(R_end)

            # --- 직선 검증 (자세: 현재 -> 목표 자연자세로 보간) ---
            print("📐 직선경로 검증 중...")
            ok, reason = validate_line(q_cur, ori_matrix, R_end, cur_mm, target_mm)

            if ok:
                print("✅ 직선 이동 가능. 부드럽게 한 번에 이동합니다...")
                moved_ok, samples = move_linear(mc, target_mm, end_rx, end_ry, end_rz)
                if moved_ok:
                    print("✅ 이동 완료 (완전 직선).")
                    plot_path(samples, cur_mm, target_mm, "straight")
                else:
                    print("⚠️ 펌웨어가 직선 이동을 거부했습니다. 관절 이동으로 대신 이동합니다 (직선 아님)...")
                    samples_j = move_joint(mc, q_end)
                    print("✅ 이동 완료 (관절 이동으로 대체됨).")
                    plot_path(samples + samples_j, cur_mm, target_mm, "joint_fallback")
                last_target_mm = target_mm
                continue

            print(f"⚠️ 직선 불가: {reason}")
            print("🔍 부드러운 우회를 위한 경유점 탐색 중...")
            via, R_via = find_via_point(q_cur, ori_matrix, cur_mm, target_mm, R_end)

            if via is None:
                print("⚠️ 직선/경유점 경로가 모두 불가능합니다 (예: 로봇 뒤쪽 사각지대를 가로지르는 경로).")
                print("⚠️ 관절 이동으로 대신 이동합니다 — 손끝이 직선을 그리지 않고 크게 돌아갈 수 있으니 주변 공간을 확인하세요. ⚠️")
                samples_j = move_joint(mc, q_end)
                print("✅ 이동 완료 (관절 이동, 직선 아님).")
                plot_path(samples_j, cur_mm, target_mm, "joint_only")
                last_target_mm = target_mm
                continue

            via_rx, via_ry, via_rz = matrix_to_rxryrz(R_via)
            print(f"↪ 경유점 발견: ({via[0]:.0f}, {via[1]:.0f}, {via[2]:.0f})")
            print("⚠️ 완전 직선이 아닌 [직선 2개 연결] 경로로 이동합니다 (경유점에서 잠깐 정지). ⚠️")
            # 경유점 경로는 구간이 짧아 가감속 때문에 체감속도가 떨어지므로 최대 속도로 보상
            leg1_ok, samples1 = move_linear(mc, via, via_rx, via_ry, via_rz, speed=100)
            if not leg1_ok:
                print("⚠️ 경유점까지의 직선이 펌웨어에서 거부됨. 관절 이동으로 목표까지 이동합니다 (직선 아님)...")
                samples_j = move_joint(mc, q_end)
                print("✅ 이동 완료 (관절 이동으로 대체됨).")
                plot_path(samples1 + samples_j, cur_mm, target_mm, "via_fail_joint")
                last_target_mm = target_mm
                continue
            leg2_ok, samples2 = move_linear(mc, target_mm, end_rx, end_ry, end_rz, speed=100)
            if not leg2_ok:
                print("⚠️ 두 번째 직선이 펌웨어에서 거부됨. 관절 이동으로 마무리합니다 (직선 아님)...")
                samples_j = move_joint(mc, q_end)
                samples2 = samples2 + samples_j
            print("✅ 이동 완료 (경유점 경유).")
            plot_path(samples1 + samples2, cur_mm, target_mm, "via")
            last_target_mm = target_mm

    except KeyboardInterrupt:
        print("\n⚠️ 강제 종료 신호 감지!")
    finally:
        print("\n🤖 [안전 복귀] 홈 자세로 복귀합니다...")
        mc.send_angles(ALIGN_ANGLES, SPEED)
        time.sleep(SETTLE_DELAY_SEC)
        while mc.is_moving():
            time.sleep(0.1)
        print("✅ 시스템 안전 종료.")


if __name__ == "__main__":
    main()