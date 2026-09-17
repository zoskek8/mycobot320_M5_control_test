"""
myCobot 320 기구학/안전검증 라이브러리
=====================================

기존 mycobot_smooth_planner.py에서 대화형 CLI 부분(좌표 입력 루프, 경유점 탐색,
이동 실행, 그래프 저장 등)을 걷어내고, mycobot_path_editor.py가 실제로 쓰는
기능만 남긴 라이브러리 모듈입니다.

제공 기능:
    - URDF 기반 순기구학(FK) / 자체 6자유도 IK 솔버(감쇠최소자승)
    - PyBullet 자가충돌 검사, 관절한계 검사
    - 야코비안 / 특이점 판정(조건수)
    - 데드존(.npz) 로드 및 빠른 안전성 조회
    - 로봇 포트 자동 탐색, 현재 관절상태 읽기

이 파일은 단독 실행용이 아닙니다. mycobot_path_editor.py 에서 import 해서 씁니다.

사전 설치:
    pip3 install ikpy pybullet numpy scipy
"""

import re
import glob
import time
import math
import struct
import threading
import numpy as np
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
import pybullet as p
from pymycobot import MyCobot320

# ---------------------------------------------------------
# 0. 설정
# ---------------------------------------------------------
URDF_PATH = "/home/kms/colcon_ws/src/mycobot_ros2/mycobot_description/urdf/mycobot_320_m5_2022/mycobot_320_m5_2022.urdf"
PACKAGE_ROOT = "/home/kms/colcon_ws/src/mycobot_ros2/mycobot_description"
GRID_FILE_PATH = "/home/kms/바탕화면/python_code/mycobot_reachability_grid.npz"

BAUD_RATE = 115200
DESK_SAFETY_MARGIN_MM = 20
ALIGN_ANGLES = [0, -30, 0, 0, 30, 0]   # 정렬(준비) 자세 - 전부 0도인 특이점을 피한 자세
# [8차 세션, §23.10] 기존 [0,-30,30,0,30,0]에서 J3만 0으로 낮췄다. 진동
# 조사(§23) 중 이 조합(J3 저부하 비교자세)이 여러 번 측정에서 거의
# 완전히 깨끗했던 것과 반대로, 기존 값(J3=30)은 팔을 뻗어 중력을 직접
# 떠받치는 자세라 정지 중에도 백래시 dither가 지속됐다(§23.4/§23.7).
# [주의] 이 상수는 mycobot_curve_math.py의 ik_seed_q()에서 인터랙티브
# 편집(점 드래그로 데드존 회피) 시 대체점 검증의 고정 시드로도 쓰인다 -
# 값이 바뀌면 그 검증이 미세하게 다른 지점에서 수렴을 시작한다(주로 이
# 관절 하나만 30도 차이). 실기에서 곡선 편집이 이상 없이 잘 되는지
# 확인 필요 - 문제가 보이면 이 상수만 원래 값 [0,-30,30,0,30,0]으로
# 되돌리면 된다(다른 코드는 안 건드려도 됨).

# [중요] 관절 한계는 모델마다 다르다. 특히 MyCobot320의 J2는 ±135도로,
#        다른 관절(±165)보다 좁다. 하드코딩하면 틀리기 쉽다.
#
# [2026-08 확인] pymycobot.robot_limit 서브모듈은 실제 배포판에 존재하지 않는다
#   (ModuleNotFoundError, 모든 버전에서 재현됨). 이 이름으로 값을 읽어오려던
#   시도는 애초에 성립할 수 없는 죽은 분기였다 - 항상 except로 빠져 아래 보수적
#   기본값을 썼을 뿐이다. 다행히 이 기본값이 실측 하드리밋(Figure 4 점선)과 일치해
#   결과적으로는 맞는 값을 쓰고 있었다.
#
# 정공법: robot_limit이라는 존재하지 않는 '내부 표'를 훔쳐보지 말고, 연결된
#   로봇에게 get_joint_min_angle()/get_joint_max_angle()로 직접 물어본다.
#   연결 전에는 이 기본값을 쓰고, find_robot_port()가 연결에 성공하면
#   refresh_joint_limits_from_robot()가 아래 두 리스트를 '제자리에서' 갱신한다
#   (리바인딩이 아니라 슬라이스 대입 - editor.py가 이미 참조 중인 같은 리스트
#   객체를 공유하므로, import 시점이 갱신보다 앞서도 최신값이 자동 반영된다).
JOINT_LIMITS_DEG = [(-165, 165), (-135, 135), (-145, 145), (-148, 148), (-165, 165), (-180, 180)]
JOINT_LIMITS_RAD = [(math.radians(lo), math.radians(hi)) for lo, hi in JOINT_LIMITS_DEG]
print("관절 한계(도) [기본값, 연결 시 실측으로 갱신 예정]:", JOINT_LIMITS_DEG)


def refresh_joint_limits_from_robot(mc):
    """연결된 로봇에게 관절 한계를 실측으로 물어보고 JOINT_LIMITS_DEG/RAD를 갱신한다.
    한 관절이라도 읽기가 실패하면 그 관절만 기존 값(기본값 또는 이전 실측값)을 유지한다.
    반환: 6개 전부 성공했으면 True.
    """
    ok_count = 0
    new_deg = list(JOINT_LIMITS_DEG)
    for i in range(6):
        try:
            lo = mc.get_joint_min_angle(i + 1)
            hi = mc.get_joint_max_angle(i + 1)
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo < hi:
                new_deg[i] = (float(lo), float(hi))
                ok_count += 1
        except Exception:
            pass   # 이 관절은 기존 값 유지

    JOINT_LIMITS_DEG[:] = new_deg                      # 제자리 갱신 (리바인딩 금지)
    JOINT_LIMITS_RAD[:] = [(math.radians(lo), math.radians(hi)) for lo, hi in new_deg]

    if ok_count == 6:
        print("✅ 관절 한계(도) 실측으로 갱신:", JOINT_LIMITS_DEG)
    else:
        print(f"⚠️ 관절 한계 실측 {ok_count}/6개만 성공 - 나머지는 기본값 유지:", JOINT_LIMITS_DEG)
    return ok_count == 6

SPEED = 50                  # 관절 이동(send_angles) 기본 속도
POS_TOL_MM = 15             # IK 수렴 허용 오차
ORIENT_TOL_DEG = 5
CONDITION_NUMBER_MAX = 90   # 특이점 근접 판정 (야코비안 조건수)
SETTLE_DELAY_SEC = 0.3      # 명령 직후 is_moving() 확인 전 유예시간
MOVE_TIMEOUT_SEC = 30
TRAJ_SAMPLE_SEC = 0.08      # 이동 중 좌표 샘플링 간격

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
    """주어진 관절자세에서 로봇이 자기 자신과 충돌하는지 검사"""
    for pb_idx, angle in zip(pb_indices_in_order, [full_q[i] for i in active_indices]):
        p.resetJointState(robot_id, pb_idx, angle)
    p.performCollisionDetection()
    return len(p.getContactPoints(bodyA=robot_id, bodyB=robot_id)) > 0


def within_joint_limits(full_q):
    """관절 한계 이내인지 검사. 반환: (통과여부, 위반한 관절 인덱스)"""
    for i, idx in enumerate(active_indices):
        lo, hi = JOINT_LIMITS_RAD[i]
        if not (lo - 1e-6 <= full_q[idx] <= hi + 1e-6):
            return False, i
    return True, None


# ---------------------------------------------------------
# 3. 자세 변환 (실측 검증 완료: scipy 'xyz'(외재적) = pymycobot rx,ry,rz)
# ---------------------------------------------------------
def matrix_to_rxryrz(rotation_matrix):
    return R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)


# ---------------------------------------------------------
# 4. 야코비안 / 특이점 / 자체 6자유도 IK 솔버
# ---------------------------------------------------------
def numerical_jacobian(full_q, delta=1e-4):
    """위치(3) 전용 야코비안 - 특이점 판정에 사용"""
    base_pos = chain.forward_kinematics(full_q)[:3, 3]
    J = np.zeros((3, n_active))
    for i, idx in enumerate(active_indices):
        q_perturbed = list(full_q)
        q_perturbed[idx] += delta
        pos = chain.forward_kinematics(q_perturbed)[:3, 3]
        J[:, i] = (pos - base_pos) / delta
    return J


def numerical_jacobian_6d(full_q, delta=1e-4):
    """위치(3) + 자세(3, 회전벡터) 를 모두 포함한 6xN 야코비안

    [9차 세션, §24 확인] 실측 프로파일링 결과 solve_pose_ik 내부 반복마다
    이 함수가 불려서 관절 6개 x (기준1+섭동6) = 웨이포인트당 FK 호출의
    56%(전체 검증시간의 절반 이상)를 차지했다. 이제 solve_pose_ik는
    이 함수 대신 analytic_jacobian_6d()를 쓴다(아래) - 이 함수 자체는
    지우지 않았다: (1) analytic_jacobian_6d 검증의 정답지 역할,
    (2) 혹시 이후 URDF가 바뀌어 analytic 쪽이 어긋나는 걸 의심할 때
    바로 대조할 수 있는 폴백."""
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


def analytic_jacobian_6d(full_q):
    """위치(3) + 자세(3, 회전벡터) 6xN 야코비안 - 해석적(기하) 공식.

    [9차 세션, §24] numerical_jacobian_6d를 유한차분(관절마다 섭동 후 재계산,
    FK 7회/웨이포인트당 최대 68회까지 실측)으로 구했었는데, 회전관절만 있는
    직렬 체인은 FK **딱 1번**으로 전체 링크 프레임을 받아 닫힌 형태로 계산할
    수 있다 - 근사가 아니라 유한차분보다 오히려 정확한 해석해다.

    회전관절 i에 대해 (표준 기하 야코비안 공식):
        J_v[:, i] = z_i x (p_e - o_i)      (위치 야코비안 열)
        J_w[:, i] = z_i                    (자세 야코비안 열)
    z_i: 관절 i의 회전축(월드좌표), o_i: 관절 i의 원점(월드좌표), p_e: 말단 위치.

    z_i/o_i는 chain.forward_kinematics(q, full_kinematics=True)가 주는
    "관절 i의 자기 회전까지 반영된" 프레임에서 뽑는다 - 자기 축을 중심으로
    도는 회전은 그 축의 방향 자체는 바꾸지 않으므로(회전축은 자기 자신의
    고유벡터) 이 시점 프레임을 써도 정확하다.

    [검증, 2026-08-19] 실제 URDF로 mycobot_jacobian_verify.py [1]을 돌려
    무작위 자세 500개에서 numerical_jacobian_6d와 대조 - 위치오차 1.8e-5mm,
    자세오차 2.6e-10도(유한차분 자체의 이산화오차 수준, 사실상 일치).
    실기 곡선 실행으로 cross-track/final error가 그대로인 것도 확인했다.
    """
    frames = chain.forward_kinematics(full_q, full_kinematics=True)
    p_e = frames[-1][:3, 3]
    J = np.zeros((6, n_active))
    for col, idx in enumerate(active_indices):
        R_i = frames[idx][:3, :3]
        local_axis = chain.links[idx].rotation   # URDFLink 속성명 - .joint_axis 아님
        axis_world = R_i @ local_axis
        origin_world = frames[idx][:3, 3]
        d = p_e - origin_world
        # [9차 세션, §24 후속] np.cross(axis_world, d) 대신 성분별 외적 공식을
        # 직접 푼다. 실측(mycobot_validate_profile.py)에서 np.cross 17,850회
        # 호출이 0.705초를 먹었다 - 3원소 벡터 외적 자체는 순식간인데,
        # np.cross가 매번 거치는 범용 디스패치(__array_function__) 오버헤드가
        # 그 짧은 계산보다 훨씬 컸다(작은 배열엔 흔한 numpy 함정). 아래는
        # 수학적으로 완전히 동일한 값을 준다 - 그냥 그 경로를 안 거칠 뿐이다.
        J[0, col] = axis_world[1] * d[2] - axis_world[2] * d[1]
        J[1, col] = axis_world[2] * d[0] - axis_world[0] * d[2]
        J[2, col] = axis_world[0] * d[1] - axis_world[1] * d[0]
        J[3:, col] = axis_world
    return J


# [9차 세션, §24] 실기 검증 완료 - mycobot_jacobian_verify.py [1] 500개 무작위
# 자세에서 위치오차 1.8e-5mm / 자세오차 2.6e-10° (수치미분의 이산화오차 수준,
# 사실상 완전히 일치) 확인 후 True로 전환. 문제 생기면 이 줄만 False로
# 되돌리면 된다(다른 코드는 무관 - solve_pose_ik가 이 스위치만 본다).
USE_ANALYTIC_JACOBIAN = True


def jacobian_condition_number(J):
    """야코비안 특이값 비율(최대/최소). 단위에 무관하며 클수록 특이점에 가까움."""
    sv = np.linalg.svd(J, compute_uv=False)
    if sv[-1] < 1e-9:
        return float("inf")
    return sv[0] / sv[-1]


def solve_pose_ik(q_seed_full, target_mm, target_R,
                  pos_tol_mm=POS_TOL_MM, orient_tol_deg=ORIENT_TOL_DEG,
                  max_iters=60, damping=0.05,
                  stall_patience=3, stall_rel=0.05, target_frac=0.5):
    """
    자체 6자유도 IK: 감쇠최소자승(damped least squares) 반복.
    위치 오차와 자세 오차를 하나의 6차원 벡터로 합쳐 균형있게 수렴시킨다.
    (ikpy의 orientation_mode="all"은 자세만 우선시하고 위치를 크게 포기하는 문제가 있어 대체)
    반환: (full_q 또는 None, 최종 위치오차mm, 최종 자세오차deg)

    ┌─ [패치 1] 조기중단 제거 ────────────────────────────────────────────────┐
    │ 이전 구현은 "오차가 허용치 안에 들어오면 그 자리에서 즉시 return" 이었다.  │
    │ 즉 반환값은 '수렴한 해'가 아니라 '겨우 통과한 해'였고, 잔차가 허용오차     │
    │ 만큼(최대 pos_tol_mm) 남았다. 곡선 실행에서 pos_tol=2mm, 웨이포인트       │
    │ 간격=2mm 였으므로 스텝 크기와 같은 크기의 노이즈가 명령값에 그대로 실렸다.│
    │ (Figure 3 commanded 속도 톱니 50↔90mm/s, Figure 4 이탈거리 물결의 원인)   │
    │                                                                         │
    │ 지금은 허용치에 들어와도 멈추지 않고 '더 이상 좋아지지 않을 때까지'       │
    │ 계속 줄인 뒤, 그중 최선의 해를 돌려준다. 합격 판정은 마지막에 한 번만.    │
    │ 추가로 스텝이 오차를 오히려 키우면 되돌리고 스텝을 절반으로 줄인다        │
    │ (백트래킹). 고정 0.6 배율은 특이점 근처에서 발산하곤 했다.                │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    q = list(q_seed_full)
    target_p_m = np.array(target_mm) / 1000.0

    # [정확도 바닥] 허용오차의 target_frac까지만 줄이고 멈춘다.
    #   기계 정밀도까지 갈아 넣어봐야 로봇의 각도 분해능(~0.1도, TCP로 ~0.5mm)에
    #   완전히 묻힌다. 이 하한이 없으면 웨이포인트당 68회씩 반복하며 시간만 태운다
    #   (실측: 233개 웨이포인트 검증에 16.2초 -> 하한 도입 후 2.0초, 8.2배).
    #
    #   [9차 세션, §24] 기본값을 0.1 -> 0.5로 올렸다. 프로파일링에서 웨이포인트당
    #   IK 반복이 10.65회였는데, 그렇게까지 갈아넣어 얻은 IK오차가 평균 0.010mm다.
    #   같은 실행의 실제 cross-track은 4.08mm - IK를 로봇 실제오차보다 400배
    #   정밀하게 풀고 있었다는 뜻이다. 0.5여도 pos_goal은 0.065mm(웨이포인트
    #   간격 1.48mm의 4.4%)로, 로봇 TCP 분해능(~0.5mm)의 8분의 1이라 여전히 과잉이다.
    #
    #   [되돌리는 법] 위 패치1이 경고하는 증상(figure_3의 commanded 속도 톱니,
    #   figure_4 이탈거리 물결)이 다시 보이면 이 값을 0.3 -> 0.1 순으로 낮출 것.
    #   그때 pos_goal은 각각 0.039mm / 0.013mm가 된다.
    #   [주의] 패치1의 사고는 pos_tol=2mm에 웨이포인트 간격도 2mm이던 시절
    #   잔차가 스텝 크기의 100%였던 상황이다. 지금은 4.4%라 상황이 다르지만,
    #   실기 확인 없이 안전하다고 단정하지 말 것.
    pos_goal = max(pos_tol_mm * target_frac, 1e-4)
    ori_goal = max(orient_tol_deg * target_frac, 1e-4)

    def _residual(q_):
        fk = chain.forward_kinematics(q_)
        e_pos = target_p_m - fk[:3, 3]                          # 미터
        e_rot = R.from_matrix(target_R @ fk[:3, :3].T).as_rotvec()   # 라디안
        pos_mm = float(np.linalg.norm(e_pos) * 1000.0)
        ori_deg = float(math.degrees(np.linalg.norm(e_rot)))
        # 수렴 목표로 정규화한 스칼라 비용 - 위치/자세를 같은 저울에 올린다.
        cost = (pos_mm / pos_goal) ** 2 + (ori_deg / ori_goal) ** 2
        return e_pos, e_rot, pos_mm, ori_deg, cost

    e_pos, e_rot, pos_mm, ori_deg, cost = _residual(q)
    best_q, best_pos, best_ori, best_cost = list(q), pos_mm, ori_deg, cost

    step = 1.0          # 전체 스텝으로 시작하고, 나빠질 때만 백트래킹으로 줄인다
    stall = 0

    for _ in range(max_iters):
        if best_pos <= pos_goal and best_ori <= ori_goal:
            break                       # 충분히 정확함 -> 더 갈 이유 없음

        # [9차 세션, §24] USE_ANALYTIC_JACOBIAN=True면 FK 1회짜리 해석적 야코비안을
        # 쓴다(웨이포인트당 FK 7회->1회, 실측상 검증시간의 절반 이상 절감 기대).
        # 실기 검증 전에는 False로 두고 기존 수치미분을 그대로 쓴다 - 위쪽
        # analytic_jacobian_6d() 문서 참고.
        J = analytic_jacobian_6d(q) if USE_ANALYTIC_JACOBIAN else numerical_jacobian_6d(q)
        JJt = J @ J.T
        try:
            dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(6),
                                       np.concatenate([e_pos, e_rot]))
        except np.linalg.LinAlgError:
            break

        # --- 백트래킹: 비용이 나빠지면 되돌리고 스텝을 줄여 다시 시도 ---
        accepted = False
        trial_step = step
        for _bt in range(4):
            q_try = list(q)
            for i, idx in enumerate(active_indices):
                q_try[idx] += dq[i] * trial_step
            r = _residual(q_try)
            if r[4] < cost:
                q = q_try
                e_pos, e_rot, pos_mm, ori_deg, cost = r
                accepted = True
                break
            trial_step *= 0.4
        if not accepted:
            break                       # 어떤 스텝으로도 개선 불가 -> 국소최소

        step = min(1.0, trial_step * 2.0)   # 성공했으면 스텝을 회복

        # --- 최선해 갱신 & 정체(stall) 판정 ---
        if cost < best_cost:
            improve = (best_cost - cost) / max(best_cost, 1e-12)
            best_q, best_pos, best_ori, best_cost = list(q), pos_mm, ori_deg, cost
            stall = 0 if improve >= stall_rel else stall + 1
        else:
            stall += 1
        if stall >= stall_patience:
            break                       # 더 줄여봐야 소용없음 -> 조기 종료(정확도 손해 없음)

    if best_pos <= pos_tol_mm and best_ori <= orient_tol_deg:
        return best_q, best_pos, best_ori
    return None, best_pos, best_ori


# ---------------------------------------------------------
# 5. 데드존(.npz) 로드 및 조회
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
    print(f"⚠️ 데드존 파일 로드 실패: {e}")
    _grid_loaded = False
    # 로드 실패해도 이름 자체는 정의해 둬야 이후 함수에서 참조 오류가 나지 않는다
    _reachable_grid = _collision_only_grid = _best_angles_2d = None
    _x_min = _y_min = _z_min = _r_min = 0.0
    _step = 1.0
    _nx = _ny = _nz = _nr = 0
    _safe_mask = None
    _safe_indices = np.empty((0, 3), dtype=int)


def quick_prefilter(x, y, z):
    """데드존 격자로 명백히 불가능한 좌표를 즉시 거름. 반환: (가능여부, 사유)"""
    if not _grid_loaded or _reachable_grid is None:
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


# [9차 세션, §29.16 정리] suggest_nearest_safe()를 삭제했다 - quick_prefilter
# 격자만 보는 구식 추천기로, mycobot_path_editor._suggest_validated_safe(실제
# IK까지 확인)로 대체된 뒤 어디서도 호출되지 않았다. 8차 때는 "참고용"으로
# 남겨뒀으나 이번 정리에서 제거. 격자 최근접 탐색 자체가 필요해지면
# _safe_indices를 쓰는 몇 줄이라 다시 짜는 편이 빠르다.


def grid_seed_pose(target_mm):
    """데드존 파일에 저장된 '검증된 자세'를 J1만 방향 보정해서 IK 시드로 반환"""
    if not _grid_loaded or _best_angles_2d is None:
        return None
    x, y, z = target_mm
    r = math.sqrt(x ** 2 + y ** 2)
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


def solve_position_ik(q_seed_full, target_mm, pos_tol_mm=POS_TOL_MM,
                       max_iters=60, damping=0.05,
                       stall_patience=3, stall_rel=0.05, target_frac=0.5):
    """[9차 세션, §24 후속] natural_pose_at()이 쓰는 위치전용(3자유도, 자세는
    완전히 자유) IK. solve_pose_ik(6자유도 - 위치+자세)와 완전히 같은
    감쇠최소자승+백트래킹 구조를 그대로 재사용하되, 목표가 위치 3개뿐이라
    야코비안도 analytic_jacobian_6d()의 위 3행(J_v)만 쓴다.

    natural_pose_at은 이전에 ikpy의 chain.inverse_kinematics()(scipy
    least_squares, 유한차분 야코비안)를 썼다 - 프로파일링(mycobot_validate_profile.py)
    실측 76ms/호출로, 자체 solve_pose_ik(1.0ms)의 76배였다. 검증 통과하는
    실행에서도 natural_pose_at이 전체 시간의 44%(앵커 7개뿐인데 0.529초)를
    먹었다.

    [주의 - solve_pose_ik와의 차이] 이 함수는 자세를 전혀 구속하지 않는다.
    같은 위치에 도달하는 자세는 무수히 많고(6DOF 로봇에 3개 제약뿐이라
    남는 자유도 3개), 이 함수가 찾는 자세는 그중 "seed에서 감쇠최소자승으로
    최소한만 움직여 도달한" 자세다 - ikpy도 초기값에서 국소 최적화하는
    방식이라 원리는 같지만, 정확히 같은 자세로 수렴한다는 보장은 없다.
    natural_pose_at의 용도(추천탐색·앵커 자세 결정)는 "이 위치에 도달
    가능한 자세가 존재하는가"이지 "특정 자세여야 하는가"가 아니므로 문제
    없다는 게 설계 전제다.

    반환: (full_q 또는 None, 최종 위치오차mm)
    """
    q = list(q_seed_full)
    target_p_m = np.array(target_mm) / 1000.0
    pos_goal = max(pos_tol_mm * target_frac, 1e-4)

    def _residual(q_):
        fk = chain.forward_kinematics(q_)
        e_pos = target_p_m - fk[:3, 3]
        pos_mm = float(np.linalg.norm(e_pos) * 1000.0)
        cost = (pos_mm / pos_goal) ** 2
        return e_pos, pos_mm, cost

    e_pos, pos_mm, cost = _residual(q)
    best_q, best_pos, best_cost = list(q), pos_mm, cost

    step = 1.0
    stall = 0

    for _ in range(max_iters):
        if best_pos <= pos_goal:
            break

        J = analytic_jacobian_6d(q)[:3, :]   # 위치 야코비안(J_v)만 - 3xN
        JJt = J @ J.T
        try:
            dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(3), e_pos)
        except np.linalg.LinAlgError:
            break

        accepted = False
        trial_step = step
        for _bt in range(4):
            q_try = list(q)
            for i, idx in enumerate(active_indices):
                q_try[idx] += dq[i] * trial_step
            r = _residual(q_try)
            if r[2] < cost:
                q = q_try
                e_pos, pos_mm, cost = r
                accepted = True
                break
            trial_step *= 0.4
        if not accepted:
            break

        step = min(1.0, trial_step * 2.0)

        if cost < best_cost:
            improve = (best_cost - cost) / max(best_cost, 1e-12)
            best_q, best_pos, best_cost = list(q), pos_mm, cost
            stall = 0 if improve >= stall_rel else stall + 1
        else:
            stall += 1
        if stall >= stall_patience:
            break

    if best_pos <= pos_tol_mm:
        return best_q, best_pos
    return None, best_pos


# [9차 세션, §24 후속] natural_pose_at을 solve_position_ik로 교체할 때 이
# 스위치를 안 만들었었다 - USE_ANALYTIC_JACOBIAN/ABSORB_SPIKES_TEST처럼
# "행동이 바뀔 수 있는 최적화"엔 되돌릴 방법을 항상 남겨야 한다는 이
# 프로젝트의 관례를 놓친 것. 뒤늦게 추가한다.
#
# [주의 - 이 둘은 진짜 다른 자세로 수렴할 수 있다] solve_position_ik와
# ikpy의 chain.inverse_kinematics는 둘 다 "그 위치에 도달하는 어떤 자세"를
# 찾는 것이지 "특정 자세"를 찾는 게 아니다(위치만 3개 구속, 자세 3자유도는
# 완전히 자유). 위치는 같아도 고르는 자세가 달라질 수 있고, 그 자세가
# build_orientations()의 RotationSpline 보간을 통해 곡선 전체로 전파된다 -
# 즉 앵커 위치는 그대로인데 이 스위치를 바꾸면 곡선 전체의 실행 결과
# (속도 프로파일, cross-track)가 달라질 수 있다. 이게 버그가 아니라
# "어느 쪽이 더 나은 자세를 고르는가"의 문제라는 게 확인되면, 이 스위치로
# 같은 곡선(같은 지문)을 양쪽으로 실행해 직접 비교할 것.
USE_FAST_NATURAL_POSE_AT = True   # False면 예전 ikpy 기반 동작으로 복귀


def natural_pose_at(target_mm, q_seed):
    """목표점에서 '자연스럽게 도달 가능한' 자세를 찾음.
    여러 시드(현재자세 -> 데드존 저장자세)를 순서대로 시도해 성공률을 높인다.
    반환: (full_q, 회전행렬) 또는 (None, None)"""
    seeds = [q_seed]
    gs = grid_seed_pose(target_mm)
    if gs is not None:
        seeds.append(gs)

    for seed in seeds:
        fk = chain.forward_kinematics(seed)
        if np.linalg.norm(fk[:3, 3] * 1000.0 - np.array(target_mm)) <= POS_TOL_MM:
            ok, _ = within_joint_limits(seed)
            if ok and not check_self_collision(seed):
                return list(seed), fk[:3, :3]

        if USE_FAST_NATURAL_POSE_AT:
            # [9차 세션, §24 후속] ikpy의 chain.inverse_kinematics()(scipy 기반,
            # 유한차분 야코비안, 실측 76ms/호출) 대신 자체 solve_position_ik를
            # 쓴다 - 같은 감쇠최소자승 구조라 solve_pose_ik(1ms급)와 비슷한
            # 속도가 나온다. 실패 시 None을 돌려주므로 예외 처리는 필요 없다.
            sol, _pos_err = solve_position_ik(seed, target_mm)
            if sol is None:
                continue
        else:
            # [예전 방식, 대조군으로 보존] USE_FAST_NATURAL_POSE_AT=False일 때만 탄다.
            try:
                sol = chain.inverse_kinematics(
                    target_position=np.array(target_mm) / 1000.0,
                    initial_position=seed,
                )
            except Exception:
                continue
            fk_chk = chain.forward_kinematics(sol)
            if np.linalg.norm(fk_chk[:3, 3] * 1000.0 - np.array(target_mm)) > POS_TOL_MM:
                continue
        fk = chain.forward_kinematics(sol)
        ok, _ = within_joint_limits(sol)
        if not ok or check_self_collision(sol):
            continue
        return sol, fk[:3, :3]

    return None, None


# ---------------------------------------------------------
# 6. 로봇 통신
# ---------------------------------------------------------
def find_robot_port():
    """실제 응답하는 시리얼 포트를 자동 탐색. 반환: (MyCobot320 객체, 포트명)"""
    candidates = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
    for port in candidates:
        try:
            mc_try = MyCobot320(port, BAUD_RATE)
            time.sleep(0.8)
            angles = mc_try.get_angles()
            if isinstance(angles, list) and len(angles) == 6:
                print(f"🔌 로봇 발견: {port}")
                refresh_joint_limits_from_robot(mc_try)   # [패치 B] 연결 직후 실측 한계로 갱신
                return mc_try, port
            try:
                mc_try._serial_port.close()
            except Exception:
                pass
        except Exception:
            continue
    return None, None


# ---------------------------------------------------------
# 6-b. 고속 전송 - pymycobot 우회 (실측: 30ms -> 0.4ms, 약 70배)
# ---------------------------------------------------------
#   pymycobot의 send_angles()는 시리얼 전송(0.4ms)에 비해 30ms를 소비한다
#   (라이브러리 내부 고정 지연). 프레임을 직접 조립해 시리얼에 쓰면 이 지연이 사라진다.
#   프레임 형식은 pymycobot의 실제 출력과 바이트 단위로 대조해 검증 완료:
#       FE FE 0F 22 <각도6개를 (도*100) int16 빅엔디안> <speed> FA
def build_send_angles_frame(angles_deg, speed):
    body = b""
    for a in angles_deg:
        body += struct.pack(">h", int(round(a * 100)))
    body += bytes([int(speed)])
    return bytes([0xFE, 0xFE, 1 + len(body) + 1, 0x22]) + body + bytes([0xFA])


# [10차, §45] 프레임 '쓰기'만 보호하는 락 - AsyncAngleReader가 생기면서
# 처음으로 두 스레드(메인 디스패치 루프의 fast_send_angles, 배경 리더의
# fast_get_angles)가 같은 시리얼 포트에 동시에 접근할 수 있게 됐다.
# **읽기(sp.read 대기)는 이 락으로 안 막는다** - UART는 보통 전이중(TX/RX
# 분리)이라 한쪽이 쓰는 동안 다른 쪽이 읽는 것 자체는 안전하고, 30ms
# 대기 내내 락을 쥐면 메인 스레드의 전송이 매번 그만큼 막혀 파이프라인의
# 존재 이유가 사라진다. **막아야 하는 건 오직 "두 쓰기가 동시에 나가는
# 것"**(바이트가 뒤섞여 프레임이 깨질 수 있는 유일한 지점) - 그래서 락은
# write() 호출 자체를 감싸는 짧은 구간에만 걸린다.
_serial_write_lock = threading.Lock()


def fast_send_angles(mc, angles_deg, speed, check_limits=True):
    """pymycobot을 우회해 프레임을 직접 전송. 반환: 전송 성공 여부.
    pymycobot의 안전검사를 건너뛰므로, 관절 한계 검사를 여기서 직접 수행한다."""
    if check_limits:
        for i, a in enumerate(angles_deg):
            lo, hi = JOINT_LIMITS_DEG[i]
            if not (lo <= a <= hi):
                return False   # 한계를 넘는 명령은 아예 보내지 않음
    try:
        with _serial_write_lock:
            mc._serial_port.write(build_send_angles_frame(angles_deg, speed))
        return True
    except Exception:
        return False


# ---------------------------------------------------------
# 6-c. 고속 읽기 - pymycobot 우회 (10차 세션, §39)
# ---------------------------------------------------------
#   [측정 근거] mycobot_encoder_noise_test.py [3]으로 실측한 결과
#   pymycobot의 get_angles()는 1회당 **평균 29.99ms**(p99 30.35, 최대 30.46,
#   200/200회 전부 20ms 초과)가 걸린다. 분포 폭이 0.5ms 미만으로 극단적으로
#   좁은데, 이건 시리얼 통신 변동이 아니라 **소프트웨어 고정 지연**의 서명이다
#   - 위 6-b에서 send_angles()에 대해 이미 확인한 것과 같은 성질이다.
#
#   [왜 중요한가] 스트리밍 '매번 실측' 모드는 매 사이클 이 호출을 한 번 한다.
#   목표주기 48ms 중 30ms(62.5%)가 이 한 호출에 쓰이고, IK·전송·피드백·
#   피드포워드가 나머지 18ms를 나눠 쓰는 셈이다. §36.1에서 끝내 못 밝힌
#   교란변수(FF ON/OFF에 따라 주기미달이 들쭉날쭉하던 현상)의 유력한 후보다.
#
#   [기대치 - 실측으로 기각됨, 10차 §40] 처음엔 send_angles처럼 라이브러리
#   오버헤드가 섞여 있으리라 봤지만 **틀렸다.** [4]번 검증 실측 결과:
#       pymycobot 평균 30.00ms  /  fast 평균 30.00ms  (절약 0.00ms)
#   두 방식이 정확히 같다. 시리얼 전송량으로 계산하면 요청 5바이트 + 응답
#   17바이트 ≈ 2ms면 충분하므로(115200baud), 나머지 ~28ms는 **ATOM 펌웨어가
#   각도 보고를 약 33Hz로만 갱신하기 때문**이다(문서 §5의 "50Hz 최대=20ms"
#   보다 실제가 나쁘다). 즉 이 30ms는 소프트웨어로 줄일 수 없는 하드웨어
#   하한이다.
#
#   [그래도 이 코드를 남기는 이유] 값 대조는 100/100회 완벽히 통과했다
#   (파싱 실패 0, 불일치 0) - 프로토콜 추정이 정확했다는 뜻이다. 속도
#   이득이 없으니 **기본값은 계속 OFF**지만, 나중에 '비동기 파이프라인
#   읽기'(요청만 보내고 블로킹하지 않은 채 다음 사이클에 응답을 수거)를
#   시도한다면 **pymycobot의 블로킹 API로는 불가능하고 이 직접 접근이
#   반드시 필요하다.** 검증된 상태로 보존해둔다(§40.3).
#
#   [프로토콜 - 실기 대조 검증 완료(10차 §40)]
#       요청: FE FE 02 20 FA                  (0x20 = GET_ANGLES)
#       응답: FE FE 0E 20 <int16 x6, 도*100 빅엔디안> FA
#   pymycobot get_angles()와 정지 자세에서 100회 대조해 전부 일치 확인.
USE_FAST_GET_ANGLES = False   # [10차, §40] 검증은 통과했으나 속도 이득이 0이라 OFF 유지

GET_ANGLES_CMD = 0x20
_GET_ANGLES_REQUEST = bytes([0xFE, 0xFE, 0x02, GET_ANGLES_CMD, 0xFA])


def parse_angles_response(buf):
    """수신 버퍼에서 GET_ANGLES 응답을 찾아 각도 6개(도)를 뽑는다.

    버퍼에 이전 명령의 잔여 바이트가 섞여 있을 수 있으므로, 헤더(FE FE)를
    찾아 앞에서부터 훑으며 **명령코드가 0x20이고 길이가 맞는** 프레임만
    받아들인다. 못 찾으면 None.
    """
    i = 0
    n = len(buf)
    while i + 4 <= n:
        if buf[i] != 0xFE or buf[i + 1] != 0xFE:
            i += 1
            continue
        data_len = buf[i + 2]          # 명령코드1 + 데이터12 + 종료1 = 14(0x0E)
        cmd = buf[i + 3]
        # 프레임 전체 길이 = 헤더2 + 길이1 + (길이 필드가 세는 부분)
        frame_end = i + 3 + data_len
        if cmd != GET_ANGLES_CMD or frame_end > n:
            i += 1
            continue
        payload = buf[i + 4: i + 4 + 12]
        if len(payload) < 12:
            i += 1
            continue
        try:
            vals = struct.unpack(">hhhhhh", payload)
        except struct.error:
            i += 1
            continue
        return [v / 100.0 for v in vals]
    return None


def fast_get_angles(mc, timeout_sec=0.05):
    """pymycobot을 우회해 현재 관절각(도) 6개를 직접 읽는다.
    실패하면 None을 돌려주므로, 호출자는 None이면 mc.get_angles()로
    폴백하면 된다(아래 read_angles 참고).

    [10차, §45] 요청 전송(reset_input_buffer+write)만 _serial_write_lock으로
    감싼다 - fast_send_angles의 쓰기와 겹치지 않게 하려는 목적이다. 그
    뒤 응답을 기다리는 폴링 루프(최대 timeout_sec)는 락 밖에서 돈다 -
    여기서 락을 계속 쥐면 그 시간만큼 메인 스레드의 전송이 막혀
    AsyncAngleReader(§6-d)의 존재 이유가 사라진다(위 락 선언부 주석 참고).
    """
    try:
        sp = mc._serial_port
        with _serial_write_lock:
            sp.reset_input_buffer()        # 이전 명령의 잔여 응답 제거
            sp.write(_GET_ANGLES_REQUEST)
        deadline = time.time() + timeout_sec
        buf = b""
        while time.time() < deadline:
            waiting = sp.in_waiting
            if waiting:
                buf += sp.read(waiting)
                angles = parse_angles_response(buf)
                if angles is not None:
                    return angles
            else:
                time.sleep(0.001)
        return None
    except Exception:
        return None


def read_angles(mc):
    """관절각 읽기의 단일 진입점 - USE_FAST_GET_ANGLES 스위치를 보고 고른다.

    빠른 경로가 실패하면 **조용히 pymycobot으로 폴백**한다. 실시간 루프에서
    간헐적 파싱 실패로 측정이 통째로 비는 것보다, 그 사이클만 느려지는 게
    훨씬 안전하다(스위치 OFF면 예전과 100% 동일 동작).
    """
    if USE_FAST_GET_ANGLES:
        angles = fast_get_angles(mc)
        if angles is not None and len(angles) == 6:
            return angles
    return mc.get_angles()


# ---------------------------------------------------------
# 6-d. 파이프라인 읽기 - 30ms 블로킹을 디스패치 주기 밖으로 (10차, §45)
# ---------------------------------------------------------
#   [배경] §39~40에서 확인했듯 get_angles() 1회는 ~30ms가 걸리고(펌웨어
#   하한, fast_get_angles로도 못 줄인다 - §40), '매번 실측' 모드는 이걸
#   메인 디스패치 루프 안에서 **동기(블로킹)로** 부른다. 즉 이 30ms가
#   그대로 그 사이클의 총 소요시간에 얹힌다.
#
#   [실제 구조 재확인 - §45] mycobot_stream_exec.py의 루프를 다시 보니,
#   P 피드백은 이미 stream_meas_angles[-1](직전 사이클에서 측정해 쌓아둔
#   값)을 쓰고 있었다 - 즉 **피드백은 원래부터 1사이클 정도 낡은 값을
#   쓴다.** 그래서 이 기능이 "새로운 지연을 추가하는" 게 아니라, "이미
#   있던 지연의 정체를 유지하면서 그걸 만드는 30ms 블로킹만 배경으로
#   빼내는" 것에 가깝다 - §40.5에서 "피드백 지연이 48ms 늘어난다"고 적은
#   초기 추정은 부정확했을 수 있다. 정확한 순효과(디스패치 속도 이득 vs
#   피드백 신선도 변화)는 이론만으로 확정하기 어려워 실기 A/B가 필요하다
#   (아래 클래스는 그 실험을 가능하게 하는 메커니즘일 뿐, 효과를
#   보장하지 않는다).
#
#   [메커니즘] 백그라운드 스레드가 fast_get_angles(mc)를 쉬지 않고 반복
#   호출해 "가장 최근에 완료된 측정값"을 공유 슬롯에 계속 갱신한다.
#   메인 디스패치 루프는 그 슬롯을 get_latest()로 즉시(비차단) 읽기만
#   한다 - 새 측정이 아직 없으면 그 전 값을 그대로 돌려준다(못 받는 것보다
#   낡은 값이라도 있는 게 낫다는 §29.20의 원칙과 같은 태도).
#
#   [값 슬롯 안전성] 슬롯은 Lock으로 보호한다 - 리스트(6개 float) 교체는
#   원자적이지 않으므로 락 없이 읽고 쓰면 드물게 절반만 갱신된 리스트를
#   볼 위험이 있다(성능에 영향 없는 짧은 임계구역이라 락 비용은 무시할
#   만하다).
#
#   [시리얼 포트 동시접근 - 별도의, 더 중요한 위험] 배경 스레드(읽기)와
#   메인 스레드(fast_send_angles로 명령 전송)가 **같은 시리얼 포트에
#   동시에 접근**하게 된다 - 이 클래스가 생기기 전까지는 모든 게 한
#   스레드에서 순서대로 일어나 이런 경쟁 자체가 없었다. 그래서 일부러
#   `read_angles(mc)`(USE_FAST_GET_ANGLES에 따라 pymycobot 내부 구현으로
#   빠질 수 있음 - 그 내부 쓰기는 내가 락을 걸 수 없다)가 아니라
#   **`fast_get_angles(mc)`를 직접 부른다** - `fast_send_angles`와 같은
#   `_serial_write_lock`을 공유해서 "두 쓰기가 동시에 나가는 것"(바이트가
#   뒤섞여 프레임이 깨질 수 있는 지점)만은 막는다. 응답을 기다리는 동안은
#   락을 놓아서(§6-c의 fast_get_angles 주석 참고) 메인 스레드의 전송이
#   막히지 않는다 - UART가 보통 전이중이라 "한쪽이 쓰는 동안 다른 쪽이
#   읽는 것"은 안전하다는 전제다. **다만 이건 소프트웨어 수준의 완화일
#   뿐, ATOM 펌웨어가 응답을 만드는 도중 새 명령을 받았을 때 내부적으로
#   올바르게 처리하는지는 이 대화에서 확인할 방법이 없다** - 처음 실기
#   테스트는 저속·주시 상태로 시작할 것을 권한다(§45.4).
#
#   [로봇 없이 검증 가능] get_angles()를 흉내내는 가짜 mc 객체로 스레드
#   시작/정지, 비차단 반환, "아직 값 없음" 상태를 전부 테스트했다
#   (mycobot_encoder_noise_test.py에 실기 대조 메뉴 추가 전이라 이 클래스
#   자체는 아직 실기 검증 전 - mycobot_stream_exec.py의
#   APPLY_ASYNC_MEASUREMENT_PIPELINE 스위치가 기본 False인 이유).
class AsyncAngleReader:
    """백그라운드 스레드로 관절각을 계속 읽어, 호출자가 블로킹 없이
    '가장 최근 측정값'을 가져다 쓸 수 있게 한다."""

    def __init__(self, mc):
        self.mc = mc
        self._lock = threading.Lock()
        self._latest_angles = None
        self._latest_time = None
        self._read_count = 0
        self._stop_flag = threading.Event()
        self._thread = None

    def start(self):
        """이미 돌고 있으면 아무것도 안 한다(중복 시작 방지)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop_flag.is_set():
            # [§45] 일부러 read_angles()가 아니라 fast_get_angles()를 직접
            # 부른다 - fast_send_angles와 같은 락을 타야 두 스레드의 쓰기가
            # 안 겹친다(위 섹션 헤더 주석 참고). 실패(None)하면 그냥 다음
            # 루프에서 다시 시도한다 - 값 슬롯은 마지막 성공값을 계속
            # 들고 있으므로 한 번 실패해도 get_latest()가 끊기지 않는다.
            angles = fast_get_angles(self.mc)
            t = time.time()
            if isinstance(angles, list) and len(angles) == 6:
                with self._lock:
                    self._latest_angles = angles
                    self._latest_time = t
                    self._read_count += 1
            # 대기 없이 바로 다음 읽기 - 어차피 읽기 자체가 ~30ms라
            # 그게 곧 자연스러운 갱신 주기가 된다(추가 sleep 불필요).

    def get_latest(self):
        """블로킹 없이 즉시 반환한다.

        반환: (angles 리스트 또는 None, 측정시각(time.time()) 또는 None).
        아직 한 번도 못 읽었으면 (None, None) - 호출자가 폴백해야 한다
        (스트리밍 루프의 첫 몇 사이클처럼 아직 데이터가 없을 때는 원래도
        피드백을 건너뛰므로 자연스럽게 맞아떨어진다).
        """
        with self._lock:
            return self._latest_angles, self._latest_time

    def read_count(self):
        with self._lock:
            return self._read_count

    def stop(self, timeout_sec=1.0):
        """스레드를 멈추고 합류를 기다린다. start()를 안 불렀으면 안전하게 무시."""
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)
        self._thread = None


# [의도적으로 미사용 - §6.1] 로봇의 현재 자세를 IK 시드로 쓰면 검증이 재현되지
#   않는다(측정 모드만 바꿔도 결과가 달라졌던 버그). 검증 경로에 다시 끌어들이지 말 것.
#   현재 호출부 없음.
def get_current_q_full(mc):
    """실제 로봇의 현재 관절각도를 ikpy용 full_q 배열로 변환"""
    angles_deg = mc.get_angles()
    full_q = [0.0] * len(chain.links)
    for i, idx in enumerate(active_indices):
        full_q[idx] = math.radians(angles_deg[i])
    return full_q


# ---------------------------------------------------------
# 7. 관절 오프셋 보정 (정지상태 백래시/처짐 보정)
# ---------------------------------------------------------
# mycobot_static_error_test.py 로 실측한 정지 상태 오차(err = measured - commanded)를
# 관절별로 모델링해서, 명령을 보내기 '전'에 미리 반대로 보정한다.
#   corrected = desired - safety_scale[j] * err_model(desired)
# (1차 근사: desired 근방에서의 오차가 desired 자체를 명령했을 때의 오차와
#  거의 같다고 가정 - 오차가 작으므로(<1.3도) 이 근사의 오차는 무시할 만하다)
#
# [2026-08-12, 2차 실측 이후 갱신 - 신뢰도가 관절마다 크게 다르다는 게 명확해짐]
#
#   재현성 확인(같은 10자세를 다른 시점에 재실측): 대체로 잘 맞았지만 **J6이
#   세션 도중 +0.26도 -> +0.08도로 계단형으로 드리프트**했다(자세1~3 vs
#   자세4~10 경계가 뚜렷함, 무작위 잡음이 아니라 값 자체가 시간에 따라
#   바뀐다는 뜻). 한 번 잰 상수가 다음 세션엔 이미 틀릴 수 있다는 경고.
#
#   J2 단독 스윕(다른 관절 고정, J2만 0~-90도를 10도씩 훑음): R²=0.815로
#   혼합회귀(R²=0.61)보다 훨씬 깨끗했다. 기울기도 2배 이상 달랐다
#   (0.00817 -> 0.01853) - 혼합회귀가 실제로 많이 오염돼 있었다는 증거.
#   **이 스윕 계수를 채택**.
#
#   방향이력(hysteresis) 테스트(같은 목표를 반대 방향에서 반복 접근): 예상보다
#   훨씬 넓게 퍼져 있었다. **J1, J3, J4, J5 - 6개 중 4개가 부호반전**했다
#   (J2, J6만 방향과 무관). 즉 이 4개 관절은 상수/선형 보정을 그대로 걸면
#   접근 방향에 따라 절반은 개선, 절반은 악화될 수 있다.
#
#   보정검증 테스트(실제로 보정 걸고 재측정): 평균 |오차| 0.83도->0.42도로
#   확실히 개선됐다. **다만 이 검증은 원래 캘리브레이션과 같은 접근 순서로
#   돌았으므로, 우연히 '맞는 방향'으로만 검증됐을 가능성이 있다** - 실제
#   곡선 실행은 이전 웨이포인트가 어디였느냐에 따라 접근 방향이 계속 바뀌므로
#   이 개선폭이 그대로 재현된다는 보장은 없다.
#
# [대응] 방향이력이 확인된 4개 관절(J1,J3,J4,J5)은 보정 강도를 절반으로
#   낮췄다(`_HYSTERESIS_SAFETY_SCALE`). 방향을 모르니 최적으로 보정할 수는
#   없지만, 절반만 걸면 방향을 잘못 짚었을 때의 최악의 피해도 절반으로
#   줄어든다 - "모르니 세게 걸지 않는다"는 보수적 선택. J2/J6은 방향이력이
#   없었으므로 그대로 전체 강도.
#
# [신뢰도 요약]
#   J2 : 단독 스윕, R²=0.815, 방향이력 없음 - 가장 신뢰도 높음. 전체 강도.
#   J6 : 방향이력은 없으나 세션간 값 드리프트 확인됨 - 전체 강도로 두되
#        오래된 세션 데이터에 의존한다는 걸 인지할 것. 주기적 재측정 권장.
#        [6차 갱신] 소스 상수를 -0.235°로 갱신(독립 재측정 3회 수렴, 아래
#        _fit_offset_models의 j6_mean 계산부 주석 참고). 드리프트가 이번엔
#        멈추고 새 값에 안착한 것인지, 계속 움직이는 중인지는 다음 세션
#        재확인(메뉴 6)으로 계속 지켜볼 것.
#   J5 : 3단계 룩업, 각 단계 표준편차 작음, 방향이력 있음 - 강도 절반.
#   J3 : J2와 결합, 방향이력 있음 - 강도 절반.
#   J1     : 원래도 약했고 방향이력까지 확인됨 - 강도 절반.
#   J4     : [6차 갱신] 다중자세 방향이력 재확인(4회 반복)에서 자세마다
#            방향이력 크기가 0.03~0.69°로 크게 요동치는 게 확인됐다. 같은
#            |J2|>=55 기준으로 실측을 갈라보니 J3와 유사하게 base -0.24° vs
#            coupled -0.36°로 뚜렷이 나뉘어(coupled 쪽이 오히려 더 일관적),
#            J3와 같은 2단계(j4_base/j4_coupled) 모델로 교체했다. 강도는
#            여전히 절반(방향이력 확인됨).
#
# [측정 이력]
#   1차: 2026-08-12, 10개 자세 (run1)
#   2차: 2026-08-12, 같은 10개 자세 재실측 - 재현성 확인용 (run2)
#   J2 스윕: 10단계 (별도 리스트, 아래)
#   방향이력: 목표 1개 x 4회 반복 x 2방향 (별도 리스트, 아래)
_STATIC_ERR_POSES_DEG = [
    # (commanded[J1..J6], measured-commanded 오차[J1..J6])  -- run1
    ([0, -30, 30, 0, 30, 0],   [0.38, -0.76, -0.39, -0.38, 0.41, 0.26]),
    ([0, -60, 60, 0, 30, 0],   [0.38, -0.82, -0.45, -0.38, 0.41, 0.26]),
    ([0, -20, 20, 0, 30, 0],   [0.35, -0.56, -0.41, -0.35, 0.41, 0.26]),
    ([45, -45, 45, 0, 30, 0],  [-0.41, -0.79, -0.41, -0.35, 0.41, 0.26]),
    ([-45, -45, 45, 0, 30, 0], [0.44, -0.79, -0.36, -0.35, 0.41, 0.26]),
    ([90, -60, 30, -30, 60, 0], [-0.44, -1.17, -0.74, -0.41, -0.24, 0.26]),
    ([-90, -60, 30, -30, 60, 0], [0.44, -1.17, -0.74, -0.41, -0.24, 0.26]),
    ([0, -90, 90, -30, 30, 0], [-0.43, -0.96, -0.44, -0.38, 0.41, 0.26]),
    ([0, -10, 10, 0, 60, 0],   [-0.35, -0.37, -0.51, -0.38, -0.24, 0.26]),
    ([60, -30, 20, 20, 90, 0], [-0.42, -0.76, -0.49, -0.32, -0.27, 0.26]),
    # -- run2 (재현성 확인, 같은 자세) --
    ([0, -30, 30, 0, 30, 0],    [0.40, -0.76, -0.44, 0.17, 0.41, 0.26]),
    ([0, -60, 60, 0, 30, 0],    [0.38, -0.82, -0.45, -0.08, 0.41, 0.26]),
    ([0, -20, 20, 0, 30, 0],    [0.35, -0.56, -0.41, -0.08, 0.41, 0.26]),
    ([45, -45, 45, 0, 30, 0],   [-0.44, -0.70, -0.36, -0.08, 0.41, 0.08]),
    ([-45, -45, 45, 0, 30, 0],  [0.44, -0.79, -0.36, -0.08, 0.41, 0.08]),
    ([90, -60, 30, -30, 60, 0], [-0.44, -1.17, -0.74, -0.41, -0.24, 0.08]),
    ([-90, -60, 30, -30, 60, 0], [0.44, -1.17, -0.74, -0.41, -0.24, 0.08]),
    ([0, -90, 90, -30, 30, 0],  [-0.40, -0.87, -0.44, -0.38, 0.41, 0.08]),
    ([0, -10, 10, 0, 60, 0],    [-0.38, -0.37, -0.42, -0.35, -0.24, 0.08]),
    ([60, -30, 20, 20, 90, 0],  [-0.42, -0.76, -0.49, -0.32, -0.27, 0.08]),
]

# J2 단독 스윕 (J1=0,J3=30,J4=0,J5=30,J6=0 고정, J2만 변화) - 혼합회귀보다
# 신뢰도가 높아 J2 계수는 이 데이터로만 별도 적합한다.
_J2_SWEEP_DEG = [
    (0, 0.430), (-10, 0.420), (-20, -0.470), (-30, -0.670), (-40, -0.860),
    (-50, -0.880), (-60, -1.080), (-70, -1.190), (-80, -1.120), (-90, -1.230),
]

# 방향이력이 확인된 관절 (0=J1,1=J2,2=J3,3=J4,4=J5,5=J6) - run_hysteresis_test 결과.
# 이 관절들은 보정 강도를 낮춘다(아래 _HYSTERESIS_SAFETY_SCALE).
_HYSTERESIS_CONFIRMED = {0, 2, 3, 4}   # J1, J3, J4, J5
_HYSTERESIS_SAFETY_SCALE = 0.5         # 확인된 관절은 계산된 보정의 절반만 적용

# ┌─ [10차 세션, §50] J3 중력처짐 보정 - 곡선 무관 모델 ──────────────────────┐
# │ **이 자리에 있던 §48/§49의 `USE_DYNAMIC_J3J4_OFFSET`(J3/J4 주행 실측    │
# │ 상수)은 제거됐다.** e433b126 한 곡선에서 잰 상수라 다른 곡선에서 오히려 │
# │ 해로웠다 - 곡선 6종 교차검증에서 "상수만" 모델의 RMSE 감소가            │
# │ **6종 전부 음수**(-28% ~ -100%, 평균 -61.5%)였다. §49에서 A/B 통과로    │
# │ 판정했던 건 그 A/B가 e433 한 곡선에서만 이뤄졌기 때문이다.             │
# │                                                                          │
# │ **대체 모델**: J3 편향은 곡선이 아니라 **자세**로 설명된다.             │
# │     err_J3 = a·sin(θ2+θ3) + b                                           │
# │ θ2+θ3는 팔뚝 링크의 절대 각도이고 그 sin은 중력 모멘트 팔에 비례한다 -  │
# │ 임의로 고른 특징이 아니라 중력 처짐 모델이다. 실제로 곡선별 J3 편향이   │
# │ J3 평균자세와 부호까지 일치했다(자세 +70°->편향 +0.51, -83°->-0.39).    │
# │                                                                          │
# │ **곡선 6종 교차검증**(한 곡선 빼고 학습 -> 뺀 곡선 평가, 28084샘플):     │
# │     상수만            : RMSE -61.5%  (6종 전부 음수)                    │
# │     θ3 선형           : RMSE  +9.3%  (ea535에서 -72%)                   │
# │   **sin(θ2+θ3)       : RMSE +21.7%  최악 곡선 -8%**   <- 채택           │
# │     θ3 + sin(θ2+θ3)  : RMSE +22.4%  최악 곡선 -20%                      │
# │ 2항 모델이 평균은 살짝 높지만 **최악 곡선에서 -20%**라 단독을 택했다 -   │
# │ 이 프로젝트가 §31/§42에서 겪은 실패가 정확히 "평균은 좋은데 특정        │
# │ 곡선에서 무너지는" 유형이라 최악의 경우를 기준으로 골랐다. 파라미터가   │
# │ 하나 적어 과적합도 덜하다. 잔여 편향 0.624 -> 0.108(-83%).              │
# │                                                                          │
# │ 계수는 곡선 균등가중으로 적합했다(e433이 20708/28084 샘플로 압도적이라   │
# │ 단순 최소제곱은 그쪽에 끌린다). 다만 두 방식 차이는 미미했다            │
# │ (a=0.463 vs 0.472) - 편중 우려는 기우였던 셈.                            │
# │                                                                          │
# │ **한계**: J2/J4는 같은 방식이 안 통했다(J2 최고 7.8%인데 e433에서 -75%, │
# │ J4는 편향 자체가 0.134로 작아 잡을 게 없음) - **J3만 적용한다.**        │
# └──────────────────────────────────────────────────────────────────────────┘
_J3_GRAVITY_A = 0.4721    # sin(θ2+θ3) 계수
_J3_GRAVITY_B = 0.1117    # 상수항

# [10차, §51 준비] §50 모델은 아직 실기 A/B 미검증인 채로 스위치까지 지워
# 놓아서 ON/OFF 비교가 불가능했다. 검증 끝날 때까지만 쓰는 임시 플래그 -
# 통과하면 다시 지운다(코드에 영구히 남기지 않음).
#
# 세 값을 갖는 이유 - §51은 서로 다른 두 실험을 한다:
#   "gravity" : §50 중력처짐 모델 (현재 채택안)
#   "static"  : §48 이전의 정지 캘리브레이션 모델. **A/B의 대조군** -
#               "새 모델이 기존 방식보다 나은가"를 §49와 같은 기준으로 본다.
#   "none"    : J3 보정 전혀 없음. **예측 검증의 기준선** - 모델이 예측하는
#               편향(0.4721*sin+0.1117)은 "보정이 없을 때 생기는 처짐"이므로,
#               예측과 실측을 직접 대조하려면 아무 보정도 없는 상태를 재야
#               한다. static과 비교하면 이미 한 번 보정된 잔차라 예측값과
#               비교 대상이 어긋난다.
J3_CORRECTION_MODE = "gravity"      # "gravity" | "static" | "none"


def _fit_offset_models():
    arr_cmd = np.array([p[0] for p in _STATIC_ERR_POSES_DEG], dtype=float)
    arr_err = np.array([p[1] for p in _STATIC_ERR_POSES_DEG], dtype=float)

    # J2: 혼합 데이터가 아니라 단독 스윕으로 적합 (더 깨끗함, R²=0.815 vs 0.61)
    j2v = np.array([x[0] for x in _J2_SWEEP_DEG], dtype=float)
    e2v = np.array([x[1] for x in _J2_SWEEP_DEG], dtype=float)
    a2, b2 = np.polyfit(j2v, e2v, 1)

    # J3: |J2| 기준 2단계 평균 (결합 여부) - 혼합 20점 데이터 사용
    coupled = np.abs(arr_cmd[:, 1]) >= 55
    j3_base = float(arr_err[~coupled, 2].mean())
    j3_coupled = float(arr_err[coupled, 2].mean())

    # [6차 신규, §9 이슈7 후속] J4: J3과 동일한 |J2|>=55 기준으로 2단계 분리.
    # 다중자세 방향이력 재확인(반복 4회) 결과 J4만 자세별로 방향이력 크기가
    # 크게 요동쳤다(0.03°~0.69°) - 그중 |J2|=75인 자세(coupled 조건)가 가장
    # 작고 일관됐다. 실제로 _STATIC_ERR_POSES_DEG를 같은 |J2|>=55 기준으로
    # 갈라보니 base -0.239°(표준편차 0.170) vs coupled -0.358°(표준편차
    # 0.106)로 뚜렷이 나뉘고, coupled 쪽이 오히려 더 일관적이었다(J2가 크게
    # 굽으면 팔뚝 전체가 같이 영향받는 것으로 추정 - J3와 같은 메커니즘일
    # 가능성). 기존 단일 평균(-0.2865°)보다 이 2단계가 실측 분산을 더 잘
    # 설명한다고 판단해 J3와 같은 구조로 교체한다.
    j4_base = float(arr_err[~coupled, 3].mean())
    j4_coupled = float(arr_err[coupled, 3].mean())

    # J5: 명령값별 평균 (관측된 값: 30, 60, 90)
    j5_lookup = {}
    for v in sorted(set(arr_cmd[:, 4])):
        j5_lookup[float(v)] = float(arr_err[arr_cmd[:, 4] == v, 4].mean())

    # J1, J6: 전체 평균 (상수로 근사). J4는 위에서 2단계로 분리했으므로 제외.
    j1_mean = float(arr_err[:, 0].mean())
    # [6차 갱신, §9 이슈8] 원래 j6_mean(+0.197°, 전부 4차 세션의 J6=0 데이터로
    # 만들어짐)이 6차 세션에서 재현되지 않았다. 대신 독립된 3번의 재측정
    # (세션 시작 재확인 1회 + J6 단독 스윕 2회, 스윕은 -90° 접근방향 이상치
    # 제외 후 12개 점 평균)이 전부 -0.23°대로 일관되게 수렴했다:
    #   4차 측정값 +0.197° -> 6차 재확인 -0.230° -> 6차 스윕#1 -0.235°
    #   -> 6차 스윕#2 -0.235°(완전 재현)
    # 세 번의 독립 측정이 같은 값을 가리키므로, 이번엔 런타임 오버라이드가
    # 아니라 소스 상수 자체를 갱신한다. 스윕에서 기울기가 무의미했으므로
    # (R²=0.25, 계수 -0.0012) 여전히 상수모델이 맞다 - 값만 갱신.
    j6_mean = -0.235

    return {
        "j1_mean": j1_mean,
        "j2_slope": float(a2), "j2_intercept": float(b2),
        "j3_base": j3_base, "j3_coupled": j3_coupled,
        "j4_base": j4_base, "j4_coupled": j4_coupled,
        "j5_lookup": j5_lookup,
        "j6_mean": j6_mean,
    }


_OFFSET_MODEL = _fit_offset_models()


def _lookup_j5_err(j5_deg):
    """J5 오차를 룩업한다. 관측된 점(30/60/90) 사이는 선형보간, 바깥은 가장
    가까운 관측값으로 고정(외삽 금지 - 데이터가 3점뿐이라 위험하다)."""
    lut = _OFFSET_MODEL["j5_lookup"]
    xs = sorted(lut.keys())
    if j5_deg <= xs[0]:
        return lut[xs[0]]
    if j5_deg >= xs[-1]:
        return lut[xs[-1]]
    return float(np.interp(j5_deg, xs, [lut[x] for x in xs]))


def update_j6_offset(new_j6_mean_deg, note=""):
    """[5차 신규] J6 세션간 드리프트(§9 이슈8) 대응용 런타임 오버라이드.

    J6는 방향이력은 없지만(그래서 100% 강도로 보정) 세션마다 값 자체가
    이동하는 게 실측으로 확인됐다(+0.26° → +0.08°). 소스코드의 상수를
    세션마다 손으로 고치는 대신, 세션 시작 시 J6만 짧게 재확인해 이 함수로
    `_OFFSET_MODEL["j6_mean"]`을 그 자리에서 갱신할 수 있게 한다.

    이건 **런타임 메모리상의 갱신**이다 - 프로세스가 끝나면 사라진다.
    드리프트가 계속 재현되고 방향성(예: 세션 진행에 따라 항상 줄어든다)이
    보이면, 그때는 이 값을 `_STATIC_ERR_POSES_DEG`/`_fit_offset_models()`
    쪽 소스 상수로 영구 반영하는 걸 고려할 것 (§9 이슈8 참고, 아직 원인
    불명이므로 영구 반영은 신중히).
    """
    old = _OFFSET_MODEL["j6_mean"]
    _OFFSET_MODEL["j6_mean"] = float(new_j6_mean_deg)
    tag = f" ({note})" if note else ""
    print(f"[J6 보정 갱신]{tag} j6_mean: {old:+.3f}° -> {new_j6_mean_deg:+.3f}°  "
          f"(이번 프로세스에서만 유효, 소스코드는 안 바뀜)")


def joint_offset_correction(desired_angles_deg):
    """목표 관절각(도, 6개)을 받아 '이걸 보내면 실제로 desired에 더 가깝게
    도달할 보정된 명령각'을 반환한다.

    [주의 - 사용 전 확인할 것]
    - J2: 단독 스윕(R²=0.815)으로 적합, 방향이력 없음 - 가장 믿을 만하다.
    - J6: 방향이력은 없지만 세션 간 값 드리프트가 확인됐다(§ 위 설명) - 오래
      전에 잰 상수에 계속 의존하고 있다는 걸 인지할 것.
    - J1/J3/J4/J5: **방향이력이 실측으로 확인됐다** - 접근 방향에 따라 오차
      부호가 뒤집힌다. 그래서 이 함수는 이 네 관절에 한해 계산된 보정량의
      50%만 적용한다(`_HYSTERESIS_SAFETY_SCALE`) - 방향을 모르는 채로 세게
      보정하면 절반의 경우 오히려 악화되므로, 절반만 걸어 최악의 피해를
      줄이는 보수적 선택이다. 최적 보정은 아니다.
    - 이 보정은 정지상태 오차만 잡는다. 추종지연(움직이는 동안의 지연,
      τ)은 별개 문제이고 이 보정으로 개선되지 않는다.
    - 검증 시(run_correction_validation_test) 이미 확인된 개선폭(0.83°→0.42°)은
      캘리브레이션과 같은 접근 순서로 검증된 것이라 실제 곡선 실행(접근
      방향이 매번 다름)에서는 그대로 재현되지 않을 수 있다.
    """
    m = _OFFSET_MODEL
    d = list(desired_angles_deg)
    c = list(d)

    err_j1 = m["j1_mean"] * _HYSTERESIS_SAFETY_SCALE
    c[0] = d[0] - err_j1

    err_j2 = m["j2_slope"] * d[1] + m["j2_intercept"]   # 방향이력 없음 - 전체 강도
    c[1] = d[1] - err_j2

    # [10차, §50] J3는 정지 캘리브레이션 대신 중력처짐 모델을 쓴다 - 정지모델은
    # 주행 중 부호가 반대였고(§48.2), 한 곡선에서 잰 상수는 다른 곡선에서
    # 해로웠다(§50). 자세의 함수라 곡선에 무관하다. 강도 100%(주행 데이터
    # 자체가 양방향 평균이라 방향이력이 이미 평균돼 있음).
    if J3_CORRECTION_MODE == "gravity":
        err_j3 = _J3_GRAVITY_A * math.sin(math.radians(d[1] + d[2])) + _J3_GRAVITY_B
    elif J3_CORRECTION_MODE == "static":
        # [10차, §51 준비] A/B 대조군 - §48 이전의 정지모델(J4와 같은 패턴).
        # §49가 실제로 비교했던 대상과 맞춰야 같은 기준으로 비교된다.
        err_j3 = (m["j3_coupled"] if abs(d[1]) >= 55 else m["j3_base"]) * _HYSTERESIS_SAFETY_SCALE
    elif J3_CORRECTION_MODE == "none":
        err_j3 = 0.0                     # 예측 검증용 기준선 - 위 설명 참고
    else:
        raise ValueError(
            f"J3_CORRECTION_MODE는 'gravity'/'static'/'none' 중 하나여야 합니다 "
            f"(현재: {J3_CORRECTION_MODE!r}). 오타로 조용히 보정이 빠지면 "
            f"실험 결과를 통째로 잘못 해석하게 되므로 예외를 낸다.")
    c[2] = d[2] - err_j3

    # J4는 같은 방식이 안 통해(편향 0.134로 작음) 기존 정지모델 유지.
    err_j4 = (m["j4_coupled"] if abs(d[1]) >= 55 else m["j4_base"]) * _HYSTERESIS_SAFETY_SCALE
    c[3] = d[3] - err_j4

    err_j5 = _lookup_j5_err(d[4]) * _HYSTERESIS_SAFETY_SCALE
    c[4] = d[4] - err_j5

    err_j6 = m["j6_mean"]   # 방향이력 없음 - 전체 강도 (단 드리프트 주의, 위 설명)
    c[5] = d[5] - err_j6
    return c
