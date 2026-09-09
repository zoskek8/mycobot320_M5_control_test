"""
myCobot 320 도달가능영역 + 자가충돌 사전계산 - 버그 수정 버전
================================================================

이전 버전 대비 수정된 사항:
    1. [치명적] URDF_USE_SELF_COLLISION_EXCLUDE_PARENT 플래그 누락 수정
       -> 이게 없으면 인접 링크끼리 항상 충돌로 오판됨 (거의 모든 자세가 "충돌"로 나옴)
    2. [되돌림] 방향을 고정하는 방식(orientation_mode="all")은 탐색 자유도를 줄여서
       실제로는 도달 가능한데 놓치는(false negative) 경우가 늘어나는 문제가 있어,
       원래 방식대로 방향은 항상 자유롭게 두고, 여러 번(N_IK_RESTARTS) 다른
       무작위 시작 자세로 재시도하는 방식으로 되돌림.
    3. [로직 버그] reachable_2d 와 collision_only_2d 가 동시에 True가 되던 문제 수정
       -> 루프를 다 돈 뒤, "충돌 없는 해가 하나라도 있었는가"로 최종 판정하도록 변경
    4. [추가] 충돌 없는 해를 찾으면 그 관절 각도(degree)를 best_angles_2d에 저장.
       실제 로봇 제어 시 이 각도를 그대로 사용하면, 사전계산 때 검증한 자세와
       실제로 로봇에 보내는 자세가 정확히 일치함.

사전 설치:
    pip3 install ikpy pybullet numpy matplotlib
"""

import os
import re
import time
import numpy as np
from ikpy.chain import Chain
import pybullet as p
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. 설정
# ---------------------------------------------------------
URDF_PATH = "/home/kms/colcon_ws/src/mycobot_ros2/mycobot_description/urdf/mycobot_320_m5_2022/mycobot_320_m5_2022.urdf"
PACKAGE_ROOT = "/home/kms/colcon_ws/src/mycobot_ros2/mycobot_description"

MAX_REACH_MM = 350       # 공식 스펙 기준 (의도적 제한: 350mm 이상은 특이점/정확도 저하 구간이라 실사용에서 제외)
Z_MAX_MM = 600
GRID_STEP_MM = 20
Z_MIN_MM = -150
IK_TOLERANCE_MM = 10

# ---------------------------------------------------------
# 1. ikpy 체인 로드
# ---------------------------------------------------------
chain = Chain.from_urdf_file(URDF_PATH, base_elements=["base"])
active_indices = [i for i, l in enumerate(chain.links) if l.joint_type != "fixed"]
joint_names_ikpy = [chain.links[i].name for i in active_indices]
print(f"ikpy 활성 관절: {joint_names_ikpy}\n")

# ---------------------------------------------------------
# 2. PyBullet용 package:// 경로 치환
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

# ---------------------------------------------------------
# 3. PyBullet 초기화
#    [수정 1] EXCLUDE_PARENT 플래그 추가 - 인접 링크 오탐 방지
# ---------------------------------------------------------
physics_client = p.connect(p.DIRECT)
robot_id = p.loadURDF(
    TEMP_URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
)

name_to_pb_index = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    if info[2] != p.JOINT_FIXED:
        name_to_pb_index[info[1].decode("utf-8")] = j

try:
    pb_indices_in_order = [name_to_pb_index[name] for name in joint_names_ikpy]
except KeyError as e:
    raise RuntimeError(
        f"관절 이름 매칭 실패: {e}\n"
        f"ikpy 이름: {joint_names_ikpy}\n"
        f"pybullet 이름: {list(name_to_pb_index.keys())}"
    )
print(f"관절 이름 매칭 성공: {list(zip(joint_names_ikpy, pb_indices_in_order))}\n")


def set_pose_and_check_collision(active_angles_rad):
    for pb_idx, angle in zip(pb_indices_in_order, active_angles_rad):
        p.resetJointState(robot_id, pb_idx, angle)
    p.performCollisionDetection()
    return len(p.getContactPoints(bodyA=robot_id, bodyB=robot_id)) > 0


def ikpy_solution_to_active_angles(sol):
    return [sol[i] for i in active_indices]


# ---------------------------------------------------------
# 4. [핵심] 2D (r, z) 단면 계산 - 축대칭 최적화
#    방향은 항상 자유로 두고, 여러 번 무작위 재시도로 충돌 없는 해를 탐색
# ---------------------------------------------------------
print("[1단계] 축대칭 2D 단면 계산 시작 (방향 자유 + 랜덤 재시도 방식)...\n")

N_IK_RESTARTS = 5  # 방향을 자유롭게 둔 채로 몇 번 다른 시작점에서 재시도할지
rng = np.random.default_rng(0)

# 관절별 한계(라디안) - ikpy 링크에 있으면 그걸 쓰고, 없으면 기본값 사용
_active_bounds = []
for idx in active_indices:
    bounds = getattr(chain.links[idx], "bounds", None)
    if bounds is not None and bounds[0] is not None and bounds[1] is not None:
        _active_bounds.append(bounds)
    else:
        _active_bounds.append((-np.radians(165), np.radians(165)))


def random_initial_guess():
    """방향 자유 재시도용 무작위 시작 자세.
    J1(첫 번째 활성 관절)은 0으로 고정하고, 나머지(J2~J6)만 무작위로 섞는다.
    -> 이렇게 해야 랜덤 재시도로 찾은 해도 J1이 0 근처에 머물 가능성이 높아져서,
       나중에 theta만큼 보정할 때 여유(headroom)가 최대로 확보된다."""
    full = [0.0] * len(chain.links)
    for i, (idx, (lo, hi)) in enumerate(zip(active_indices, _active_bounds)):
        if i == 0:
            continue  # J1은 항상 0으로 고정
        full[idx] = rng.uniform(lo, hi)
    return full


def zero_j1_initial_guess():
    """J1은 0으로, 나머지는 0으로 시작하는 '기준' 시도용 초기값.
    y=0 목표에서는 J1을 움직여도 위치오차가 줄지 않으므로,
    솔버가 이 초기값 근처(J1≈0)에 머물 가능성이 높아진다."""
    full = [0.0] * len(chain.links)
    return full


r_range = np.arange(0, MAX_REACH_MM + 1, GRID_STEP_MM)
z_range = np.arange(Z_MIN_MM, Z_MAX_MM + 1, GRID_STEP_MM)
nr, nz = len(r_range), len(z_range)
print(f"격자 크기: {nr}(r) x {nz}(z) = {nr*nz}개 후보점\n")

reachable_2d = np.zeros((nr, nz), dtype=bool)
collision_only_2d = np.zeros((nr, nz), dtype=bool)
# 충돌 없는 해가 나왔을 때의 실제 관절 각도(degree)를 저장 (r=0, y=0 기준면에서의 값)
best_angles_2d = np.full((nr, nz, len(active_indices)), np.nan)

last_free_solution = None  # position-only 시도의 연속성을 위해 재사용
start_time = time.time()

for ir, r in enumerate(r_range):
    for iz, z in enumerate(z_range):
        # r(수평 거리)만으로 1차 필터링. z는 베이스 높이 오프셋 때문에
        # 원점 기준 구형으로 자르면 실제로 도달 가능한 상단부가 잘려나가므로,
        # z 자체의 범위(Z_MIN_MM~Z_MAX_MM)에만 맡기고 여기서는 r만 본다.
        if r > MAX_REACH_MM + GRID_STEP_MM:
            continue

        target_m = np.array([r, 0.0, z]) / 1000.0

        found_collision_free = False
        found_any_valid = False

        # --- (A) J1=0 기준 시도 (최우선) ---
        # 방향 보정(theta) 시 여유가 최대가 되도록, J1이 0 근처인 해를 우선적으로 찾음
        try:
            sol = chain.inverse_kinematics(target_position=target_m, initial_position=zero_j1_initial_guess())
            fk = chain.forward_kinematics(sol)
            error_mm = np.linalg.norm(fk[:3, 3] * 1000.0 - np.array([r, 0.0, z]))
            if error_mm <= IK_TOLERANCE_MM:
                found_any_valid = True
                active_angles = ikpy_solution_to_active_angles(sol)
                if not set_pose_and_check_collision(active_angles):
                    found_collision_free = True
                    best_angles_2d[ir, iz, :] = np.degrees(active_angles)
                    last_free_solution = sol
        except Exception:
            pass

        # --- (B) 이전 해 이어서 시도 (A가 실패했을 때만) ---
        if not found_collision_free:
            try:
                if last_free_solution is not None:
                    sol = chain.inverse_kinematics(target_position=target_m, initial_position=last_free_solution)
                else:
                    sol = chain.inverse_kinematics(target_position=target_m)
                fk = chain.forward_kinematics(sol)
                error_mm = np.linalg.norm(fk[:3, 3] * 1000.0 - np.array([r, 0.0, z]))
                if error_mm <= IK_TOLERANCE_MM:
                    found_any_valid = True
                    active_angles = ikpy_solution_to_active_angles(sol)
                    if not set_pose_and_check_collision(active_angles):
                        found_collision_free = True
                        best_angles_2d[ir, iz, :] = np.degrees(active_angles)
                        last_free_solution = sol
            except Exception:
                pass

        # --- (C) 방향 자유, 여러 번 랜덤 시작점으로 재시도 (A, B 다 실패했을 때만) ---
        # [수정] 방향을 고정하지 않고 자유롭게 둔 채 재시도 (탐색 자유도 유지)
        if not found_collision_free:
            for _ in range(N_IK_RESTARTS):
                try:
                    sol = chain.inverse_kinematics(
                        target_position=target_m,
                        initial_position=random_initial_guess(),
                    )
                    fk = chain.forward_kinematics(sol)
                    error_mm = np.linalg.norm(fk[:3, 3] * 1000.0 - np.array([r, 0.0, z]))
                    if error_mm > IK_TOLERANCE_MM:
                        continue

                    found_any_valid = True
                    active_angles = ikpy_solution_to_active_angles(sol)
                    if not set_pose_and_check_collision(active_angles):
                        found_collision_free = True
                        best_angles_2d[ir, iz, :] = np.degrees(active_angles)
                        last_free_solution = sol
                        break  # 충돌 없는 해를 찾았으니 더 볼 필요 없음
                except Exception:
                    continue

        # --- 최종 판정 (수정된 로직: 두 배열이 동시에 True가 될 수 없음) ---
        if found_collision_free:
            reachable_2d[ir, iz] = True
        elif found_any_valid:
            collision_only_2d[ir, iz] = True
        # else: 아예 도달 불가능한 점 -> 둘 다 False로 남음 (정상)

    elapsed = time.time() - start_time
    print(f"r={r}mm 처리 완료 ({ir+1}/{nr}), 경과 {elapsed:.1f}초")

print(f"\n도달 가능(충돌 없음): {reachable_2d.sum()}개")
print(f"충돌 불가피: {collision_only_2d.sum()}개")

# ---------------------------------------------------------
# 5. 3D 격자로 확장 (축대칭 브로드캐스팅)
# ---------------------------------------------------------
print("\n[2단계] 3D 격자 매핑 중...")
x_range = np.arange(-MAX_REACH_MM, MAX_REACH_MM + 1, GRID_STEP_MM)
y_range = np.arange(-MAX_REACH_MM, MAX_REACH_MM + 1, GRID_STEP_MM)
nx_len, ny_len, nz_len = len(x_range), len(y_range), len(z_range)

reachable_grid = np.zeros((nx_len, ny_len, nz_len), dtype=bool)
collision_only_grid = np.zeros((nx_len, ny_len, nz_len), dtype=bool)

xs_2d, ys_2d = np.meshgrid(x_range, y_range, indexing="ij")
rs_2d = np.sqrt(xs_2d**2 + ys_2d**2)
ir_indices = np.round(rs_2d / GRID_STEP_MM).astype(int)
valid_mask = ir_indices < nr

for iz in range(nz_len):
    reachable_grid[:, :, iz][valid_mask] = reachable_2d[ir_indices[valid_mask], iz]
    collision_only_grid[:, :, iz][valid_mask] = collision_only_2d[ir_indices[valid_mask], iz]

# ---------------------------------------------------------
# 6. 저장
# ---------------------------------------------------------
np.savez_compressed(
    "mycobot_reachability_grid.npz",
    reachable_grid=reachable_grid,
    collision_only_grid=collision_only_grid,
    best_angles_2d=best_angles_2d,   # 관절각도 조회용 (r, z 기준, degree)
    r_min=0.0,
    x_min=x_range[0], y_min=y_range[0], z_min=z_range[0],
    step=GRID_STEP_MM,
    nx=nx_len, ny=ny_len, nz=nz_len,
    nr=nr,
)
print(f"\n저장 완료! (총 {time.time()-start_time:.1f}초)")

# ---------------------------------------------------------
# 7. 시각화
# ---------------------------------------------------------
DESK_MARGIN = 20
xs, ys, zs = np.meshgrid(x_range, y_range, z_range, indexing="ij")

green = reachable_grid & (~collision_only_grid) & (zs >= DESK_MARGIN)
red = reachable_grid & (~collision_only_grid) & (zs < DESK_MARGIN)
blue = collision_only_grid  # 이제 reachable_grid와 절대 겹치지 않음

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
if np.any(green):
    ax.scatter(xs[green], ys[green], zs[green], c="green", s=2, alpha=0.15, label="safe zone")
if np.any(red):
    ax.scatter(xs[red], ys[red], zs[red], c="red", s=3, alpha=0.4, label="dead zone (z < 20mm)")
if np.any(blue):
    ax.scatter(xs[blue], ys[blue], zs[blue], c="blue", s=3, alpha=0.5, label="self collision zone")

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("myCobot 320 zone simulation")
ax.legend()
plt.tight_layout()
plt.savefig("mycobot_workspace_full.png", dpi=150)
print("이미지 저장됨: mycobot_workspace_full.png")
plt.show()