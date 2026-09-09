"""
pymycobot의 send_coords/get_coords 자세(rx,ry,rz) 표현 방식 검증용 진단 스크립트
================================================================================

목적:
    로봇을 특정 관절각도로 이동시킨 뒤, pymycobot이 보고하는 rx,ry,rz 값과
    저희가 ikpy로 계산한 회전행렬을 여러 오일러각 변환 방식으로 비교해서,
    어떤 변환이 실제로 일치하는지 찾아낸다.

사용법:
    1. 로봇 주변에 충돌할 물건이 없는지 확인 (테스트 자세로 실제로 움직입니다)
    2. python3 mycobot_orientation_diagnostic.py 실행
    3. 출력된 결과 전체를 그대로 복사해서 보여주세요.
"""

import time
import math
import numpy as np
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
from pymycobot import MyCobot320

URDF_PATH = "/home/kms/colcon_ws/src/mycobot_ros2/mycobot_description/urdf/mycobot_320_m5_2022/mycobot_320_m5_2022.urdf"
PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# 특이점을 피하면서, 6축 모두가 뚜렷하게 서로 다른 값을 갖는 안전한 테스트 자세
TEST_ANGLES_DEG = [15, -20, 25, 10, -15, 30]
SPEED = 20

# ---------------------------------------------------------
# 1. ikpy 체인 로드
# ---------------------------------------------------------
chain = Chain.from_urdf_file(URDF_PATH, base_elements=["base"])
active_indices = [i for i, l in enumerate(chain.links) if l.joint_type != "fixed"]

# ---------------------------------------------------------
# 2. 로봇에 테스트 자세 명령
# ---------------------------------------------------------
mc = MyCobot320(PORT, BAUD_RATE)
time.sleep(1)

print(f"➡️ 테스트 자세로 이동: {TEST_ANGLES_DEG}")
mc.send_angles(TEST_ANGLES_DEG, SPEED)
time.sleep(0.3)
while mc.is_moving():
    time.sleep(0.1)
time.sleep(1)  # 완전히 안정화될 시간을 넉넉히 줌

# ---------------------------------------------------------
# 3. 실제 도달한 각도 및 좌표 읽기
# ---------------------------------------------------------
actual_angles_deg = mc.get_angles()
actual_coords = mc.get_coords()  # [x, y, z, rx, ry, rz]

print(f"\n실제 도달한 관절각도 (get_angles): {actual_angles_deg}")
print(f"pymycobot이 보고한 좌표 (get_coords): {actual_coords}")

# ---------------------------------------------------------
# 4. ikpy로 같은 관절각도의 순기구학 계산
# ---------------------------------------------------------
full_q = [0.0] * len(chain.links)
for i, idx in enumerate(active_indices):
    full_q[idx] = math.radians(actual_angles_deg[i])

fk = chain.forward_kinematics(full_q)
position_mm = fk[:3, 3] * 1000.0
rotation_matrix = fk[:3, :3]

print(f"\nikpy로 계산한 위치 (mm): {position_mm.tolist()}")
print(f"pymycobot 위치 (mm)     : {actual_coords[:3]}")
print(f"위치 차이 (mm)           : {(position_mm - np.array(actual_coords[:3])).tolist()}")

# ---------------------------------------------------------
# 5. 회전행렬을 여러 오일러각 변환 방식으로 비교
# ---------------------------------------------------------
print(f"\npymycobot이 보고한 자세 (rx,ry,rz): {actual_coords[3:]}")
print("\n다양한 오일러각 변환 방식과 비교:")

conventions = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
               "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]

r = R.from_matrix(rotation_matrix)
for conv in conventions:
    euler_deg = r.as_euler(conv, degrees=True)
    print(f"  {conv:>4}: {euler_deg.round(2).tolist()}")

print("\n위 목록 중 pymycobot이 보고한 rx,ry,rz와 가장 비슷한(또는 부호만 반대인) "
      "조합을 찾아서 알려주세요.")

