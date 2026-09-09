"""
send_angles 병목 진단 스크립트
================================

목적:
    1. send_angles() 한 번이 실제로 몇 ms 걸리는지 순수하게 측정
    2. 그 시간이 '시리얼 전송'인지 'pymycobot 내부 sleep'인지 구분
    3. pymycobot 소스에서 인위적 지연(time.sleep)을 찾아냄

30ms가 하드웨어 한계인지, 아니면 라이브러리가 넣은 인위적 지연인지 가려내는 것이 목적입니다.
후자라면 우회해서 훨씬 빠른 명령 주기를 얻을 수 있습니다.

주의: 로봇이 실제로 아주 조금씩 움직입니다. 주변 공간을 확보하세요.
"""

import time
import inspect
import numpy as np
from pymycobot import MyCobot320
import pymycobot

print("=" * 60)
print("pymycobot 버전:", getattr(pymycobot, "__version__", "unknown"))
print("=" * 60)

# ---------------------------------------------------------
# 1. pymycobot 소스에서 인위적 지연(sleep) 찾기
# ---------------------------------------------------------
print("\n[1] pymycobot 내부의 time.sleep 탐색")
try:
    from pymycobot import common
    src = inspect.getsource(common)
    for i, line in enumerate(src.splitlines(), 1):
        if "sleep" in line and not line.strip().startswith("#"):
            print(f"   common.py:{i}: {line.strip()}")
except Exception as e:
    print("   common.py 확인 실패:", e)

try:
    import pymycobot.mycobot320 as m320
    src2 = inspect.getsource(m320)
    found = False
    for i, line in enumerate(src2.splitlines(), 1):
        if "sleep" in line and not line.strip().startswith("#"):
            print(f"   mycobot320.py:{i}: {line.strip()}")
            found = True
    if not found:
        print("   mycobot320.py: sleep 없음")
except Exception as e:
    print("   mycobot320.py 확인 실패:", e)

# ---------------------------------------------------------
# 2. 로봇 연결
# ---------------------------------------------------------
import glob
port = None
for p in sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*")):
    try:
        mc_try = MyCobot320(p, 115200)
        time.sleep(0.8)
        if isinstance(mc_try.get_angles(), list):
            mc, port = mc_try, p
            break
    except Exception:
        continue

if port is None:
    print("\n❌ 로봇을 찾지 못했습니다. USB/전원/Transponder(USB) 모드를 확인하세요.")
    raise SystemExit

print(f"\n🔌 연결됨: {port}")
mc.power_on()
time.sleep(2)
mc.clear_error_information()
mc.set_fresh_mode(1)
time.sleep(0.3)

base = mc.get_angles()
print("현재 각도:", base)

# ---------------------------------------------------------
# 3. send_angles 순수 소요시간 측정
# ---------------------------------------------------------
print("\n[2] send_angles() 순수 소요시간 (100회)")
N = 100
durations = []
for k in range(N):
    target = list(base)
    target[0] = base[0] + (0.3 if k % 2 == 0 else -0.3)   # 아주 미세하게만 움직임
    t0 = time.perf_counter()
    mc.send_angles(target, 30)
    durations.append((time.perf_counter() - t0) * 1000)

d = np.array(durations)
print(f"   평균 {d.mean():.2f}ms / 중앙값 {np.median(d):.2f}ms / "
      f"최소 {d.min():.2f}ms / 최대 {d.max():.2f}ms")

# ---------------------------------------------------------
# 4. 순수 시리얼 쓰기 속도 (비교 기준)
# ---------------------------------------------------------
print("\n[3] 순수 시리얼 write 소요시간 (같은 크기의 더미 프레임 100회)")
try:
    sp = mc._serial_port
    dummy = bytes([0xFE, 0xFE, 0x02, 0x20, 0xFA])   # 관측된 프레임 형식 예시
    wd = []
    for _ in range(N):
        t0 = time.perf_counter()
        sp.write(dummy)
        sp.flush()
        wd.append((time.perf_counter() - t0) * 1000)
    wd = np.array(wd)
    print(f"   평균 {wd.mean():.3f}ms / 최대 {wd.max():.3f}ms")
except Exception as e:
    print("   측정 실패:", e)

# ---------------------------------------------------------
# 5. 결론
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("해석 방법:")
print("  - [2]가 [3]보다 훨씬 크다면  -> pymycobot 내부 지연이 병목 (우회 가능!)")
print("  - [2]와 [3]이 비슷하다면      -> 시리얼/펌웨어의 진짜 한계 (우회 불가)")
print("=" * 60)

mc.send_angles(base, 30)
time.sleep(1)
