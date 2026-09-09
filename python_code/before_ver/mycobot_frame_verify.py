"""
send_angles 프레임 캡처 & 직접 인코더 검증
==========================================

pymycobot의 send_angles()는 시리얼 전송에 0.4ms면 되는 일을 30ms나 붙잡고 있습니다.
(내부에 고정 지연이 박혀 있음 - 진단 결과 확인됨)

이 스크립트는:
  1. pymycobot이 send_angles로 실제 보내는 바이트 프레임을 캡처
  2. 같은 값을 제가 직접 만든 인코더로 조립
  3. 둘이 완전히 일치하는지 대조

일치가 확인되면, 앞으로는 pymycobot을 거치지 않고 직접 프레임을 써서
명령 주기를 30ms -> 1ms 수준으로 낮출 수 있습니다.

주의: 로봇은 움직이지 않습니다 (전송만 가로채서 분석). 안전합니다.
"""

import time
import glob
import struct
from pymycobot import MyCobot320


# ---------------------------------------------------------
# 직접 구현한 프레임 인코더 (검증 대상)
# ---------------------------------------------------------
def build_send_angles_frame(angles_deg, speed):
    """
    mycobot 시리얼 프로토콜 추정 형식:
        FE FE <len> <cmd=0x22> <angle1_hi angle1_lo> ... <angle6_hi angle6_lo> <speed> FA
      - 각도는 (도 * 100)을 부호있는 16비트 빅엔디안으로
      - len = 명령바이트(1) + 데이터(13) + 종료바이트(1) = 15 = 0x0F
    """
    body = b""
    for a in angles_deg:
        body += struct.pack(">h", int(round(a * 100)))
    body += bytes([int(speed)])
    length = 1 + len(body) + 1
    return bytes([0xFE, 0xFE, length, 0x22]) + body + bytes([0xFA])


# ---------------------------------------------------------
# 로봇 연결
# ---------------------------------------------------------
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
    print("❌ 로봇을 찾지 못했습니다.")
    raise SystemExit

print(f"🔌 연결됨: {port}\n")

# ---------------------------------------------------------
# 실제 전송 바이트 가로채기 (write를 감싸서 기록만 하고 통과)
# ---------------------------------------------------------
sp = mc._serial_port
captured = []
original_write = sp.write


def spy_write(data):
    captured.append(bytes(data))
    return original_write(data)


sp.write = spy_write

# ---------------------------------------------------------
# 여러 케이스로 대조
# ---------------------------------------------------------
test_cases = [
    ([0, 0, 0, 0, 0, 0], 30),
    ([10.5, -20.25, 30, -40, 50.75, -60], 50),
    ([-165, 165, -90.5, 45.25, 0, 175], 100),
]

all_match = True
for angles, speed in test_cases:
    captured.clear()
    mc.send_angles(angles, speed)
    time.sleep(0.1)

    actual = captured[0] if captured else b""
    mine = build_send_angles_frame(angles, speed)

    match = (actual == mine)
    all_match = all_match and match

    print(f"입력: {angles}, speed={speed}")
    print(f"  pymycobot: {actual.hex(' ')}")
    print(f"  내 인코더 : {mine.hex(' ')}")
    print(f"  일치 여부 : {'✅ 일치' if match else '❌ 불일치'}\n")

sp.write = original_write

print("=" * 60)
if all_match:
    print("✅ 모든 케이스 일치! 직접 프레임 전송으로 우회할 수 있습니다.")
    print("   -> 명령 주기를 30ms에서 1ms 수준으로 낮출 수 있습니다.")
else:
    print("❌ 불일치가 있습니다. 위의 pymycobot 실제 바이트를 보고 인코더를 고쳐야 합니다.")
    print("   출력된 hex를 그대로 알려주시면 형식을 맞춰드리겠습니다.")
print("=" * 60)
