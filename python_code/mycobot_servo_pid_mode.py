# -*- coding: utf-8 -*-
"""
mycobot_servo_pid_mode.py
==========================
[8차 세션, 진단/조치용 1회성 스크립트] §23(정지자세 진동 조사)에서 찾은
단서를 실제로 테스트하기 위한 도구.

배경(§23.6 참고): Elephant Robotics 공식 문서(myCobot 280 기준, "5.4 PID
control")에 pymycobot의 저수준 API `set_servo_data(servo_no, data_id, value)`
/ `get_servo_data(servo_no, data_id)`로 서보 설정을 건드리는 예제가 있다.
`data_id=23`은 문서에 명시적으로 이렇게 설명돼 있다:

    value=0 → "부드러운 움직임(smooth motion)" 모드 - 안정성 우선,
              정밀도는 약간 희생될 수 있음
    value=4 → "고정밀(high-precision)" 모드 - 원문: "will cause the arm
              to continuously adjust the steady state error"
              (정상상태 오차를 계속 미세조정하려 든다)

value=4의 설명이 우리가 §23에서 실측한 "정지 중에도 계속 두 값 사이를
왔다갔다 하는" 증상과 정확히 일치한다 - 로봇이 지금 고정밀 모드라서
목표각과의 미세한 차이를 서보가 계속 잡으려 드는 것일 수 있다.

[중요, 반드시 읽을 것]
  - 이 문서는 myCobot 280 Pi 기준이다. myCobot 320 M5에서 data_id=23이
    정확히 같은 의미인지는 100% 확인되지 않았다 - 이 스크립트로 먼저
    "읽기"만 해서 현재 값이 0이나 4처럼 이 두 프리셋 중 하나에 해당하는
    합리적인 값인지부터 확인할 것.
  - 이 설정이 재부팅/재연결 후에도 유지되는지(서보 내부 EEPROM에 저장)
    아니면 매 세션 다시 설정해야 하는지도 미확인 - 이 스크립트로 값을
    바꾼 뒤 로봇 전원을 껐다 켜고 다시 읽어보면 확인 가능하다.
  - "부드러운 모드"가 진동은 잡아도 실행 정밀도(cross-track 등)를
    악화시킬 수 있다고 문서에 명시돼 있다 - 바꾼 뒤에는 반드시
    mycobot_static_error_test.py나 실제 곡선 실행으로 정밀도 저하가
    없는지도 같이 확인할 것.

사용법:
    python3 mycobot_servo_pid_mode.py

    [1] 전체 관절의 현재 data_id=23 값 조회 (쓰기 없음, 항상 먼저 해볼 것)
    [2] 특정 관절(들)을 "부드러운 모드"(0)로 변경
    [3] 특정 관절(들)을 "고정밀 모드"(4)로 변경 (되돌리기용)
    [q] 종료

    [2], [3] 모두 쓰기 전에 대상 관절과 현재값을 보여주고 확인을 받는다.
    쓴 뒤에는 즉시 다시 읽어서(get_servo_data) 실제로 반영됐는지 확인한다.
"""

import sys
import time

from mycobot_kinematics import find_robot_port

# ---------------------------------------------------------------------------
# [Elephant Robotics 공식문서 기준, 280 Pi] data_id=23 = PID 프리셋 스위치
# ---------------------------------------------------------------------------
PID_MODE_DATA_ID = 23
PID_MODE_SMOOTH = 0     # "부드러운 움직임" - 안정성 우선, 정밀도 약간 희생 가능
PID_MODE_PRECISE = 4    # "고정밀" - 정상상태 오차를 계속 미세조정 (진동 의심 원인)

JOINT_COUNT = 6

# [8차 세션 추가] 6관절을 한 번에 처리(특히 all)할 때 -1 읽기오류와 쓰기
# 씹힘이 관찰됐다 - 버스 혼잡으로 추정. 관절당 대기시간을 늘리고, 읽기/쓰기
# 모두 재시도 로직을 추가했다.
INTER_SERVO_DELAY_SEC = 0.25   # 관절 하나 처리 후 다음 관절로 넘어가기 전 대기 (기존 0.1 -> 0.25)
RETRY_COUNT = 4                 # -1(오류)이거나 기대값과 다르면 이만큼 재시도
RETRY_DELAY_SEC = 0.2


def _get_with_retry(mc, servo_no, data_id, retries=RETRY_COUNT, delay=RETRY_DELAY_SEC):
    """get_servo_data를 재시도한다. -1(오류)이나 예외가 나오면 delay 후 재시도.
    마지막 시도 결과를 그대로 반환한다(성공 못해도 -1이나 예외메시지를 반환)."""
    last = None
    for attempt in range(retries):
        try:
            v = mc.get_servo_data(servo_no, data_id)
        except Exception as e:
            v = f"읽기실패({e})"
        last = v
        if v != -1 and not (isinstance(v, str) and v.startswith("읽기실패")):
            return v
        time.sleep(delay)
    return last


def _set_with_retry(mc, servo_no, data_id, value, retries=RETRY_COUNT, delay=RETRY_DELAY_SEC):
    """set_servo_data 후 읽어서 반영 확인, 실패하면 재시도. (성공여부, 최종읽은값) 반환."""
    last_read = None
    last_exc = None
    for attempt in range(retries):
        try:
            mc.set_servo_data(servo_no, data_id, value)
        except Exception as e:
            # [8차 점검] 예전엔 그냥 pass로 삼켰다 - 어차피 아래에서 읽어서 확인하므로
            # 동작은 안전했지만, 통신 예외 내용이 완전히 사라져 -1 오류 같은 걸
            # 진단할 단서가 없었다. 마지막 예외를 남겨뒀다가 전부 실패했을 때만 알린다.
            last_exc = e
        time.sleep(delay)
        last_read = _get_with_retry(mc, servo_no, data_id, retries=2, delay=delay)
        if last_read == value:
            return True, last_read
    if last_exc is not None:
        print(f"     (J{servo_no} 쓰기 중 마지막 예외: {last_exc})")
    return False, last_read


def read_all(mc):
    """전체 관절의 현재 PID 모드 값을 조회만 한다 (쓰기 없음). 재시도 포함."""
    print(f"\n=== data_id={PID_MODE_DATA_ID} 현재값 (관절별, 재시도 최대 {RETRY_COUNT}회) ===")
    values = {}
    for j in range(1, JOINT_COUNT + 1):
        v = _get_with_retry(mc, j, PID_MODE_DATA_ID)
        values[j] = v
        note = ""
        if v == PID_MODE_SMOOTH:
            note = " -> 부드러운 모드"
        elif v == PID_MODE_PRECISE:
            note = " -> 고정밀 모드"
        elif v == -1:
            note = " -> ⚠️ 재시도해도 오류(-1) - 통신 문제 지속, 서보 케이블/연결 확인 필요"
        else:
            note = " -> 알려진 두 프리셋(0/4)과 다른 값 - 320 M5에서는 의미가 다를 수 있음"
        print(f"  J{j}: {v}{note}")
        time.sleep(INTER_SERVO_DELAY_SEC)
    return values


def parse_joint_selection(raw, joint_count=JOINT_COUNT):
    """'3', '3,4', 'all' 같은 입력을 관절 번호 리스트로 변환."""
    raw = raw.strip().lower()
    if raw in ("all", "전체", "*"):
        return list(range(1, joint_count + 1))
    joints = []
    for tok in raw.replace(" ", "").split(","):
        if not tok:
            continue
        try:
            n = int(tok)
        except ValueError:
            continue
        if 1 <= n <= joint_count:
            joints.append(n)
    return sorted(set(joints))


def write_mode(mc, joints, mode_value, mode_label):
    if not joints:
        print("⚠️ 유효한 관절 번호가 없습니다. 취소합니다.")
        return

    print(f"\n대상 관절: {['J'+str(j) for j in joints]}  ->  {mode_label}(value={mode_value})로 변경합니다.")
    before = {}
    for j in joints:
        before[j] = _get_with_retry(mc, j, PID_MODE_DATA_ID)
        time.sleep(INTER_SERVO_DELAY_SEC)
    print("변경 전 값:", {f"J{j}": v for j, v in before.items()})

    confirm = input("정말로 진행할까요? (yes 입력 시 진행): ").strip().lower()
    if confirm != "yes":
        print("취소했습니다.")
        return

    print(f"\n관절당 최대 {RETRY_COUNT}회 재시도하며 순차 처리합니다 (버스 혼잡 방지를 위해 천천히 진행)...")
    results = {}
    for j in joints:
        ok, final_val = _set_with_retry(mc, j, PID_MODE_DATA_ID, mode_value)
        results[j] = (ok, final_val)
        status = "✅ 성공" if ok else f"⚠️ 실패(최종읽은값={final_val})"
        print(f"  J{j}: {status}")
        time.sleep(INTER_SERVO_DELAY_SEC)

    print("\n=== 최종 결과 ===")
    ok_joints = [j for j, (ok, _) in results.items() if ok]
    fail_joints = [j for j, (ok, _) in results.items() if not ok]
    if ok_joints:
        print(f"✅ {mode_label} 반영 성공: {['J'+str(j) for j in ok_joints]}")
    if fail_joints:
        print(f"⚠️ 반영 실패(재시도 {RETRY_COUNT}회 후에도): {['J'+str(j) for j in fail_joints]}")
        print("   -> 버스 혼잡이 계속되면 한 번에 1~2개 관절씩만 처리해보세요.")
        print("   -> 이 스크립트를 다시 실행해 [1]로 재확인하는 것도 방법입니다.")

    if ok_joints:
        print("\n   -> 이제 mycobot_vibration_diagnostic.py로 같은 자세에서 다시 재보세요.")
        print("   -> 정밀도 저하 여부도 mycobot_static_error_test.py나 곡선 실행으로 확인하세요.")


def main():
    print("🔌 로봇 포트 탐색 중...")
    mc, port = find_robot_port()
    if mc is None:
        print("❌ 로봇을 찾지 못했습니다. USB 연결/권한을 확인하세요.")
        sys.exit(1)
    print(f"✅ 연결됨: {port}")

    print(
        "\n[주의] data_id=23 프리셋은 myCobot 280 Pi 공식 문서 기준입니다.\n"
        "320 M5에서 동일한 의미인지 먼저 [1]로 확인하세요."
    )

    while True:
        print("\n--- 메뉴 ---")
        choice = input(
            "  [1] 전체 관절 현재값 조회 (쓰기 없음)\n"
            "  [2] 선택 관절을 '부드러운 모드'(0)로 변경\n"
            "  [3] 선택 관절을 '고정밀 모드'(4)로 변경 (되돌리기)\n"
            "  [q] 종료\n"
            "> "
        ).strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            read_all(mc)
        elif choice == "2":
            raw = input("대상 관절 번호(예: 3 / 3,4 / all): ")
            joints = parse_joint_selection(raw)
            write_mode(mc, joints, PID_MODE_SMOOTH, "부드러운 모드")
        elif choice == "3":
            raw = input("대상 관절 번호(예: 3 / 3,4 / all): ")
            joints = parse_joint_selection(raw)
            write_mode(mc, joints, PID_MODE_PRECISE, "고정밀 모드")
        else:
            print("잘못된 입력입니다.")

    print("종료합니다.")


if __name__ == "__main__":
    main()
