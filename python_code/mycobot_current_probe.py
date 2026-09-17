# -*- coding: utf-8 -*-
"""
mycobot_current_probe.py
=========================
[신규 프로젝트, 사전확인] 동역학 식별의 `u`(모터 입력)를 읽을 수 있는가?

왜 이걸 먼저 하는가
--------------------
계획서 1단계가 "자세를 고정하고 u2를 측정"인데, 이건 버티는 데 드는
토크(=전류)를 읽는다는 뜻이다. 그런데 myCobot 320 M5의 공개 API에는
`get_servo_voltages` / `get_servo_temps` / `get_servo_status`는 있어도
전류 읽기가 명시돼 있지 않다. 서보 레지스터를 직접 읽는
`get_servo_data(id, data_id)`로 우회될 수도 있지만 확인이 필요하다.

**이게 안 되면 T1~T12 데이터표의 u 열이 통째로 비고, 접근 자체를 바꿔야
한다**(토크 대신 명령각-실측각 추종오차를 쓰는 방식). 그래서 데이터 수집을
시작하기 전에 여기서 먼저 가린다.

무엇을 확인하는가
------------------
1. **존재**: 전류로 보이는 API/레지스터가 값을 돌려주는가
2. **의미**: 그 값이 정말 부하를 반영하는가
   - 숫자가 나오는 것만으로는 부족하다. 온도든 무의미한 레지스터든 숫자는
     나온다. **부하를 바꿨을 때 값이 따라 움직여야** 전류다.
   - J2를 수평(중력 토크 최대)과 수직(최소)으로 놓고 값을 비교한다.
3. **속도**: 초당 몇 번 읽히는가
   - 이전 프로젝트에서 `get_angles()` 하나로 이미 33Hz가 한계였다(§45).
     매 사이클 6축 전류까지 읽으면 그 주기가 더 느려진다. T1~T12는 이
     주기로 q,q̇,q̈,u를 전부 남겨야 하므로 실측이 필요하다.

사용법
-------
    python3 mycobot_current_probe.py              # 1,3단계만 (로봇 안 움직임)
    python3 mycobot_current_probe.py --move       # 2단계 포함 (J2를 움직임)

**--move는 로봇이 실제로 움직인다.** 주변을 비우고 비상정지에 손을 두고
실행할 것. 이동은 느린 속도(20)로만 하고, 매 동작 전에 확인을 받는다.
"""

import argparse
import time

from mycobot_kinematics import find_robot_port

# Feetech STS/SCS 계열 서보의 레지스터 주소 후보.
# 어느 것이 전류인지 문서로 확정할 수 없으므로 후보를 훑어서 **부하에 따라
# 변하는 것**을 찾는다(2단계). 아래 숫자는 "여기부터 훑어본다"는 범위일 뿐
# 확정된 의미가 아니다 - 값이 나온다고 전류로 단정하지 말 것.
REG_CANDIDATES = list(range(53, 72))

# J2를 이 두 자세로 놓고 비교한다. 수평일 때 중력 토크가 최대, 수직일 때 최소.
# 차이가 크게 날수록 전류 레지스터를 식별하기 쉽다.
POSE_LOADED = [0, 0, 0, 0, 0, 0]        # J2=0 : 팔뚝이 수평 -> 중력 부하 큼
POSE_UNLOADED = [0, -90, 0, 0, 0, 0]    # J2=-90: 세워짐    -> 중력 부하 작음


def step1_list_apis(mc):
    """전류로 보이는 공개 API가 있는지 훑는다."""
    print("\n[1단계] 전류 관련 API 탐색")
    names = [n for n in dir(mc)
             if not n.startswith("_") and any(k in n.lower()
                                              for k in ("current", "servo_data", "voltage", "temp", "load"))]
    if not names:
        print("  전류/서보 관련 메서드를 찾지 못했습니다.")
        return []
    found = []
    for n in sorted(names):
        try:
            fn = getattr(mc, n)
            if not callable(fn):
                continue
            # 인자 없는 것만 호출해본다 - 인자가 필요한 것(get_servo_data 등)은
            # 2단계에서 레지스터를 지정해 따로 다룬다.
            try:
                val = fn()
            except TypeError:
                print(f"  {n:<28} (인자 필요 - 2단계에서 확인)")
                found.append(n)
                continue
            print(f"  {n:<28} -> {val}")
            found.append(n)
        except Exception as e:
            print(f"  {n:<28} -> 호출 실패: {type(e).__name__}")
    return found


def read_regs(mc, joint_id, regs):
    """한 관절의 여러 레지스터를 읽어 {주소: 값} 으로 돌려준다."""
    out = {}
    for addr in regs:
        try:
            v = mc.get_servo_data(joint_id, addr)
        except Exception:
            v = None
        out[addr] = v
    return out


def step2_load_test(mc, joint_id=2):
    """부하를 바꿔가며 값이 따라 움직이는 레지스터를 찾는다.

    **핵심**: 숫자가 나오는 레지스터가 전류인 게 아니라, **부하에 반응하는
    레지스터**가 전류다. 여기를 건너뛰면 온도나 무의미한 값을 u로 쓰게 된다.
    """
    print(f"\n[2단계] 부하 반응 검사 (J{joint_id})")
    print("  ⚠️ 로봇이 움직입니다. 주변을 비우고 비상정지에 손을 두세요.")
    if input("  진행하려면 yes 입력: ").strip().lower() != "yes":
        print("  건너뜁니다.")
        return None

    results = {}
    for label, pose in (("부하 큼(J2 수평)", POSE_LOADED),
                        ("부하 작음(J2 수직)", POSE_UNLOADED)):
        print(f"\n  -> {label} 자세로 이동 중...")
        mc.send_angles(pose, 20)
        time.sleep(4.0)                  # 도달 + 정착 대기
        # 정지 상태에서 여러 번 읽어 평균낸다 - 한 번만 읽으면 노이즈와
        # 부하 변화를 구분할 수 없다.
        samples = []
        for _ in range(10):
            samples.append(read_regs(mc, joint_id, REG_CANDIDATES))
            time.sleep(0.05)
        avg = {}
        for addr in REG_CANDIDATES:
            vals = [s[addr] for s in samples if isinstance(s[addr], (int, float))]
            avg[addr] = sum(vals) / len(vals) if vals else None
        results[label] = avg
        print(f"     {len([v for v in avg.values() if v is not None])}개 레지스터가 값을 반환")

    print(f"\n  {'주소':>6}{'부하 큼':>12}{'부하 작음':>12}{'차이':>10}")
    keys = list(results.keys())
    changed = []
    for addr in REG_CANDIDATES:
        a = results[keys[0]][addr]
        b = results[keys[1]][addr]
        if a is None or b is None:
            continue
        d = a - b
        mark = "  <- 부하 반응" if abs(d) > 5 else ""
        print(f"  {addr:>6}{a:>12.1f}{b:>12.1f}{d:>+10.1f}{mark}")
        if abs(d) > 5:
            changed.append((addr, d))

    print()
    if changed:
        print("  부하에 반응한 레지스터:", ", ".join(str(a) for a, _ in changed))
        print("  -> 이 중 하나가 전류일 가능성이 높습니다. 다만 온도도 부하를")
        print("     따라 천천히 오르므로, 자세를 여러 번 왕복시켜 **즉시** 따라오는지")
        print("     확인해야 확정할 수 있습니다(온도는 느리게 변합니다).")
    else:
        print("  ★ 부하에 반응한 레지스터가 없습니다.")
        print("    전류를 읽을 수 없다는 뜻일 수 있습니다 - 그렇다면 u 열을")
        print("    토크로 채우는 계획을 바꿔야 합니다.")
    return changed


def step3_rate(mc, joint_ids=(1, 2, 3, 4, 5, 6), addr=None, n=50):
    """읽기 속도 측정 - T1~T12를 이 주기로 기록할 수 있는지 본다."""
    print("\n[3단계] 읽기 속도 측정")

    t0 = time.perf_counter()
    for _ in range(n):
        mc.get_angles()
    dt_ang = (time.perf_counter() - t0) / n
    print(f"  get_angles() 단독            : {dt_ang*1000:6.1f} ms/회  ({1/dt_ang:5.1f} Hz)")

    if addr is None:
        print("  (전류 레지스터가 확정되지 않아 전류 읽기 속도는 생략)")
        return
    t0 = time.perf_counter()
    for _ in range(n):
        for jid in joint_ids:
            mc.get_servo_data(jid, addr)
    dt_cur = (time.perf_counter() - t0) / n
    print(f"  6축 전류 읽기(주소 {addr})      : {dt_cur*1000:6.1f} ms/회  ({1/dt_cur:5.1f} Hz)")

    total = dt_ang + dt_cur
    print(f"  각도+전류 합산               : {total*1000:6.1f} ms/회  ({1/total:5.1f} Hz)")
    print()
    print(f"  참고: 이전 프로젝트의 실행 주기는 48ms(약 21Hz)였습니다.")
    if total > 0.048:
        print(f"  ★ 합산 {total*1000:.0f}ms > 48ms - 기존 주기를 유지할 수 없습니다.")
        print("    T1~T12를 더 느린 주기로 돌리거나, 전류를 매 사이클이 아니라")
        print("    간헐적으로만 읽는 식으로 계획을 조정해야 합니다.")
    else:
        print(f"  합산이 48ms 안에 들어옵니다 - 기존 주기로 u까지 기록 가능.")


def main():
    ap = argparse.ArgumentParser(description="서보 전류 읽기 가능 여부 진단")
    ap.add_argument("--move", action="store_true",
                    help="2단계(부하 반응 검사) 포함 - 로봇이 실제로 움직입니다")
    ap.add_argument("--addr", type=int, default=None,
                    help="전류 레지스터 주소를 이미 안다면 지정 (3단계 속도측정용)")
    args = ap.parse_args()

    mc, port = find_robot_port()
    if mc is None:
        print("로봇을 찾지 못했습니다 - 전원과 USB 연결을 확인하세요.")
        return

    step1_list_apis(mc)

    addr = args.addr
    if args.move:
        changed = step2_load_test(mc)
        if changed and addr is None:
            addr = max(changed, key=lambda x: abs(x[1]))[0]   # 가장 크게 반응한 것
    else:
        print("\n[2단계] 건너뜀 - 부하 반응 검사를 하려면 --move 를 붙이세요.")
        print("  ※ 값이 나온다고 전류인 건 아닙니다. 이 단계 없이는 확정 못 합니다.")

    step3_rate(mc, addr=addr)

    print("\n결과를 그대로 복사해 보내주시면 다음 단계를 정하겠습니다.")


if __name__ == "__main__":
    main()
