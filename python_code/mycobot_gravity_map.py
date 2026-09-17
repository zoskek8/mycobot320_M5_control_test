# -*- coding: utf-8 -*-
"""
mycobot_gravity_map.py
=======================
[신규 프로젝트, §1] 정적 중력 매핑 - 자세별로 버티는 데 드는 모터 부하 측정.

계획서 1단계를 수행한다: 한 관절을 여러 각도로 놓고 그 자세를 유지하는 데
드는 값을 재서 `g(q)`를 매핑한다.

무엇을 재는가 - 전류가 아니라 '부하'
--------------------------------------
1차 스윕에서 확인된 사실:
  - `get_servo_currents()`와 서보 레지스터 69가 모든 자세에서 같은 값을
    돌려줬다. API가 그 레지스터를 읽는 것이다.
  - **그런데 전류는 쓸 수 없다.** 값이 0/4/8.4/12뿐으로 4단위 양자화돼
    있고 전 구간 범위가 0~12다. 9개 자세 중 4개가 정확히 0이었다.
    이 해상도로 g(q)를 매핑할 수 없다.
  - **부하(주소 60)는 범위가 8~112로 14배 넓고** 자세를 따라 V자를
    그렸다(J2=0에서 최소). 이쪽을 u의 대리값으로 쓴다.

레지스터 주소는 실측으로 확정했다: 62 -> 234.x (실제 전압 23.4V),
63 -> 26 (get_servo_temps의 J2와 일치), 69 -> get_servo_currents와 일치.
세 개가 맞으므로 같은 표(Feetech STS 계열)의 60=Present Load도 맞다.

1차 스윕에서 남은 문제 두 가지
--------------------------------
(a) **좌우 비대칭**: -45도에서 112인데 +45도에서 72. 중력만이라면 대칭이어야
    한다. 게다가 -60(96)이 -45(112)보다 작아 |sin| 모양과도 어긋난다.
    유력한 원인은 **접근 방향에 따른 마찰 이력** - 1차는 한 방향으로만
    훑었고, 서보는 데드밴드 안 어딘가에서 멈추는데 그 지점이 어느 쪽에서
    왔느냐에 따라 달라진다. 이전 프로젝트 §48에서 같은 교란변수에 걸렸다.
    -> **양방향 스윕**으로 분리한다. 두 방향 차이가 사라지면 마찰 이력이
       원인이었던 것이 확정되고, 남으면 다른 원인을 찾아야 한다.
(b) **부하 2바이트 여부**: STS 계열 Present Load는 2바이트이고 방향 비트가
    들어 있다. 하위 바이트만 읽으면 255를 넘을 때 값이 돌아가고 부호를
    잃는다. 지금은 최대 112라 괜찮지만 주행 중에는 넘을 수 있다.
    -> 60과 61을 같이 읽어 분해한다.

사용법
-------
    python3 mycobot_gravity_map.py --joint 2
    python3 mycobot_gravity_map.py --joint 2 --lo -60 --hi 60 --step 15
    python3 mycobot_gravity_map.py --joint 3 --hold 0,30,0,0,0,0   # 커플링(계획 3단계)
    python3 mycobot_gravity_map.py --bypass-only   # 프레임 우회 속도만 확인

**로봇이 실제로 움직입니다.** 주변을 비우고 비상정지에 손을 두세요.
출력은 gravity_map_J<n>.csv 로 저장되어 MATLAB 등에서 바로 읽힙니다.
"""

import argparse
import csv
import time

from mycobot_kinematics import find_robot_port

REG_LOAD_L = 60      # Present Load 하위바이트
REG_LOAD_H = 61      # Present Load 상위바이트 (방향비트 포함)
REG_VOLTAGE = 62     # 확정됨 - 실제 전압의 10배
REG_TEMP = 63        # 확정됨
REG_CURRENT = 69     # 확정됨 - 다만 해상도가 낮아 주력으로 못 씀

SETTLE_SEC = 2.5
N_SAMPLE = 8
TEMP_LIMIT_C = 55


def decode_load(lo, hi):
    """2바이트 Present Load를 크기와 방향으로 분해한다.

    STS 계열은 하위 10비트가 크기, 비트10이 방향이다. 하위바이트만 보면
    255를 넘을 때 값이 돌아가고 방향을 잃으므로 둘 다 읽어 합친다.
    """
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return None, None, None
    raw = int(lo) + int(hi) * 256
    mag = raw & 0x3FF
    direction = (raw >> 10) & 1
    signed = -mag if direction else mag
    return raw, mag, signed


def read_all(mc, joint_id):
    out = {}
    try:
        cur = mc.get_servo_currents()
        out["api_current"] = cur[joint_id - 1] if isinstance(cur, list) and len(cur) >= 6 else None
    except Exception:
        out["api_current"] = None
    for key, addr in (("load_l", REG_LOAD_L), ("load_h", REG_LOAD_H),
                      ("voltage", REG_VOLTAGE), ("temp", REG_TEMP),
                      ("current", REG_CURRENT)):
        try:
            out[key] = mc.get_servo_data(joint_id, addr)
        except Exception:
            out[key] = None
    return out


def avg(samples, key):
    vals = [s[key] for s in samples if isinstance(s[key], (int, float))]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------
# 프레임 우회 - 프로토콜 코드를 추측하지 않고 실제 출력에서 학습한다
# ---------------------------------------------------------------
def learn_request_frame(mc, joint_id, addr):
    """pymycobot이 get_servo_data를 부를 때 내보내는 바이트를 가로챈다.

    프로토콜 코드를 문서에서 추측하면 엉뚱한 레지스터를 읽을 수 있다.
    실제로 나가는 프레임을 그대로 복사하면 그 위험이 없다.
    반환: (요청바이트, pymycobot이 돌려준 값)
    """
    sp = mc._serial_port
    captured = []
    orig_write = sp.write

    def spy(data):
        captured.append(bytes(data))
        return orig_write(data)

    sp.write = spy
    try:
        val = mc.get_servo_data(joint_id, addr)
    finally:
        sp.write = orig_write
    if not captured:
        return None, val
    return captured[-1], val


def fast_read_reg(mc, request, timeout_sec=0.05):
    """학습한 요청 프레임을 직접 써서 응답의 데이터 바이트를 돌려준다.

    응답 형식은 FE FE <len> <cmd> <data...> FA. 데이터 길이는 레지스터마다
    다를 수 있어 payload 전체를 돌려주고, 해석은 호출자가 한다.
    """
    try:
        sp = mc._serial_port
        sp.reset_input_buffer()
        sp.write(request)
        deadline = time.time() + timeout_sec
        buf = b""
        while time.time() < deadline:
            w = sp.in_waiting
            if w:
                buf += sp.read(w)
                i = 0
                while i + 4 <= len(buf):
                    if buf[i] != 0xFE or buf[i + 1] != 0xFE:
                        i += 1
                        continue
                    dlen = buf[i + 2]
                    end = i + 3 + dlen
                    if end > len(buf):
                        break
                    return buf[i + 4:end - 1]
                # 헤더는 찾았지만 프레임이 덜 왔으면 더 기다린다
            else:
                time.sleep(0.0005)
        return None
    except Exception:
        return None


def try_bypass(mc, joint_id=2, addr=REG_LOAD_L, n=30):
    """우회 경로가 pymycobot과 같은 값을 주는지 확인하고 속도를 잰다."""
    print("\n[프레임 우회 시도]")
    req, ref = learn_request_frame(mc, joint_id, addr)
    if req is None:
        print("  요청 프레임을 가로채지 못했습니다 - 우회 불가, pymycobot 경로를 씁니다.")
        return None
    print(f"  학습한 요청: {req.hex(' ')}   (pymycobot 반환값 {ref})")

    payload = fast_read_reg(mc, req)
    if payload is None:
        print("  ★ 우회 읽기 응답을 못 받았습니다 - pymycobot 경로를 씁니다.")
        return None
    print(f"  우회 응답 payload: {payload.hex(' ')}")

    # payload를 정수로 해석해 pymycobot 값과 대조한다.
    cand = None
    if len(payload) == 1:
        cand = payload[0]
    elif len(payload) >= 2:
        cand = int.from_bytes(payload[:2], "big")
    print(f"  해석값 {cand}  vs  pymycobot {ref}")
    if cand != ref:
        print("  ★ 값이 다릅니다 - 해석 규칙이 맞지 않으므로 우회를 쓰지 않습니다.")
        print("    (틀린 값으로 데이터를 모으느니 느린 경로가 낫습니다)")
        return None

    t0 = time.perf_counter()
    ok = 0
    for _ in range(n):
        if fast_read_reg(mc, req) is not None:
            ok += 1
    dt = (time.perf_counter() - t0) / n
    print(f"  ✅ 값 일치. 우회 읽기 {dt*1000:.1f} ms/회 ({1/dt:.0f} Hz), 성공 {ok}/{n}")
    print(f"     (pymycobot 경로는 약 30ms - {30/(dt*1000):.0f}배)")
    return req


def measure_rate(mc, n=20):
    print("\n[읽기 속도 - pymycobot 경로]")
    t0 = time.perf_counter()
    for _ in range(n):
        mc.get_angles()
    a = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        mc.get_servo_data(2, REG_LOAD_L)
    b = (time.perf_counter() - t0) / n
    print(f"  get_angles()           : {a*1000:6.1f} ms  ({1/a:5.1f} Hz)")
    print(f"  get_servo_data(부하)   : {b*1000:6.1f} ms  ({1/b:5.1f} Hz)")
    print(f"  합산                   : {(a+b)*1000:6.1f} ms  ({1/(a+b):5.1f} Hz)")
    return a, b


def sweep(mc, joint, base, angles, label):
    """한 방향으로 훑으며 각 자세의 값을 모은다."""
    rows = []
    print(f"\n  --- {label} ---")
    print(f"  {'각도':>8}{'부하':>9}{'방향':>6}{'raw':>7}{'전류':>8}{'온도':>7}")
    for ang in angles:
        pose = list(base)
        pose[joint - 1] = ang
        mc.send_angles(pose, 20)
        time.sleep(SETTLE_SEC)

        samples = [read_all(mc, joint) for _ in range(N_SAMPLE)]
        lo = avg(samples, "load_l")
        hi = avg(samples, "load_h")
        raw, mag, signed = decode_load(lo, hi)
        r = {
            "joint": joint, "angle_deg": ang, "direction": label,
            "hold_pose": "|".join(str(x) for x in base),
            "load_raw": raw, "load_mag": mag, "load_signed": signed,
            "load_l": lo, "load_h": hi,
            "api_current": avg(samples, "api_current"),
            "voltage": avg(samples, "voltage"),
            "temp": avg(samples, "temp"),
        }
        rows.append(r)

        def f(v, w=9, p=1):
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"
        d = "-" if (isinstance(hi, (int, float)) and int(hi) & 0x04) else "+"
        print(f"{ang:>8.1f}{f(mag)}{d:>6}{f(raw,7,0)}{f(r['api_current'],8)}{f(r['temp'],7,0)}")

        if isinstance(r["temp"], (int, float)) and r["temp"] > TEMP_LIMIT_C:
            print(f"\n  ★ 서보 온도 {r['temp']:.0f}도 - 중단합니다.")
            break
    return rows


def main():
    ap = argparse.ArgumentParser(description="정적 중력 매핑 (양방향 스윕)")
    ap.add_argument("--joint", type=int, default=2)
    ap.add_argument("--lo", type=float, default=-60.0)
    ap.add_argument("--hi", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=15.0)
    ap.add_argument("--hold", type=str, default=None,
                    help="나머지 관절 고정값 6개, 쉼표 구분 (커플링 실험용)")
    ap.add_argument("--bypass-only", action="store_true",
                    help="스윕 없이 프레임 우회 검증/속도측정만 (로봇 안 움직임)")
    args = ap.parse_args()

    mc, port = find_robot_port()
    if mc is None:
        print("로봇을 찾지 못했습니다.")
        return

    measure_rate(mc)
    try_bypass(mc, joint_id=args.joint, addr=REG_LOAD_L)

    if args.bypass_only:
        print("\n--bypass-only 이므로 스윕은 생략합니다.")
        return

    base = [0.0] * 6
    if args.hold:
        try:
            base = [float(x) for x in args.hold.split(",")]
            assert len(base) == 6
        except Exception:
            print("--hold 는 쉼표로 구분한 6개 숫자여야 합니다.")
            return

    up = []
    a = args.lo
    while a <= args.hi + 1e-9:
        up.append(round(a, 2))
        a += args.step
    down = list(reversed(up))

    print(f"\nJ{args.joint}: {up[0]}도 ~ {up[-1]}도, {args.step}도 간격, **양방향**")
    print(f"  나머지 관절 고정: {base}")
    print(f"  1차 스윕의 좌우 비대칭(-45:112 vs +45:72)이 접근 방향 탓인지 가린다.")
    print(f"  예상 소요: 약 {2*len(up)*(SETTLE_SEC+N_SAMPLE*0.25+1)/60:.1f}분")
    print("\n  ⚠️ 로봇이 실제로 움직입니다. 주변을 비우고 비상정지에 손을 두세요.")
    if input("  진행하려면 yes 입력: ").strip().lower() != "yes":
        print("  중단합니다.")
        return

    rows = sweep(mc, args.joint, base, up, "up")
    rows += sweep(mc, args.joint, base, down, "down")

    out = f"gravity_map_J{args.joint}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: {out}  ({len(rows)}행)")

    # 판정 - 방향 차이와 좌우 대칭
    byang = {}
    for r in rows:
        byang.setdefault(r["angle_deg"], {})[r["direction"]] = r["load_mag"]
    print("\n[판정 1] 접근 방향에 따른 차이")
    diffs = []
    print(f"  {'각도':>8}{'올라가며':>10}{'내려오며':>10}{'차이':>9}")
    for ang in sorted(byang):
        u = byang[ang].get("up")
        d = byang[ang].get("down")
        if isinstance(u, (int, float)) and isinstance(d, (int, float)):
            diffs.append(abs(u - d))
            print(f"  {ang:>8.1f}{u:>10.1f}{d:>10.1f}{u-d:>+9.1f}")
    if diffs:
        m = sum(diffs) / len(diffs)
        print(f"\n  평균 |차이| = {m:.1f}")
        if m > 10:
            print("  -> 접근 방향이 값을 크게 바꿉니다. 마찰 이력이 실재하므로")
            print("     앞으로 모든 정적 측정은 양방향 평균을 써야 합니다.")
        else:
            print("  -> 방향 차이가 작습니다. 1차의 비대칭은 다른 원인입니다.")

    print("\n[판정 2] 좌우 대칭 (양방향 평균 기준)")
    for ang in sorted(byang):
        if ang <= 0:
            continue
        pos = byang.get(ang, {})
        neg = byang.get(-ang, {})
        pv = [v for v in pos.values() if isinstance(v, (int, float))]
        nv = [v for v in neg.values() if isinstance(v, (int, float))]
        if pv and nv:
            p, n = sum(pv) / len(pv), sum(nv) / len(nv)
            print(f"  ±{ang:>5.1f}도 : +{p:>7.1f}  -{n:>7.1f}   차이 {p-n:+.1f}")
    print("\n  중력만이라면 좌우가 비슷해야 합니다. CSV를 보내주시면 같이 보겠습니다.")


if __name__ == "__main__":
    main()
