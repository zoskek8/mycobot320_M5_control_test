# -*- coding: utf-8 -*-
"""
mycobot_traj_collect.py
========================
[신규 프로젝트, §4.2] T1~T12 궤적 데이터 수집. 우선 **T1** 부터.

계획 5단계의 표를 그대로 만든다:

    time | q1..q6 | qd1..qd6 | qdd1..qdd6 | u1..u6

T1 이란
--------
**J1 만** 속도가 사인파를 그리며 회전. 나머지 관절은 자세 고정.
(T2 는 J2 만, T3 는 J3 만 - `--joint` 로 지정한다.)

속도를 사인파로 주면 위치와 가속도가 수식으로 나온다:

    q̇(t) = V·sin(ωt)
    q(t)  = q0 + (V/ω)·(1 - cos(ωt))
    q̈(t) = V·ω·cos(ωt)

**이게 중요한 이유**: 60ms 주기(§2.4) 데이터를 두 번 미분해 q̈ 을 구하면
노이즈가 심하다. 지령이 해석적 함수면 q̇/q̈ 를 미분 없이 정확히 안다.
T1~T3 에서 사인파를 쓰는 것의 실질적 이점이 이것이다.

u 를 어떻게 다 채우나 - 다중 패스
-----------------------------------
서보 부하 읽기는 **관절당 30ms x 2바이트 = 60ms** 이고 이건 펌웨어 왕복
한계라 우회할 수 없다(§2.4에서 프레임 직접 조립으로도 30ms 그대로였다).
6관절을 매 사이클 읽으면 360ms + 각도 30ms = 2.6Hz 로, 궤적을 담을 수 없다.

그래서 **같은 궤적을 여러 번 돌리며 패스마다 다른 관절의 부하를 읽는다.**
지령이 해석적이고 매번 동일하므로 시간축으로 합칠 수 있다. 한 패스는
각도 30ms + 부하 60ms = 약 90ms(11Hz) 로 돌아간다.

패스 간 재현성은 자동으로 확인한다 - 각 패스의 실측 q 를 비교해서
어긋나면 경고한다. 어긋나면 합치면 안 되는 데이터다.

q, q̇, q̈ 를 무엇으로 채우나
-----------------------------
- `q1..q6`   : **실측** 각도(get_angles). 실제로 로봇이 있던 자세다.
- `qd`, `qdd`: **지령 궤적의 해석적 미분**. 위 이유로 실측 미분을 쓰지 않는다.
- `q_cmd*`   : 지령 각도. 추종오차를 보려면 q 와 비교하면 된다.
- `track_err`: max|실측 - 지령|. 이 값이 크면 그 구간은 신뢰할 수 없다.

실측 q 와 지령 기반 q̇/q̈ 를 섞어 쓰는 것이므로, **추종오차가 작을 때만
유효하다.** track_err 열로 반드시 확인할 것.

사용법
-------
    python3 mycobot_traj_collect.py --dry              # 궤적만 확인(로봇 불필요)
    python3 mycobot_traj_collect.py --joint 1          # T1 실행(J1 부하만)
    python3 mycobot_traj_collect.py --joint 1 --multipass   # u1~u6 전부
    python3 mycobot_traj_collect.py --joint 2 --amp 25 --period 6

**로봇이 실제로 움직입니다.** 주변을 비우고 비상정지에 손을 두세요.
"""

import argparse
import csv
import math
import time

import numpy as np

from mycobot_kinematics import (
    find_robot_port, fast_send_angles, fast_get_angles,
    JOINT_LIMITS_DEG, chain, active_indices,
    check_self_collision, DESK_SAFETY_MARGIN_MM,
)

REG_LOAD_L, REG_LOAD_H = 60, 61
# send_angles 의 speed 인자는 **로봇 자체의 이동 속도 상한**이다(0~100).
# 1차 실행에서 60 으로 뒀더니 실측 속도가 13~15도/s 에서 평평해졌다 - 사인의
# 최대 24도/s 를 못 내고 상한에 잘린 것이다. 빠른 구간에서 밀리고 느린
# 구간에서 33도/s 로 따라잡는 계단 파형이 나왔다.
# 지령 속도의 최대값이 이 상한보다 충분히 낮아야 사인이 살아난다.
SEND_SPEED = 100     # --send-speed 로 덮어쓴다


def decode_load(lo, hi):
    """2바이트 Present Load -> 부호 있는 값. §2.3 참고 - 하위바이트만 읽으면
    255를 넘을 때 값이 돌아가고 방향을 잃는다."""
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return None
    raw = int(lo) + int(hi) * 256
    mag = raw & 0x3FF
    return -mag if (raw >> 10) & 1 else mag


def traj(t, q0, v_mean, ratio, period_sec, direction=+1):
    """**한 방향으로만** 도는 궤적. 속도가 평균 위에서 사인으로 물결친다.

        q̇(t) = v0 + v1·sin(ωt)            v1 = ratio·v0,  ratio < 1 이면 항상 > 0
        q(t)  = q0 + v0·t + (v1/ω)(1 - cos ωt)     단조 증가
        q̈(t) = v1·ω·cos(ωt)

    [1차 시도 실패 - q̇ = V·sin(ωt)]
    순수 사인은 반주기마다 음수가 되어 **방향이 뒤집힌다.** 전환할 때마다
    쿨롱 마찰의 부호가 뒤집히고(§3.9에서 J2 기준 -9.7 로 실측), 속도 0 을
    지날 때 정지마찰이, 전환점에서 백래시가 끼어든다. 전부 동역학 식별에
    교란으로 들어간다.

    [2차도 부족 - q̇ = V₀(1-cos ωt)]
    방향 전환은 없지만 **매 주기마다 속도가 정확히 0 을 찍는다.** 그 순간
    정지마찰이 다시 끼어든다.

    [현재] ratio<1 이면 속도가 v0(1-ratio) 이상으로 **항상 양수**다. 0 을
    건드리지 않으므로 마찰이 한 방향의 상수 오프셋으로만 남는다.
    계획서 그래프도 0 을 가로지르는 사인이 아니라 **가로선 주위에서
    물결치는 곡선**이다("정속, 개변").
    """
    w = 2.0 * math.pi / period_sec
    v1 = ratio * v_mean
    prog = v_mean * t + (v1 / w) * (1.0 - math.cos(w * t))
    q = q0 + direction * prog
    qd = direction * (v_mean + v1 * math.sin(w * t))
    qdd = direction * (v1 * w * math.cos(w * t))
    return q, qd, qdd


def estimate_lag(ts, q_meas, cmd_fn, max_lag=1.0):
    """실측 q 가 지령보다 얼마나 늦는지 추정한다.

    [왜 필요한가] 1차 실행에서 추종오차가 평균 6.4도, 최대 16.5도였다.
    속도 20도/s 에서 6.4도면 약 320ms 지연이다. 그런데 CSV 의 q 는 실측,
    qd/qdd 는 지령 기준이라 **서로 다른 시각의 값**이 한 행에 섞인다.
    지연 τ 를 추정해 qd/qdd 를 t-τ 에서 평가하면 시각이 맞는다.
    궤적이 해석적이므로 격자 탐색으로 충분하다.
    """
    ts = np.asarray(ts, dtype=float)
    qm = np.asarray(q_meas, dtype=float)
    good = ~np.isnan(qm)
    if good.sum() < 10:
        return 0.0, float("nan")
    best, best_err = 0.0, float("inf")
    for lag in np.arange(0.0, max_lag, 0.005):
        pred = np.array([cmd_fn(t - lag) for t in ts[good]])
        err = float(np.sqrt(((qm[good] - pred) ** 2).mean()))
        if err < best_err:
            best, best_err = float(lag), err
    return best, best_err


def verify_sine(ts, q_meas, v_mean, ratio, period, lag, joint, png=None):
    """**속도가 정말 사인 모양인지** 실측으로 확인한다.

    지령이 사인이어도 로봇이 그대로 따라갔는지는 별개다. 실측 각도를
    수치미분해 해석 속도와 겹쳐 본다. 미분은 노이즈를 키우므로 중앙차분
    후 이동평균으로 평활한다 - 그래도 형태 확인에는 충분하다.
    """
    ts = np.asarray(ts, dtype=float)
    qm = np.asarray(q_meas, dtype=float)
    g = ~np.isnan(qm)
    ts, qm = ts[g], qm[g]
    if len(ts) < 12:
        print("  샘플이 부족해 속도 확인을 건너뜁니다.")
        return
    v_num = np.gradient(qm, ts)
    k = 5
    ker = np.ones(k) / k
    v_smooth = np.convolve(v_num, ker, mode="same")
    v_ana = np.array([traj(t - lag, 0.0, v_mean, ratio, period)[1] for t in ts])

    m = slice(k, -k)     # 평활 경계는 제외
    resid = v_smooth[m] - v_ana[m]
    denom = float(((v_ana[m] - v_ana[m].mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / denom if denom > 1e-9 else float("nan")

    print(f"\n[속도 사인 확인] J{joint}")
    print(f"  해석 속도 범위 {v_ana.min():.1f} ~ {v_ana.max():.1f} 도/s"
          f"  (항상 양수여야 함: {'예' if v_ana.min() > 0 else '★아니오'})")
    print(f"  실측 속도 범위 {v_smooth[m].min():.1f} ~ {v_smooth[m].max():.1f} 도/s")
    print(f"  실측 vs 해석  R² = {r2:.3f}   잔차 RMS = {np.sqrt((resid**2).mean()):.2f} 도/s")
    print(f"  부호가 바뀐 샘플: {(v_smooth[m] < 0).sum()} / {len(v_smooth[m])}"
          f"   (0 이어야 방향 전환이 없다는 뜻)")

    # 각도 읽기가 정체(같은 값 반복)됐는지 - 수치미분이 0과 스파이크를
    # 번갈아 내는 원인이 된다. 로봇이 멈춘 게 아니라 **읽기가 갱신 안 된**
    # 경우를 가린다.
    dq = np.diff(qm)
    same = int((np.abs(dq) < 1e-6).sum())
    runs, cur = [], 0
    for d in dq:
        if abs(d) < 1e-6:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    print(f"\n  [각도 읽기 정체] 직전과 같은 값: {same}/{len(dq)}"
          f" ({100.0*same/max(len(dq),1):.0f}%)"
          + (f", 최장 연속 {max(runs)}회" if runs else ""))
    if same > len(dq) * 0.15:
        print("    ★ 같은 각도가 반복해서 읽혔습니다. 로봇이 멈춘 게 아니라")
        print("      **읽기가 갱신되지 않은 것**일 수 있습니다. 수치미분이 0과")
        print("      스파이크를 번갈아 내므로 속도 파형이 망가집니다.")
        print("      --no-load 로 부하 읽기를 빼고 돌려서 비교해 보세요.")

    # 속도 상한에 잘렸는지 진단 - 1차 실행의 실패 양상이 이것이었다.
    vm = v_smooth[m]
    if r2 < 0.5:
        hi_ana = v_ana[m] > np.percentile(v_ana[m], 75)
        lo_ana = v_ana[m] < np.percentile(v_ana[m], 25)
        gap_hi = float(np.mean(v_ana[m][hi_ana] - vm[hi_ana]))
        gap_lo = float(np.mean(v_ana[m][lo_ana] - vm[lo_ana]))
        print(f"\n  ★ 속도가 사인을 따라가지 못합니다(R²={r2:.2f}).")
        print(f"    빠른 구간에서 지령보다 평균 {gap_hi:+.1f} 도/s,"
              f" 느린 구간에서 {gap_lo:+.1f} 도/s")
        if gap_hi > 2.0 and gap_lo < 0.0:
            print("    -> 빠를 때 못 따라가고 느릴 때 따라잡는 모습입니다.")
            print("       **send_angles 의 speed 상한에 잘린 것**입니다.")
            print("       --calibrate 로 실제 상한을 재고 --speed 를 낮추세요.")
        else:
            print("    -> 주기가 불규칙하거나 지연이 큰 경우일 수 있습니다.")
            print("       실제 주기 출력과 지연 추정값을 확인하세요.")

    step = max(1, len(ts) // 14)
    print(f"\n  {'t':>7}{'실측 속도':>11}{'해석 속도':>11}{'차이':>9}")
    for i in range(0, len(ts), step):
        if i < k or i >= len(ts) - k:
            continue
        print(f"  {ts[i]:>7.2f}{v_smooth[i]:>11.2f}{v_ana[i]:>11.2f}{v_smooth[i]-v_ana[i]:>+9.2f}")

    if png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            ax[0].plot(ts, qm, label="q measured")
            ax[0].set_ylabel("angle (deg)"); ax[0].legend(); ax[0].grid(alpha=.3)
            ax[0].set_title(f"T{joint}: one-way sweep, sinusoidal speed")
            ax[1].plot(ts[m], v_smooth[m], label="qd measured (numeric diff + smooth)")
            ax[1].plot(ts[m], v_ana[m], "--", label="qd analytic (lag-corrected)")
            ax[1].axhline(0, color="k", lw=.8)
            ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("speed (deg/s)")
            ax[1].legend(); ax[1].grid(alpha=.3)
            fig.tight_layout(); fig.savefig(png, dpi=110)
            print(f"\n  그래프 저장: {png}")
        except Exception as e:
            print(f"\n  그래프 저장 실패({type(e).__name__}) - 표로만 확인하세요.")


def total_travel(v_mean, period_sec, cycles):
    return v_mean * period_sec * cycles


def full_q(q6_deg):
    q = [0.0] * len(chain.links)
    for k, idx in enumerate(active_indices):
        q[idx] = math.radians(q6_deg[k])
    return q


def lowest_height_mm(q6_deg):
    """**모든 링크 원점 중 가장 낮은 높이**(mm)와 그 링크 번호.

    [왜 손끝만 보면 안 되나] 처음엔 TCP 높이만 검사했다. J2 사고는 손끝이
    닿은 경우라 잡혔겠지만, 팔을 크게 접으면 **팔꿈치나 손목이 손끝보다
    먼저 바닥에 닿는다.** 링크 원점을 전부 보는 편이 안전하다.

    엄밀한 메시 충돌 검사는 아니다(링크의 부피는 고려하지 않는다) - 그래서
    DESK_SAFETY_MARGIN_MM 의 여유를 남겨두고, 처음 도는 궤적은 눈으로
    지켜볼 것.
    """
    frames = chain.forward_kinematics(full_q(q6_deg), full_kinematics=True)
    best_h, best_i = float("inf"), None
    for k, idx in enumerate(active_indices):
        h = float(frames[idx][2, 3] * 1000.0)
        if h < best_h:
            best_h, best_i = h, k + 1
    tcp = float(frames[-1][2, 3] * 1000.0)
    if tcp < best_h:
        best_h, best_i = tcp, 0          # 0 = TCP
    return best_h, best_i


def tcp_height_mm(q6_deg):
    return lowest_height_mm(q6_deg)[0]


def scan_trajectory(joint, home, q_start, v_mean, ratio, period, cycles, n=400):
    """궤적 전체를 미리 훑어 **책상 충돌과 자가충돌**을 확인한다.

    [왜 넣었나] 관절 한계만 검사하고 실행했다가 J2 를 -125도로 돌려
    **바닥을 박았다.** 한계 안이어도 팔이 뒤로 누우면 손끝이 바닥에 닿는다.
    J1 은 회전축이 수직이라 문제가 없었지만 J2/J3 는 팔이 눕는다.
    이전 프로젝트의 FK 와 자가충돌 검사가 이미 있으므로 그대로 쓴다.

    반환: (안전여부, 메시지, 최저 손끝 높이, 그때의 각도)
    """
    T = period * cycles
    worst_h, worst_a, worst_w = float("inf"), None, None
    coll_at = None
    for i in range(n + 1):
        t = T * i / n
        q = traj(t, q_start, v_mean, ratio, period)[0]
        pose = list(home); pose[joint - 1] = q
        h, which = lowest_height_mm(pose)
        if h < worst_h:
            worst_h, worst_a, worst_w = h, q, which
        if coll_at is None and check_self_collision(full_q(pose)):
            coll_at = q
    msgs = []
    ok = True
    if worst_h < DESK_SAFETY_MARGIN_MM:
        ok = False
        part = "TCP" if worst_w == 0 else f"J{worst_w} 부근"
        msgs.append(f"★ 책상 충돌: J{joint}={worst_a:.0f}도 에서 {part} 높이 "
                    f"{worst_h:.0f}mm (안전선 {DESK_SAFETY_MARGIN_MM}mm)")
    if coll_at is not None:
        ok = False
        msgs.append(f"★ 자가충돌: J{joint}={coll_at:.0f}도 부근")
    return ok, "\n    ".join(msgs), worst_h, worst_a, worst_w


def safe_range(joint, home, margin_deg=5.0, step=2.0):
    """그 관절을 혼자 돌릴 때 **바닥에 안 닿는 각도 구간**을 훑어 찾는다.

    시작점을 추측으로 정하지 않기 위한 도구. 가장 넓은 연속 구간을 돌려준다.
    """
    lo, hi = JOINT_LIMITS_DEG[joint - 1]
    lo += margin_deg; hi -= margin_deg
    good, cur, best = [], None, None
    a = lo
    while a <= hi:
        pose = list(home); pose[joint - 1] = a
        okh = lowest_height_mm(pose)[0] >= DESK_SAFETY_MARGIN_MM
        okc = not check_self_collision(full_q(pose))
        if okh and okc:
            cur = (a, a) if cur is None else (cur[0], a)
        else:
            if cur:
                good.append(cur); cur = None
        a += step
    if cur:
        good.append(cur)
    if good:
        best = max(good, key=lambda x: x[1] - x[0])
    return best, good


def centered_start(joint, travel, margin=5.0):
    """주행 구간이 관절 범위 가운데에 오도록 시작 각도를 정한다.

    한 방향 주행이라 시작점을 0 에 두면 한쪽으로만 270도를 가게 되어
    금방 한계를 넘는다. 가운데 정렬하면 같은 거리를 양쪽으로 나눠 쓴다
    (예: 270도 주행 -> -135 에서 시작해 +135 에서 끝).
    """
    lo, hi = JOINT_LIMITS_DEG[joint - 1]
    center = (lo + hi) / 2.0
    return center - travel / 2.0


def check_range(joint, q0, travel):
    """한 방향 주행이므로 q0 에서 q0+travel 까지가 전부 한계 안이어야 한다."""
    lo, hi = JOINT_LIMITS_DEG[joint - 1]
    a, b = min(q0, q0 + travel), max(q0, q0 + travel)
    margin = 5.0
    if a < lo + margin or b > hi - margin:
        span = (hi - margin) - (lo + margin)
        return False, (f"J{joint} 주행 {a:.0f}~{b:.0f}도가 한계 {lo}~{hi}도를 벗어납니다.\n"
                       f"    이 관절이 한 방향으로 갈 수 있는 최대 거리는 {span:.0f}도입니다"
                       f" (360도 연속 회전은 불가).")
    return True, ""


def wait_pose(mc, target, tol_deg=2.0, timeout=25.0):
    """지정 자세에 실제로 도달할 때까지 기다린다.

    [1차 실행 실패] 패스가 끝난 자리(+135도)에서 다음 패스 시작점(-135도)으로
    돌아가는 데 고정 2초만 기다렸다. 270도를 2초에 갈 수 없으니 다음 패스가
    엉뚱한 자리에서 시작했고, 패스 간 실측 자세 차이가 257도까지 났다.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        a = read_angles(mc)
        if a and max(abs(a[k] - target[k]) for k in range(6)) <= tol_deg:
            time.sleep(0.5)
            return True
        time.sleep(0.1)
    return False


def read_u(mc, source, load_joint):
    """u 로 쓸 값을 읽는다. {관절: 값}

    source="current": `get_servo_currents()` 로 **6축을 한 번(30ms)에** 읽는다.
      부하는 관절당 레지스터 2개 x 30ms = 60ms 라, 6축이면 360ms 가 되어
      루프가 130±73ms 로 불규칙해지고 로봇이 가다-서다를 반복했다
      (--no-load 는 R²=0.978, 부하 읽기는 0.456). 전류는 한 호출이라
      주기를 65ms 로 유지할 수 있고 **다중 패스가 필요 없다.**
      단점은 해상도다(정지 시 0/4/8/12 만 관측됨). 다만 그건 중력이 안
      걸리는 정지 상태 이야기이고, 가속 중에는 값이 커질 수 있다 -
      실행 후 출력되는 u 표준편차로 실제로 쓸 만한지 판단할 것.

    source="load": 지정한 한 관절의 Present Load(2바이트). 값 범위는 넓지만
      느리다. 다중 패스로 6축을 채워야 한다.
    """
    if source == "current":
        try:
            cur = mc.get_servo_currents()
            if isinstance(cur, list) and len(cur) >= 6:
                return {k + 1: float(cur[k]) for k in range(6)}
        except Exception:
            pass
        return {}
    if load_joint is None:
        return {}
    try:
        v = decode_load(mc.get_servo_data(load_joint, REG_LOAD_L),
                        mc.get_servo_data(load_joint, REG_LOAD_H))
        return {} if v is None else {load_joint: float(v)}
    except Exception:
        return {}


def read_angles(mc):
    a = fast_get_angles(mc)
    if a is None:
        a = mc.get_angles()
    return a if (isinstance(a, list) and len(a) >= 6) else None


def calibrate_speed(mc, joint, home, test_speeds=(20, 40, 60, 80, 100)):
    """speed 인자별로 **실제로 나오는 각속도**를 잰다.

    [왜 필요한가] speed 는 0~100 의 추상 값이라 몇 도/s 인지 문서로 알 수
    없다. 1차 실행에서 speed=60 이 약 13~15도/s 였던 것으로 보이는데,
    이걸 모르고 지령 최대 속도를 24도/s 로 잡아 사인이 상한에 잘렸다.
    **지령 최대 속도가 이 상한보다 충분히 낮아야** 사인이 살아난다.

    한 방향으로 60도를 보내고 걸린 시간으로 평균 속도를 구한다.
    """
    lo, hi = JOINT_LIMITS_DEG[joint - 1]
    a = max(lo + 10, -70.0)
    b = min(hi - 10, a + 60.0)
    print(f"\n[speed 보정] J{joint} 를 {a:.0f} -> {b:.0f}도 ({b-a:.0f}도) 이동시켜 측정")
    print(f"  {'speed':>7}{'소요(s)':>10}{'평균 속도(도/s)':>17}")
    out = {}
    for sp in test_speeds:
        p0 = list(home); p0[joint - 1] = a
        mc.send_angles(p0, 40)
        if not wait_pose(mc, p0):
            print(f"  {sp:>7}   시작 자세 복귀 실패 - 건너뜀")
            continue
        p1 = list(home); p1[joint - 1] = b
        t0 = time.perf_counter()
        mc.send_angles(p1, sp)
        ok = wait_pose(mc, p1, tol_deg=1.5, timeout=30.0)
        dt = time.perf_counter() - t0
        if not ok:
            print(f"  {sp:>7}   도달 실패({dt:.1f}s)")
            continue
        v = (b - a) / max(dt - 0.5, 1e-3)     # 도달 확인용 여유 0.5s 빼기
        out[sp] = v
        print(f"  {sp:>7}{dt:>10.2f}{v:>17.1f}")
    if out:
        print("\n  지령 최대 속도 = --speed x (1 + --ratio) 가 위 값보다")
        print("  충분히(30% 이상) 낮아야 사인이 상한에 잘리지 않습니다.")
    return out


def run_pass(mc, joint, q0_full, v_mean, ratio, period, cycles, load_joint,
             dt_target, source="load", direction=+1):
    """궤적 1회 실행. load_joint 의 부하만 읽는다(None 이면 안 읽음)."""
    rows = []
    t_end = period * cycles
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t > t_end:
            break
        q, qd, qdd = traj(t, q0_full[joint - 1], v_mean, ratio, period, direction)
        cmd = list(q0_full)
        cmd[joint - 1] = q
        fast_send_angles(mc, cmd, SEND_SPEED)

        meas = read_angles(mc)
        us = read_u(mc, source, load_joint) if source != "none" else {}
        rows.append({
            "t": round(t, 4), "cmd": cmd, "qd": qd, "qdd": qdd,
            "meas": meas, "us": us,
        })
        # 목표 주기까지 대기 (읽기가 이미 대부분을 소비한다)
        rest = dt_target - (time.perf_counter() - t0 - t)
        if rest > 0:
            time.sleep(rest)
    return rows


def merge_passes(passes, joint):
    """패스들을 시간축으로 합친다. 기준은 첫 패스."""
    base = passes[0]["rows"]
    ts = np.array([r["t"] for r in base])
    merged = []
    for i, r in enumerate(base):
        row = dict(r)
        row["loads"] = {}
        merged.append(row)
    for p in passes:
        pt = np.array([r["t"] for r in p["rows"]])
        joints = set()
        for r in p["rows"]:
            joints |= set(r.get("us", {}).keys())
        for j in sorted(joints):
            pv = np.array([r.get("us", {}).get(j, np.nan) for r in p["rows"]],
                          dtype=float)
            good = ~np.isnan(pv)
            if good.sum() < 2:
                continue
            # 시간축 보간 - 지령이 해석적이라 같은 t 는 같은 자세다
            vals = np.interp(ts, pt[good], pv[good])
            for i, v in enumerate(vals):
                merged[i]["loads"][j] = float(v)
    return merged


def check_repeatability(passes, joint):
    """패스 간 실측 자세가 같은지 본다. 다르면 시간축 합치기가 무효다."""
    if len(passes) < 2:
        return 0.0
    base = passes[0]["rows"]
    ts = np.array([r["t"] for r in base])
    bq = np.array([r["meas"][joint - 1] if r["meas"] else np.nan for r in base])
    worst = 0.0
    for p in passes[1:]:
        pt = np.array([r["t"] for r in p["rows"]])
        pq = np.array([r["meas"][joint - 1] if r["meas"] else np.nan
                       for r in p["rows"]])
        g = ~np.isnan(pq)
        if g.sum() < 2:
            continue
        iq = np.interp(ts, pt[g], pq[g])
        d = np.nanmax(np.abs(bq - iq))
        worst = max(worst, float(d))
    return worst


def repeatability_report(reps, joint):
    """회차 간 u 가 얼마나 재현되는지, 방향에 따라 얼마나 다른지 본다.

    [왜 필요한가]
    - **잡음 크기를 알아야** residual 로 충돌을 판정할 문턱값을 정할 수 있다.
      한 번만 재면 지금 u 값이 로봇의 성질인지 그 회차의 우연인지 모른다.
    - **방향 차이는 마찰이다.** 정방향 평균과 역방향 평균의 차이가 쿨롱
      마찰의 2배다(§3.9 에서 정적 측정을 양방향 평균한 것과 같은 이유).
      단방향으로만 돌면 그 오프셋이 파라미터에 그대로 섞여 들어간다.

    u 를 **시간이 아니라 각도(q)의 함수로** 비교한다. 역방향 회차는 같은
    각도를 반대 순서로 지나므로 시간축으로는 못 맞춘다.
    """
    fwd = [r for r in reps if r["dir"] > 0]
    rev = [r for r in reps if r["dir"] < 0]
    if not fwd and not rev:
        return

    lo = max(min(x["q"][j] for j in range(len(x["q"]))) for x in reps)
    hi = min(max(x["q"][j] for j in range(len(x["q"]))) for x in reps)
    if not (hi > lo):
        print("\n  [재현성] 회차들이 공통으로 지나간 각도 구간이 없습니다.")
        return
    grid = np.linspace(lo, hi, 60)

    def on_grid(run, j):
        q = np.asarray(run["q"], dtype=float)
        u = np.asarray([run["u"].get(j, [np.nan] * len(q))][0], dtype=float)
        order = np.argsort(q)
        qs, us = q[order], u[order]
        good = ~np.isnan(us)
        if good.sum() < 5:
            return None
        return np.interp(grid, qs[good], us[good])

    print(f"\n[재현성]  같은 각도에서 회차끼리 얼마나 같은가 "
          f"(정방향 {len(fwd)}회, 역방향 {len(rev)}회)")
    print(f"  {'관절':>5}{'정방향 평균':>12}{'회차간 표준편차':>16}"
          f"{'역방향 평균':>12}{'방향 차이':>11}")
    for j in range(1, 7):
        fc = [x for x in (on_grid(r, j) for r in fwd) if x is not None]
        rc = [x for x in (on_grid(r, j) for r in rev) if x is not None]
        if not fc and not rc:
            continue
        fm = np.nanmean(fc, axis=0) if fc else None
        rm = np.nanmean(rc, axis=0) if rc else None
        sd = float(np.nanmean(np.nanstd(fc, axis=0))) if len(fc) > 1 else float("nan")
        fmean = float(np.nanmean(fm)) if fm is not None else float("nan")
        rmean = float(np.nanmean(rm)) if rm is not None else float("nan")
        diff = (fmean - rmean) if (fm is not None and rm is not None) else float("nan")
        def f(v, w=12, p=2):
            return f"{v:>{w}.{p}f}" if v == v else f"{'--':>{w}}"
        print(f"  {('u%d' % j):>5}{f(fmean)}{f(sd,16)}{f(rmean)}{f(diff,11)}")
    print("\n  회차간 표준편차 = 잡음 크기. residual 문턱값을 이보다 크게 잡아야 한다.")
    if fwd and rev:
        print("  방향 차이 = 쿨롱 마찰의 2배. 양방향을 함께 쓰면 상쇄된다.")
    else:
        print("  ★ 한 방향만 돌았습니다. 마찰 오프셋이 파라미터에 섞입니다"
              " (--reps 2 이상 + 교대).")


def write_csv(merged, joint, path, mode="w"):
    """계획 5단계 형식 그대로: time | q1..q6 | qd1..qd6 | qdd1..qdd6 | u1..u6"""
    fields = (["time"]
              + [f"q{k+1}" for k in range(6)]
              + [f"qd{k+1}" for k in range(6)]
              + [f"qdd{k+1}" for k in range(6)]
              + [f"u{k+1}" for k in range(6)]
              + [f"q{k+1}_cmd" for k in range(6)]
              + ["track_err", "dt", "dq", "rep", "dir"])
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            w.writeheader()
        for r in merged:
            row = {"time": r["t"]}
            meas = r["meas"]
            for k in range(6):
                row[f"q{k+1}"] = round(meas[k], 4) if meas else ""
                row[f"q{k+1}_cmd"] = round(r["cmd"][k], 4)
                # 움직이는 관절 외에는 지령상 정지이므로 0
                row[f"qd{k+1}"] = round(r["qd"], 5) if (k + 1) == joint else 0.0
                row[f"qdd{k+1}"] = round(r["qdd"], 5) if (k + 1) == joint else 0.0
                v = r["loads"].get(k + 1)
                row[f"u{k+1}"] = round(v, 2) if v is not None else ""
            row["track_err"] = (round(max(abs(meas[k] - r["cmd"][k]) for k in range(6)), 3)
                                if meas else "")
            row["dt"] = r.get("dt", "")
            row["dq"] = r.get("dq", "")
            row["rep"] = r.get("rep", 0)
            row["dir"] = r.get("dir", 1)
            w.writerow(row)
    return fields


def main():
    ap = argparse.ArgumentParser(description="T1~T3 궤적 데이터 수집")
    ap.add_argument("--joint", type=int, default=1, help="움직일 관절 (T1=1, T2=2, T3=3)")
    ap.add_argument("--speed", type=float, default=15.0, help="평균 각속도(도/s)")
    ap.add_argument("--ratio", type=float, default=0.6,
                    help="속도 변동폭 / 평균. 1 미만이어야 속도가 항상 양수")
    ap.add_argument("--period", type=float, default=6.0, help="사인파 주기(초)")
    ap.add_argument("--cycles", type=float, default=3.0, help="반복 횟수")
    ap.add_argument("--home", type=str, default="0,0,0,0,0,0",
                    help="나머지 관절의 고정 자세")
    ap.add_argument("--start", type=float, default=None,
                    help="움직일 관절의 시작 각도. 생략하면 주행 구간을 관절 범위 "
                         "가운데에 자동 배치한다")
    ap.add_argument("--reps", type=int, default=4,
                    help="같은 궤적을 몇 회 반복할지. 홀수 회차는 반대 방향으로 돈다. "
                         "회차간 흩어짐이 잡음 크기이고, 방향 차이가 마찰이다")
    ap.add_argument("--one-way", action="store_true",
                    help="반대 방향 회차를 넣지 않는다(마찰 오프셋이 남으므로 권장하지 않음)")
    ap.add_argument("--multipass", action="store_true",
                    help="같은 궤적을 6번 돌려 u1~u6 를 전부 채운다")
    ap.add_argument("--dry", action="store_true", help="궤적만 계산해 보여준다(로봇 불필요)")
    ap.add_argument("--safe-range", action="store_true",
                    help="그 관절이 바닥에 안 닿는 각도 구간을 찾아준다(로봇 불필요)")
    ap.add_argument("--source", choices=("current", "load", "none"), default="current",
                    help="u 를 무엇으로 읽을지. current=6축 한 번에(빠름, 해상도 낮음), "
                         "load=관절당 2바이트(느림, 다중 패스 필요), none=안 읽음")
    ap.add_argument("--send-speed", type=int, default=None,
                    help="send_angles 의 speed. 생략하면 지령 최대 속도에 맞춰 자동. "
                         "너무 크면 로봇이 먼저 도착해 멈춰 서므로 가다-서다가 된다")
    ap.add_argument("--no-load", action="store_true",
                    help="부하를 읽지 않고 각도만 기록한다. 루프가 30ms 로 빨라지므로 "
                         "'부하 읽기가 루프를 망치는가'를 분리해 확인할 수 있다")
    ap.add_argument("--calibrate", action="store_true",
                    help="speed 인자별 실제 각속도를 먼저 잰다(로봇 움직임)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    home = [float(x) for x in args.home.split(",")]
    assert len(home) == 6

    if args.safe_range:
        best, good = safe_range(args.joint, home)
        print(f"[안전 구간] J{args.joint} (나머지 {home}, 책상 여유 {DESK_SAFETY_MARGIN_MM}mm)")
        print("  ※ 링크 원점 높이 기준입니다. 링크의 부피는 안 봅니다 -")
        print("    처음 도는 궤적은 눈으로 지켜보세요.")
        if not good:
            print("  안전한 구간이 없습니다. 나머지 관절 자세(--home)를 바꿔보세요.")
            return
        for a, b in good:
            mark = "  <- 가장 넓음" if (a, b) == best else ""
            print(f"  {a:+7.0f} ~ {b:+7.0f}도  ({b-a:.0f}도){mark}")
        print(f"\n  예시: --start {best[0]:.0f} 로 시작해 {best[1]-best[0]:.0f}도 이내로 주행")
        print(f"        주행거리 = speed x period x cycles 이므로,")
        print(f"        speed 8, period 4 이면 --cycles {(best[1]-best[0])/32:.1f} 이하")
        return
    if not (0.0 < args.ratio < 1.0):
        print("--ratio 는 0과 1 사이여야 합니다 (1 이상이면 속도가 음수가 되어"
              " 방향이 뒤집힙니다).")
        return
    travel = total_travel(args.speed, args.period, args.cycles)
    if args.start is None:
        # 시작점을 안 주면 주행 구간을 관절 범위 가운데 놓는다.
        home[args.joint - 1] = centered_start(args.joint, travel)
        auto = True
    else:
        home[args.joint - 1] = args.start
        auto = False
    ok, msg = check_range(args.joint, home[args.joint - 1], travel)
    if not ok:
        print(msg)
        lo, hi = JOINT_LIMITS_DEG[args.joint - 1]
        span = (hi - 5.0) - (lo + 5.0)
        max_cycles = span / (args.speed * args.period)
        print(f"\n  줄이는 방법 (셋 중 하나):")
        print(f"    --cycles {max_cycles:.1f} 이하")
        print(f"    --speed {span/(args.period*args.cycles):.1f} 이하")
        print(f"    --period {span/(args.speed*args.cycles):.1f} 이하")
        return
    n_est = int(args.period * args.cycles / 0.065)
    w = 2 * math.pi / args.period
    v1 = args.ratio * args.speed
    print(f"T{args.joint}: J{args.joint} 만 **한 방향**으로 돌며 속도가 사인으로 변함")
    print(f"  평균 속도 {args.speed} 도/s, 변동 ±{v1:.1f} 도/s"
          f"  -> 속도 범위 {args.speed-v1:.1f} ~ {args.speed+v1:.1f} 도/s (항상 양수)")
    q0j = home[args.joint - 1]
    print(f"  사인 주기 {args.period}초 x {args.cycles}회 = 총 {travel:.0f}도 이동")
    print(f"  J{args.joint}: {q0j:.0f}도 -> {q0j+travel:.0f}도"
          + ("  (시작점 자동 배치)" if auto else ""))
    print(f"  최대 각가속도 {v1*w:.1f} 도/s²")
    # send speed: 너무 크면 로봇이 먼저 도착해 멈춰 선다(가다-서다).
    # 지령 최대 속도보다 조금만 높게 잡아 계속 움직이게 한다.
    global SEND_SPEED
    if args.send_speed is not None:
        SEND_SPEED = args.send_speed
    else:
        peak = args.speed * (1 + args.ratio)
        # --calibrate 실측: speed 20 -> 25.5 도/s (가감속 포함 평균, 정상속도는 더 높음)
        # 상한이 지령보다 훨씬 높으면 로봇이 먼저 도착해 멈춰 선다.
        # 조금만 높게 잡아 계속 움직이게 한다. 가다-서다가 남으면
        # --send-speed 를 더 낮춰볼 것.
        SEND_SPEED = int(max(10, min(100, round(peak * 1.2))))
    # 실행 전에 궤적 전체를 훑어 책상/자가충돌을 본다.
    safe, why, worst_h, worst_a, worst_w = scan_trajectory(
        args.joint, home, q0j, args.speed, args.ratio, args.period, args.cycles)
    _part = "TCP" if worst_w == 0 else f"J{worst_w} 부근"
    print(f"  최저 높이 {worst_h:.0f}mm - {_part}, J{args.joint}={worst_a:.0f}도 일 때")
    if not safe:
        print(f"\n  {why}")
        best, _ = safe_range(args.joint, home)
        if best:
            print(f"\n  이 관절의 안전 구간: {best[0]:+.0f} ~ {best[1]:+.0f}도"
                  f" ({best[1]-best[0]:.0f}도)")
            print(f"  --start {best[0]:.0f} --cycles "
                  f"{(best[1]-best[0])/(args.speed*args.period):.1f} 이하로 다시 시도하세요.")
        print("  (--safe-range 로 전체 구간을 볼 수 있습니다)")
        return

    print(f"  send_angles speed = {SEND_SPEED}"
          + ("" if args.send_speed is not None else " (지령 최대 속도 기준 자동)"))

    if args.dry:
        print(f"\n  {'t':>6}{'q':>10}{'qd':>10}{'qdd':>10}")
        for i in range(0, 13):
            t = args.period * i / 12
            q, qd, qdd = traj(t, home[args.joint - 1], args.speed, args.ratio, args.period)
            print(f"  {t:>6.2f}{q:>10.2f}{qd:>10.2f}{qdd:>10.2f}")
        print("\n  (--dry 이므로 로봇은 움직이지 않았습니다)")
        return

    print("\n  ⚠️ 로봇이 실제로 움직입니다. 주변을 비우고 비상정지에 손을 두세요.")
    if input("  진행하려면 yes 입력: ").strip().lower() != "yes":
        return

    mc, _ = find_robot_port()
    if mc is None:
        print("로봇을 찾지 못했습니다.")
        return

    # 시작 자세로 이동 후 안정화
    start = list(home)
    start[args.joint - 1] = traj(0.0, home[args.joint - 1], args.speed, args.ratio, args.period)[0]
    print("  시작 자세로 이동 중...")
    mc.send_angles(start, 40)
    if not wait_pose(mc, start):
        print("  ★ 시작 자세에 도달하지 못했습니다. 중단합니다.")
        return

    if args.calibrate:
        cal = calibrate_speed(mc, args.joint, home)
        peak = args.speed * (1 + args.ratio)
        if cal:
            cap = cal.get(SEND_SPEED, max(cal.values()))
            print(f"\n  현재 설정의 지령 최대 속도: {peak:.1f} 도/s")
            print(f"  speed={SEND_SPEED} 의 실측 상한: {cap:.1f} 도/s")
            if peak > cap * 0.7:
                print(f"  ★ 너무 가깝습니다. --speed {cap*0.7/(1+args.ratio):.0f} 이하를 권합니다.")
            else:
                print(f"  여유 있습니다.")
        return

    source = "none" if args.no_load else args.source
    if source == "none":
        load_joints = [None]; dt_target = 0.035
        print("\n  [u 안 읽음] 각도만 기록합니다 (목표 주기 35ms).")
    elif source == "current":
        load_joints = [None]; dt_target = 0.065
        print("\n  [u=전류] get_servo_currents() 로 6축을 한 번에 읽습니다"
              " (목표 주기 65ms, 다중 패스 불필요).")
    else:
        load_joints = [1, 2, 3, 4, 5, 6] if args.multipass else [args.joint]
        dt_target = 0.09
        print("\n  [u=부하] 관절당 2바이트를 읽습니다 (느림). 6축을 채우려면"
              " --multipass 가 필요합니다.")
    out = args.out or f"T{args.joint}_dataset.csv"
    q0j = home[args.joint - 1]
    travel_j = total_travel(args.speed, args.period, args.cycles)
    reps_info, all_ts, all_qm, first_lag = [], None, None, None
    n_rows = 0

    for rep in range(args.reps):
        direction = +1 if (args.one_way or rep % 2 == 0) else -1
        # 역방향은 반대쪽 끝에서 시작해 되돌아온다.
        rep_start = list(home)
        rep_start[args.joint - 1] = q0j if direction > 0 else q0j + travel_j
        label = "정방향" if direction > 0 else "역방향"
        print(f"\n=== 회차 {rep+1}/{args.reps} ({label}) ===")

        passes = []
        for k, lj in enumerate(load_joints):
            if len(load_joints) > 1:
                print(f"  패스 {k+1}/{len(load_joints)} - J{lj}")
            mc.send_angles(rep_start, 40)
            if not wait_pose(mc, rep_start):
                print("    ★ 시작 자세 복귀 실패 - 건너뜁니다.")
                continue
            rows = run_pass(mc, args.joint, rep_start, args.speed, args.ratio,
                            args.period, args.cycles, lj, dt_target, source,
                            direction)
            passes.append({"load_joint": lj, "rows": rows})
            dts = np.diff([r["t"] for r in rows])
            print(f"    {len(rows)}샘플, 주기 {dts.mean()*1000:.0f}±{dts.std()*1000:.0f}ms")
        if not passes:
            continue

        merged = merge_passes(passes, args.joint)
        for i2, r in enumerate(merged):
            if i2 == 0 or not r["meas"] or not merged[i2-1]["meas"]:
                r["dt"] = ""; r["dq"] = ""
            else:
                r["dt"] = round(r["t"] - merged[i2-1]["t"], 4)
                r["dq"] = round(r["meas"][args.joint-1]
                                - merged[i2-1]["meas"][args.joint-1], 4)
            r["rep"] = rep + 1
            r["dir"] = direction

        ts = [r["t"] for r in merged]
        qmj = [r["meas"][args.joint - 1] if r["meas"] else float("nan") for r in merged]
        qs0 = rep_start[args.joint - 1]
        lag, lag_err = estimate_lag(
            ts, qmj,
            lambda t: traj(max(t, 0.0), qs0, args.speed, args.ratio,
                           args.period, direction)[0])
        for r in merged:
            _, qd, qdd = traj(max(r["t"] - lag, 0.0), qs0, args.speed,
                              args.ratio, args.period, direction)
            r["qd"], r["qdd"] = qd, qdd
        errs = [max(abs(r["meas"][k] - r["cmd"][k]) for k in range(6))
                for r in merged if r["meas"]]
        print(f"    지연 {lag*1000:.0f}ms, 보정 후 추종오차 {lag_err:.2f}도"
              + (f", 최대 {max(errs):.2f}도" if errs else ""))

        write_csv(merged, args.joint, out, mode=("w" if n_rows == 0 else "a"))
        n_rows += len(merged)

        # 재현성 분석용 - u 를 각도의 함수로 모아둔다
        uu = {}
        for jj in range(1, 7):
            vals = [r["loads"].get(jj, float("nan")) for r in merged]
            if any(v == v for v in vals):
                uu[jj] = vals
        reps_info.append({"dir": direction, "q": qmj, "u": uu})
        if rep == 0:
            all_ts, all_qm, first_lag = ts, qmj, lag

    if n_rows == 0:
        print("\n수집된 데이터가 없습니다.")
        return

    print(f"\n저장: {out} ({n_rows}행, {len(reps_info)}회차)")
    print("  열: time, q1~q6(실측), qd1~qd6, qdd1~qdd6(지연보정 해석해),")
    print("      u1~u6, q1_cmd~q6_cmd(지령), track_err, dt, dq, rep, dir")

    # u 가 실제로 정보를 담고 있는지 - 전류는 해상도가 낮아 상수가 될 수 있다.
    print("\n[u 분포]  (표준편차가 0에 가까우면 그 관절은 정보가 없다)")
    for jj in range(1, 7):
        vals = [v for r in reps_info for v in r["u"].get(jj, []) if v == v]
        if not vals:
            continue
        v = np.array(vals, dtype=float)
        print(f"  u{jj}: 범위 {v.min():+7.1f}~{v.max():+7.1f}"
              f"  표준편차 {v.std():6.2f}  서로 다른 값 {len(np.unique(np.round(v,3)))}개")

    repeatability_report(reps_info, args.joint)

    ts, qmj, lag = all_ts, all_qm, first_lag
    verify_sine(ts, qmj, args.speed, args.ratio, args.period, lag, args.joint,
                png=f"T{args.joint}_speed.png")


if __name__ == "__main__":
    main()
