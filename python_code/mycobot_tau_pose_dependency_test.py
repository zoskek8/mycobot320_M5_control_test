# -*- coding: utf-8 -*-
"""
mycobot_tau_pose_dependency_test.py
=====================================
[9차 세션, §27 조사] 관절별 추종지연(tau)이 **그 관절이 놓인 자세에 따라
달라지는가**를 단독 스윕으로 측정한다.

[왜 이 조사를 하는가]
매 실행마다 나오는 joint_lag.txt의 tau가 곡선마다 크게 달랐다:

    curve_stop 계열 : J1 195.7 / J2 169.7 / J3 168.5  (J1이 최대)
    급커브 곡선     : J1 200.5 / J2 208.3 / J3 211.5  (J3이 최대 - 역전)
    급커브 곡선     : J1 181.9 / J2 181.4 / J3 181.4  (셋이 동률)

특히 결정적인 관측: 급커브 곡선의 한 실행에서 **J2가 각속도 19.2도/s로
전체 관절 중 가장 느린데 tau는 208.3ms로 2위**였다. 순수 속도비례
모델이면 나올 수 없는 조합이다.

[왜 지금 방식으로는 못 잡나 - §14와 같은 문제]
joint_lag.txt는 곡선 전체를 하나로 뭉쳐 관절당 회귀 하나를 뽑는다. 그
곡선이 J2를 -110도부터 +40도까지 훑었다면 그 전 구간의 서로 다른 특성이
하나의 평균으로 뭉개진다. 곡선이 바뀌면 방문 영역이 바뀌니 평균도 바뀐다.

이건 §14에서 이미 겪은 문제다 - 정지오차 보정계수를 10개 자세 혼합
데이터로 회귀했을 때 R²=0.61이었는데, J2만 단독 스윕하니 R²=0.815로
올라가고 기울기도 2배 이상 달랐다. "여러 관절이 동시에 바뀌는 데이터로
회귀하면 계수가 오염된다"는 그 교훈을 정지오차엔 적용했지만 tau에는
아직 적용하지 않았다.

[측정 방법]
mycobot_static_error_test.py의 J2/J6 스윕과 같은 구조 - 다른 관절을
고정하고 대상 관절만 여러 자세로 옮겨가며, 각 자세에서 **작은 스텝을
주고 그 응답 시계열을 1차 지연 모델로 피팅**해 그 자세에서의 tau를 뽑는다.

    y(t) = y0 + (y1-y0) * (1 - exp(-(t - t_d)/tau))     (t >= t_d)

tau(시정수)와 t_d(순수지연)를 함께 추정한다. joint_lag.txt의 "유효지연"은
이 둘이 섞인 값이라 직접 비교하면 안 되고, 여기서는 tau 자체의 자세
의존성만 본다.

[파라미터 근거 - 합성 데이터로 사전 검증함]
  - STEP_DEG = 15: 스텝 5도로는 추정오차가 9~10%였다(노이즈 0.3도 가정).
                   15도면 2% 이내로 떨어진다.
  - SAMPLE_SEC = 1.2: 관측창 0.5s에서 6.8%, 1.0s에서 2.2%, 1.5s에서 1.8%.
                   자세당 반복 횟수를 감안해 1.2초로 잡았다.
  - REPEATS = 4: §15에서 반복 2회로는 노이즈와 실제 패턴을 구분 못 했던
                 전례가 있어 4회로 잡는다.

사용법:
    python3 mycobot_tau_pose_dependency_test.py

    [1] J2 자세별 tau 스윕 (권장 - 중력부하 가장 큼)
    [2] J3 자세별 tau 스윕
    [3] 둘 다 연속 실행
    [4] J2 방향효과 직접 확인 (같은 자세에서 +/- 둘 다 측정)
    [5] J3 방향효과 직접 확인
    [6] [최우선] J3 자세0도 `-`방향 이봉분포 재현확인 (REPEATS=10, §27.4-1)
    [7] [10차 세션 신규, §34] J4 자세별 tau 스윕 - joint_lag.txt에서 계속
        평균/최대오차가 가장 큰 관절인데 이번 조사 범위 밖이었다. fixed
        자세는 J2/J3 패턴을 재사용했으나 실기 충돌검증은 안 됐다 - 처음
        실행 전 그 자세로 한 번 손으로 이동시켜 눈으로 확인할 것.
    [8] [10차 세션 신규] J4 방향효과 직접 확인
    [q] 종료

[안전] 스텝은 15도로 작다. 방향은 스윕 전체에서 **고정**이다(기본 +) -
이번 스윕 범위(SWEEP_CONFIG)는 고정방향으로도 전부 관절한계 안쪽에
든다는 걸 사전에 확인했다. 혹시 특정 자세에서만 한계를 넘으면 그
자세만 자동으로 반대방향으로 처리하고 콘솔에 경고를 남긴다.

[검증 중 발견한 혼란변수 - 중요] 원래는 "한계에서 멀어지는 쪽"으로
자세마다 방향을 자동 선택했었다. 그런데 실측해보니 J3에서 tau/t_d가
같이 튀는 지점이 정확히 그 자동선택이 방향을 뒤집는 지점과 겹쳤다 -
§15의 방향이력(backlash)이 다시 나타난 것으로 의심됐다. 그래서 스윕
전체를 고정방향으로 바꿨고, [4]/[5]로 방향 자체의 효과를 자세와
분리해서 따로 정량화할 수 있게 했다.
"""

import sys
import time

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from mycobot_kinematics import (
    find_robot_port, JOINT_LIMITS_DEG, SPEED,
    SETTLE_DELAY_SEC, MOVE_TIMEOUT_SEC,
)

SETTLE_EXTRA_SEC = 1.0     # 이동 완료 후 서보가 완전히 자리 잡을 때까지
STEP_DEG = 15.0            # 스텝 크기 (위 파라미터 근거 참고)
SAMPLE_SEC = 1.2           # 스텝 후 응답을 관측할 시간
REPEATS = 4                # 자세당 반복 (노이즈와 실제 패턴 분리)
MIN_R2 = 0.90              # 이 미만이면 피팅 실패로 간주하고 버림

# [9차 세션, §27.4-1] J3 자세0도 `-`방향에서 나온 반반 갈림(92ms대/108ms대)이
# 진짜 이봉분포인지 확인하려면 REPEATS=4로는 부족하다 - 8~10회 중 여유
# 있게 10으로 잡는다(통계적 힘 확보, 소요시간은 자세 1곳 x 10회라
# 전체 스윕(6자세)보다 훨씬 짧다).
BIMODAL_CHECK_REPEATS = 10

# [9차 세션, §27 후속 - 세 번째 혼란변수 수정] goto_and_settle(base)이 접근
# 방향을 통제하지 않아서, 반복1과 반복2~4가 서로 다른 방향에서 베이스에
# 도달하는 문제가 실측으로 확인됐다(반복2~4는 항상 직전 반복이 끝난
# 자리=스텝 반대쪽에서 돌아오는데, 반복1만 직전 블록에서 넘어온다).
# §15의 방향이력 실험이 이미 이 문제를 "경유점(via pose)"으로 풀었다 -
# 목표 전에 항상 같은 경유점을 거치게 해서 접근방향을 고정한다.
# VIA_MARGIN_DEG는 §15의 HYSTERESIS_MULTIPOSE_VIA_DELTA_DEG(J2/J3=25도)를
# 그대로 재사용한다 - 이미 검증된 값이라 새로 최적화하지 않는다.
VIA_MARGIN_DEG = 25.0
APPROACH_FROM = +1.0       # 모든 측정을 이 방향(항상 '위'쪽)에서 접근하도록 고정.
                           # 스텝 방향과 무관하게 고정해야 접근방향 효과와
                           # 스텝방향 효과를 분리해서 볼 수 있다.

# 대상 관절별: (고정할 나머지 관절 값, 스윕할 자세 목록)
# 나머지 관절은 §17의 J2 스윕과 같은 값을 쓴다 - 이미 검증된 조합이라
# 새로 최적화하지 않는다(데이터가 적을 때 기준까지 새로 만들면 과적합 위험).
SWEEP_CONFIG = {
    2: {  # J2
        "fixed": {1: 0.0, 3: 30.0, 4: 0.0, 5: 30.0, 6: 0.0},
        "poses": [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0],
    },
    3: {  # J3
        "fixed": {1: 0.0, 2: -30.0, 4: 0.0, 5: 30.0, 6: 0.0},
        "poses": [-60.0, -30.0, 0.0, 30.0, 60.0, 90.0],
    },
    # [10차 세션, §34 신규] J4 확장 - joint_lag.txt에서 6축 중 계속 평균/최대
    # 오차가 가장 크게 나오는 관절인데도 이번 조사(§27) 범위 밖이었다.
    # J2/J3의 fixed 패턴(스윕 대상이 아닌 나머지를 0에서 30도씩 벌려 특이자세를
    # 피함)을 그대로 재사용했다 - **다만 이 조합 자체가 실기로 충돌 확인된 적은
    # 없다**(J2/J3는 §17에서 이미 검증됐지만 J4는 이번이 처음). 전체 스윕(6자세
    # ×4반복) 전에, 이 fixed 자세로 J4만 손으로 한 번 이동시켜 육안으로 충돌
    # 여지가 없는지 먼저 확인할 것을 권장한다.
    4: {  # J4
        "fixed": {1: 0.0, 2: -30.0, 3: 30.0, 5: 30.0, 6: 0.0},
        "poses": [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0],
    },
}


def step_response(t, y0, y1, tau, t_d):
    """1차 지연 스텝응답. t_d 이전엔 y0 유지, 이후 지수적으로 y1에 접근."""
    out = np.full_like(t, y0, dtype=float)
    m = t >= t_d
    out[m] = y0 + (y1 - y0) * (1.0 - np.exp(-(t[m] - t_d) / max(tau, 1e-6)))
    return out


def estimate_tau(times, angles):
    """스텝응답 시계열에서 (tau초, t_d초, R2)를 추정. 실패하면 (None,None,None).

    [검증] 합성 데이터로 참값 복원을 확인했다 - 참 tau 80~350ms 범위에서
    추정오차 2% 이내, R²>0.998(노이즈 0.1도 가정).
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(angles, dtype=float)
    if len(t) < 10:
        return None, None, None

    y0g = float(np.mean(y[:max(2, len(y) // 20)]))
    y1g = float(np.mean(y[-max(2, len(y) // 10):]))
    if abs(y1g - y0g) < 0.5:
        # 스텝이 실제로 안 실렸다 - 명령이 씹혔거나 관절이 안 움직였다
        return None, None, None

    # 63.2% 도달 시점으로 tau 초기추정 (1차 지연의 정의)
    target = y0g + 0.632 * (y1g - y0g)
    idx = int(np.argmin(np.abs(y - target)))
    tau_g = max(0.01, float(t[idx]) * 0.6)

    try:
        popt, _ = curve_fit(
            step_response, t, y,
            p0=[y0g, y1g, tau_g, 0.05],
            bounds=([y0g - 5, y1g - 5, 0.005, 0.0],
                    [y0g + 5, y1g + 5, 2.0, 0.5]),
            maxfev=20000,
        )
    except Exception:
        return None, None, None

    y_fit = step_response(t, *popt)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return float(popt[2]), float(popt[3]), float(r2)


def goto_and_settle(mc, angles):
    """mycobot_static_error_test.py와 같은 방식 - 보내고 멈출 때까지 기다린다."""
    mc.send_angles(list(angles), SPEED)
    time.sleep(SETTLE_DELAY_SEC)
    t0 = time.time()
    while mc.is_moving():
        if time.time() - t0 > MOVE_TIMEOUT_SEC:
            break
        time.sleep(0.1)
    time.sleep(SETTLE_EXTRA_SEC)


def _clip(joint_idx, value):
    """관절한계 안쪽으로 클램프 (여유 5도)."""
    lo, hi = JOINT_LIMITS_DEG[joint_idx - 1]
    return float(min(hi - 5.0, max(lo + 5.0, value)))


def _via_pose(joint_idx, base_full):
    """base_full로 가기 전에 거칠 경유점 - joint_idx만 APPROACH_FROM 방향으로
    VIA_MARGIN_DEG만큼 떨어뜨린다. 이걸 거쳐서 base로 가면 항상 같은
    방향에서 base에 도달한다(§15의 경유점 방식과 동일)."""
    via = list(base_full)
    via[joint_idx - 1] = _clip(joint_idx, base_full[joint_idx - 1] + APPROACH_FROM * VIA_MARGIN_DEG)
    return via


def _step_direction(joint_idx, base_deg, fixed_direction):
    """[검증 중 발견한 혼란변수 - 수정] 원래는 '한계에서 멀어지는 쪽'으로
    자동 선택했다. 그런데 실측 결과 J3에서 tau/t_d가 동시에 튀는 지점이
    정확히 이 자동선택이 방향을 뒤집는 지점(0도->30도)과 일치했다 -
    §15에서 확인된 방향이력(backlash)이 스텝응답의 t_d를 밀어올리는
    형태로 다시 나타난 것으로 의심된다(백래시는 유격을 없앨 때까지
    반응이 늦어지므로 t_d를 올리는 쪽이 물리적으로 자연스럽다).

    이번에 쓰는 스윕 범위(SWEEP_CONFIG)는 고정된 한쪽 방향으로만 스텝해도
    전부 관절한계 안쪽에 든다(사전 확인함) - 애초에 방향을 적응시킬
    필요가 없었다. 그래서 **스윕 전체에 고정 방향**을 쓰도록 바꾼다 -
    이러면 자세 간 비교에 방향이 안 섞여 들어간다.

    혹시 다른 스윕 범위를 추가해 한계에 닿을 상황이면, 방향을 바꾸는
    대신 그 자세를 건너뛰는 쪽을 권한다(측정_tau_at_pose의 스텝잘림
    감지 로직이 이미 그렇게 처리한다) - 방향을 바꾸느니 그 지점을
    비우는 게 낫다."""
    lo, hi = JOINT_LIMITS_DEG[joint_idx - 1]
    target = base_deg + fixed_direction * STEP_DEG
    if lo + 1.0 <= target <= hi - 1.0:
        return fixed_direction
    # 고정방향이 이 자세에서만 한계를 넘으면 그때만 반대로 - 흔치 않은 경우라
    # 콘솔에 경고를 남겨 결과 해석 시 그 자세를 특별 취급하게 한다.
    print(f"    ⚠️ 자세 {base_deg:+.0f}도에서는 고정방향({'+' if fixed_direction>0 else '-'})이 "
          f"한계를 넘어 이 자세만 반대방향으로 스텝합니다 - 결과 비교 시 감안할 것.")
    return -fixed_direction


def measure_tau_at_pose(mc, joint_idx, base_pose_deg, fixed, sweep_direction, repeats=REPEATS):
    """한 자세에서 스텝응답을 repeats회 재고 tau 목록을 반환 (기본 REPEATS회 -
    run_sweep/run_direction_check는 이 기본값을 그대로 쓰고, §27.4의
    이봉분포 재현확인만 더 많은 반복수를 넘긴다).
    반환: (tau리스트[ms], t_d리스트[ms], 샘플시계열 하나(그래프용))"""
    base = [0.0] * 6
    for j, v in fixed.items():
        base[j - 1] = _clip(j, v)
    base[joint_idx - 1] = _clip(joint_idx, base_pose_deg)

    direction = _step_direction(joint_idx, base[joint_idx - 1], sweep_direction)
    stepped = list(base)
    stepped[joint_idx - 1] = _clip(joint_idx, base[joint_idx - 1] + direction * STEP_DEG)

    actual_step = stepped[joint_idx - 1] - base[joint_idx - 1]
    if abs(actual_step) < STEP_DEG * 0.5:
        print(f"    ⚠️ 스텝이 한계에 잘려 {actual_step:+.1f}도밖에 안 됩니다 - 이 자세는 건너뜁니다.")
        return [], [], None

    taus, tds = [], []
    sample_trace = None

    for rep in range(repeats):
        # [수정] 직접 base로 가지 않고 경유점을 먼저 거친다 - 매 반복이
        # 항상 APPROACH_FROM 방향에서 base에 도달하도록 강제한다.
        goto_and_settle(mc, _via_pose(joint_idx, base))
        goto_and_settle(mc, base)

        # 스텝 명령 직전부터 시계열 수집 시작
        times, vals = [], []
        t0 = time.time()
        mc.send_angles(list(stepped), SPEED)
        while time.time() - t0 < SAMPLE_SEC:
            a = mc.get_angles()
            if isinstance(a, list) and len(a) == 6:
                times.append(time.time() - t0)
                vals.append(a[joint_idx - 1])

        tau, t_d, r2 = estimate_tau(times, vals)
        if tau is None or r2 < MIN_R2:
            print(f"    반복{rep+1}: 피팅 실패 또는 R²={r2 if r2 is not None else float('nan'):.3f} < {MIN_R2} - 버림")
            continue
        taus.append(tau * 1000.0)
        tds.append(t_d * 1000.0)
        print(f"    반복{rep+1}: tau={tau*1000:6.1f}ms  t_d={t_d*1000:5.1f}ms  R²={r2:.4f}")
        if sample_trace is None:
            sample_trace = (np.array(times), np.array(vals), tau, t_d,
                            float(np.mean(vals[:3])), float(np.mean(vals[-3:])))

    return taus, tds, sample_trace


def run_sweep(mc, joint_idx, sweep_direction=+1.0):
    cfg = SWEEP_CONFIG[joint_idx]
    print(f"\n{'='*70}")
    print(f"  J{joint_idx} 자세별 tau 스윕 (고정 방향: {'+' if sweep_direction>0 else '-'})")
    print(f"  고정 관절: {cfg['fixed']}")
    print(f"  스윕 자세: {cfg['poses']}")
    print(f"  스텝 {STEP_DEG}도 · 관측 {SAMPLE_SEC}s · 자세당 {REPEATS}회")
    print(f"{'='*70}")

    results = []   # (자세, tau평균, tau표준편차, t_d평균, 유효반복수, 샘플trace)
    for pose in cfg["poses"]:
        print(f"\n  --- J{joint_idx} = {pose:+.0f}도 ---")
        taus, tds, trace = measure_tau_at_pose(mc, joint_idx, pose, cfg["fixed"], sweep_direction)
        if not taus:
            print("    ⚠️ 유효한 측정이 없습니다 - 이 자세는 결과에서 제외됩니다.")
            results.append((pose, None, None, None, 0, None))
            continue
        results.append((pose, float(np.mean(taus)), float(np.std(taus)),
                        float(np.mean(tds)), len(taus), trace))

    # ---- 결과 표 ----
    print(f"\n{'='*70}")
    print(f"  J{joint_idx} 결과")
    print(f"{'='*70}")
    print(f"  {'자세(도)':<10}{'tau평균(ms)':<14}{'tau표준편차':<14}{'t_d평균(ms)':<14}{'유효반복'}")
    valid = [r for r in results if r[1] is not None]
    for pose, tm, ts, td, n, _ in results:
        if tm is None:
            print(f"  {pose:<10.0f}{'측정실패':<14}{'-':<14}{'-':<14}{n}")
        else:
            print(f"  {pose:<10.0f}{tm:<14.1f}{ts:<14.2f}{td:<14.1f}{n}")

    if len(valid) < 3:
        print("\n  ⚠️ 유효 자세가 3개 미만이라 추세 판정을 생략합니다.")
        return results

    poses = np.array([r[0] for r in valid], dtype=float)
    taus = np.array([r[1] for r in valid], dtype=float)
    stds = np.array([r[2] for r in valid], dtype=float)

    # 자세 간 변동이 반복 노이즈보다 큰가? (§15에서 쓴 '변동비'와 같은 발상)
    spread = float(taus.max() - taus.min())
    noise = float(np.mean(stds))
    ratio = spread / max(noise, 1e-9)

    # 선형 추세 (참고용 - 아래 판정에는 쓰지 않는다, 이유는 주석 참고)
    slope, intercept = np.polyfit(poses, taus, 1)
    pred = slope * poses + intercept
    ss_res = float(np.sum((taus - pred) ** 2))
    ss_tot = float(np.sum((taus - taus.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    print(f"\n  📎 자세간 tau 변동폭: {spread:.1f}ms")
    print(f"     반복 노이즈(표준편차 평균): {noise:.2f}ms")
    print(f"     변동/노이즈 비: {ratio:.1f}")
    print(f"     참고) 선형회귀: 기울기 {slope:+.3f} ms/도, R²={r2:.3f}")

    # [검증 중 정정] 처음엔 "선형(A) / 무관(B) / 2단계(C)"를 자동으로
    # 구분하려 했는데, 합성 데이터로 검증하니 A와 C 구분이 신뢰할 수
    # 없었다(선형 20/20이지만 계단형은 14/20만 맞고 나머지는 A로 오분류).
    # 자세 6점으로 "선형 추세"와 "임계자세에서 꺾임"을 가르는 건 원리적으로
    # 무리다 - 계단형도 선형회귀 R²가 0.7을 쉽게 넘는다. §25에서 ACF
    # 주기 자동판정을 포기했던 것과 같은 상황이라 같은 원칙을 따른다:
    # **신뢰할 수 있는 판정만 자동으로 하고, 나머지는 그래프로 사람이 본다.**
    # "의존 있음/없음"만 보면 합성검증에서 의존(선형·계단 모두) 30/30을
    # 잡았고, 무관인데 의존이라 잘못 단정하는 경우는 거의 없었다.
    print()
    if ratio < 1.5:
        print("  → [자세 무관] 자세간 차이가 반복 노이즈에 묻힙니다.")
        print("     그렇다면 joint_lag.txt의 tau가 곡선마다 달랐던 건 회귀")
        print("     자체의 잡음이라는 뜻이므로, 그 숫자를 지금까지보다 덜")
        print("     신뢰해야 합니다 - 이것도 중요한 결론입니다.")
    elif ratio < 2.5:
        print("  → [애매] 자세간 차이가 노이즈보다 크긴 하나 결정적이지 않습니다.")
        print("     REPEATS를 늘려 노이즈를 더 줄인 뒤 재측정하거나,")
        print("     아래 그래프에서 추세가 눈에 보이는지 직접 확인하세요.")
    else:
        print("  → [자세 의존 있음] tau가 자세에 따라 뚜렷이 달라집니다.")
        print("     §7.1/§20의 'J1=관성형 고정지연' 결론에 조건을 붙여야 합니다:")
        print("     J2/J3는 자세 영역에 따라 J1을 넘을 수 있습니다.")
        print()
        print("     [선형인지 2단계인지는 자동판정하지 않습니다]")
        print("     아래 그래프에서 직접 보세요 - 점들이 직선을 따르면 선형,")
        print("     특정 자세에서 계단처럼 꺾이면 §17의 J4(|J2|>=55 분기)처럼")
        print("     2단계 모델입니다. 어느 쪽이냐에 따라 보정 테이블 형태가")
        print("     달라집니다.")

    _plot(joint_idx, results, slope, intercept)
    return results


def _plot(joint_idx, results, slope, intercept):
    valid = [r for r in results if r[1] is not None]
    if not valid:
        return
    poses = [r[0] for r in valid]
    taus = [r[1] for r in valid]
    stds = [r[2] for r in valid]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].errorbar(poses, taus, yerr=stds, fmt="o-", capsize=4, color="#2166ac")
    xs = np.linspace(min(poses), max(poses), 50)
    axes[0].plot(xs, slope * xs + intercept, "--", color="gray", linewidth=0.9,
                 label=f"선형 {slope:+.3f} ms/도")
    axes[0].set_xlabel(f"J{joint_idx} 자세 (도)")
    axes[0].set_ylabel("tau (ms)")
    axes[0].set_title(f"J{joint_idx} 자세별 추종지연 tau (오차막대=반복 표준편차)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    # 대표 스텝응답 하나를 피팅과 겹쳐 그린다 (피팅이 타당한지 눈으로 확인)
    trace = next((r[5] for r in valid if r[5] is not None), None)
    if trace is not None:
        t, y, tau, t_d, y0, y1 = trace
        axes[1].plot(t, y, ".", markersize=4, color="#c44e52", label="실측")
        axes[1].plot(t, step_response(t, y0, y1, tau, t_d), "-", linewidth=1.5,
                     color="#2166ac", label=f"피팅 tau={tau*1000:.0f}ms")
        axes[1].set_xlabel("스텝 명령 이후 시간 (s)")
        axes[1].set_ylabel(f"J{joint_idx} 각도 (도)")
        axes[1].set_title("대표 스텝응답과 1차지연 피팅")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = f"/tmp/tau_pose_dependency_J{joint_idx}_{int(time.time())}.png"
    try:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  💾 그래프 저장: {out}")
    except Exception as e:
        print(f"  ⚠️ 그래프 저장 실패: {e}")
    fig.show()
    plt.pause(0.1)


def run_direction_check(mc, joint_idx):
    """[9차 세션, §27 후속] 방향(+/-)이 tau/t_d에 미치는 순수한 효과를
    자세와 분리해서 직접 잰다. 원래 스윕(run_sweep)은 자세마다 방향을
    고정하지만, 그것만으로는 "이 자세 자체의 효과"와 "이 자세에서 마침
    이 방향으로 쟀다는 효과"를 못 가른다. 여기서는 **같은 자세에서
    +/- 두 방향을 모두** 재서 그 차이만 뽑는다 - §15의 방향이력 실험과
    같은 발상을 tau/t_d에 적용한 것."""
    cfg = SWEEP_CONFIG[joint_idx]
    check_poses = cfg["poses"][len(cfg["poses"]) // 3::2][:3]   # 대표로 3개 정도만
    print(f"\n{'='*70}")
    print(f"  J{joint_idx} 방향효과 직접 확인 (자세 {check_poses}에서 +/- 둘 다 측정)")
    print(f"{'='*70}")

    for pose in check_poses:
        print(f"\n  --- J{joint_idx} = {pose:+.0f}도 ---")
        row = {}
        for sign, label in [(+1.0, "+"), (-1.0, "-")]:
            print(f"    [{label} 방향]")
            taus, tds, _ = measure_tau_at_pose(mc, joint_idx, pose, cfg["fixed"], sign)
            if taus:
                row[label] = (float(np.mean(taus)), float(np.mean(tds)))
            else:
                row[label] = None
        if row.get("+") and row.get("-"):
            dt = row["+"][0] - row["-"][0]
            dtd = row["+"][1] - row["-"][1]
            print(f"    📎 방향차이(+ minus -): tau {dt:+.1f}ms, t_d {dtd:+.1f}ms")
            print("       (이 값이 크면 run_sweep의 '고정 자세방향' 결과에도")
            print("        그 방향의 편향이 그대로 실려 있다는 뜻)")


def _max_gap_split(values):
    """정렬 후 가장 큰 갭을 기준으로 값을 두 그룹으로 나눈다 (1차원 2군집의
    가장 단순한 형태). 반환: (저그룹 리스트, 고그룹 리스트, 갭/전체범위 비).
    갭비가 클수록(예: >0.5) 두 그룹이 뚜렷이 갈라져 있다는 뜻 - 표본이 8~10개뿐이라
    정식 이봉성 검정(예: dip test)은 신뢰하기 어렵고, 이 조사가 필요로 하는 건
    "육안으로도 뚜렷한 갈림인가" 수준의 판정이라 이 정도로 충분하다."""
    s = sorted(values)
    if len(s) < 2:
        return s, [], 0.0
    gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    gi = int(np.argmax(gaps))
    span = s[-1] - s[0]
    gap_ratio = gaps[gi] / span if span > 1e-9 else 0.0
    return s[:gi + 1], s[gi + 1:], gap_ratio


def run_j3_zero_bimodal_check(mc):
    """[9차 세션, §27.4-1, 최우선] J3=0도 · `-`방향 고정에서 REPEATS를
    8~10으로 늘려 §27.3의 반반 갈림(반복1·2 92ms대 / 반복3·4 108ms대)이
    재현되는지 확인한다. 두 가설을 분리해서 본다:

      (A) 정착 중(warm-up) 가설 - 방향 블록 전환 직후 1~2회만 불안정하고
          그 뒤로는 한 값에 정착한다면, 반복 순서(전반부 vs 후반부)와
          값이 상관관계를 보여야 한다.
      (B) 준확률적 이봉분포 가설 - 매 반복이 독립적으로 두 상태 중
          하나로 무작위 낙착한다면, 순서와는 무관하게 값 자체가 두
          군집으로 갈라지되 순서 상관은 약해야 한다.

    이 스크립트만으로 인과를 증명할 수는 없다 - 8~10개 표본으로는 정식
    통계검정력이 약하므로, 아래 출력과 그래프를 사람이 직접 보고 어느
    가설에 가까운지 판단하는 것을 전제로 한다(§25 ACF 자동판정 포기,
    §27 2단계/선형 자동판정 포기와 같은 원칙)."""
    cfg = SWEEP_CONFIG[3]
    pose = 0.0
    print(f"\n{'='*70}")
    print(f"  [§27.4-1] J3 자세0도 `-`방향 이봉분포 재현확인 "
          f"(REPEATS={BIMODAL_CHECK_REPEATS})")
    print(f"{'='*70}")

    taus, tds, trace = measure_tau_at_pose(
        mc, 3, pose, cfg["fixed"], sweep_direction=-1.0, repeats=BIMODAL_CHECK_REPEATS
    )

    if len(taus) < 4:
        print(f"\n  ⚠️ 유효 반복이 {len(taus)}회뿐 - 판정하기엔 너무 적습니다.")
        return

    print(f"\n  --- 반복 순서대로(1~{len(taus)}) ---")
    for i, (t, d) in enumerate(zip(taus, tds), start=1):
        half = "전반부" if i <= len(taus) / 2 else "후반부"
        print(f"    반복{i:>2}: tau={t:6.1f}ms  t_d={d:5.1f}ms  [{half}]")

    # (A) 순서 기반 - 전/후반부 평균 비교
    mid = len(taus) // 2
    first_half = taus[:mid] if len(taus) % 2 == 0 else taus[:mid + 1]
    second_half = taus[len(first_half):]
    if second_half:
        fh_mean, sh_mean = float(np.mean(first_half)), float(np.mean(second_half))
        print(f"\n  (A) 순서 기반: 전반부 평균 {fh_mean:.1f}ms (n={len(first_half)})"
              f"  vs  후반부 평균 {sh_mean:.1f}ms (n={len(second_half)})"
              f"  차이 {sh_mean - fh_mean:+.1f}ms")

    # (B) 값 기반 - 정렬 후 최대갭으로 2그룹 분리
    low, high, gap_ratio = _max_gap_split(taus)
    print(f"\n  (B) 값 기반(정렬 후 최대갭 분리): "
          f"저그룹 n={len(low)} 평균={np.mean(low):.1f}ms  /  "
          f"고그룹 n={len(high)} 평균={np.mean(high):.1f}ms  "
          f"(갭비={gap_ratio:.2f}, 1에 가까울수록 뚜렷한 이분)")

    print()
    if gap_ratio > 0.5 and len(low) >= 2 and len(high) >= 2:
        print("  → 값 자체가 뚜렷한 두 군집으로 갈라집니다(이봉 가능성 높음).")
        print("     전/후반부 평균 차이가 크면(A) '정착 중'과 겹친 효과일 수 있고,")
        print("     차이가 작으면(A) 순서와 무관한 순수 이봉(B)에 가깝습니다.")
        print("     -> §27.4-1에서 원했던 답: 이 경우 '자세별 tau'가 아니라")
        print("        '이 자세·방향의 tau 자체가 불안정하다'는 결론이 지지됩니다.")
    else:
        print("  → 값이 두 군집으로 뚜렷이 갈리지 않습니다 - REPEATS=4였을 때 우연히")
        print("     반반으로 보였을 가능성을 배제할 수 없습니다. 이 경우 §27.3의")
        print("     '반반 갈림'은 재현되지 않은 것으로 봐야 합니다.")

    _plot_bimodal(taus, tds, trace)


def _plot_bimodal(taus, tds, trace):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    order = np.arange(1, len(taus) + 1)
    axes[0].plot(order, taus, "o-", color="#2166ac")
    axes[0].axhline(np.mean(taus), color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("반복 순서")
    axes[0].set_ylabel("tau (ms)")
    axes[0].set_title("반복 순서별 tau (정착 중 가설이면 앞쪽만 다르게 보임)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(taus, bins=min(6, max(3, len(taus) // 2)), color="#5aa9c9", edgecolor="white")
    axes[1].set_xlabel("tau (ms)")
    axes[1].set_ylabel("빈도")
    axes[1].set_title("tau 분포 (이봉이면 두 덩이로 갈라짐)")
    axes[1].grid(alpha=0.3)

    fig.suptitle("J3 자세0도 `-`방향 이봉분포 재현확인 (§27.4-1)")
    fig.tight_layout()
    out = f"/tmp/tau_j3_zero_bimodal_check_{int(time.time())}.png"
    try:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  💾 그래프 저장: {out}")
    except Exception as e:
        print(f"  ⚠️ 그래프 저장 실패: {e}")
    fig.show()
    plt.pause(0.1)


def main():
    print("🔌 로봇 포트 탐색 중...")
    mc, port = find_robot_port()
    if mc is None:
        print("❌ 로봇을 찾지 못했습니다. USB 연결/권한을 확인하세요.")
        sys.exit(1)
    print(f"✅ 연결됨: {port}")
    print("\n[주의] 로봇이 여러 자세로 반복 이동합니다. 주변을 비워주세요.")
    print(f"       예상 소요: 관절당 자세 6개 × 반복 {REPEATS}회 ≈ 15분")
    print("       [10차 세션, §34] J4 스윕은 fixed 자세가 실기로 충돌검증된 적이")
    print("       없다 - [7] 실행 전에 그 자세로 한 번 손으로 이동시켜 눈으로")
    print("       먼저 확인하는 걸 권장한다.")

    while True:
        print("\n--- 메뉴 ---")
        choice = input(
            "  [1] J2 자세별 tau 스윕 (고정방향, 권장 - 중력부하 가장 큼)\n"
            "  [2] J3 자세별 tau 스윕 (고정방향)\n"
            "  [3] 둘 다 연속 실행\n"
            "  [4] J2 방향효과 직접 확인 (자세는 고정, +/- 둘 다 측정)\n"
            "  [5] J3 방향효과 직접 확인\n"
            "  [6] [최우선] J3 자세0도 `-`방향 이봉분포 재현확인 (REPEATS=10, §27.4-1)\n"
            "  [7] [10차 신규] J4 자세별 tau 스윕 (고정방향, §34 - fixed자세 미검증 주의)\n"
            "  [8] [10차 신규] J4 방향효과 직접 확인\n"
            "  [q] 종료\n"
            "> "
        ).strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            run_sweep(mc, 2)
        elif choice == "2":
            run_sweep(mc, 3)
        elif choice == "3":
            run_sweep(mc, 2)
            run_sweep(mc, 3)
        elif choice == "4":
            run_direction_check(mc, 2)
        elif choice == "5":
            run_direction_check(mc, 3)
        elif choice == "6":
            run_j3_zero_bimodal_check(mc)
        elif choice == "7":
            run_sweep(mc, 4)
        elif choice == "8":
            run_direction_check(mc, 4)
        else:
            print("잘못된 입력입니다.")

    print("종료합니다.")


if __name__ == "__main__":
    main()
