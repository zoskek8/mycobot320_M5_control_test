# -*- coding: utf-8 -*-
"""
mycobot_encoder_noise_test.py
===============================
[10차 세션, §38 조사] **정지 상태에서 `get_angles()`를 반복해서 읽어,
측정값 자체의 노이즈 성질을 정량화한다.**

[왜 이 조사를 하는가 - §38 논의]
지금까지 이 프로젝트가 다뤄온 "노이즈"는 전부 뭉뚱그려져 있었다.
그런데 처방이 완전히 갈리는 두 종류가 섞여 있다:

  (a) **측정 노이즈** - 로봇은 실제로 안 움직이는데 숫자만 튀는 것.
      엔코더 양자화, 시리얼 통신 지터 등. Figure 2의 "1-sample diff"가
      목표 35mm/s인데 165mm/s까지 치솟는 게 전형적인 증거다(실제로
      5배 빠를 리 없다).
  (b) **실제 진동/백래시** - 로봇이 진짜로 떠는 것. §27.6.1의 J3
      이봉분포가 여기 해당.

(a)는 소프트웨어로 지울 수 있고 (b)는 못 지운다. 그런데 현재 P 피드백은
둘을 구분하지 않고 전부 "오차"로 받아 밀어낸다 - §29.20의 D항이
실패한(§34.1) 이유도 정확히 이것으로 보인다: **원시 측정값을 미분**하니
양자화 노이즈가 그대로 증폭됐다.

**정지 상태에서 읽으면 (b)가 0이므로 (a)만 순수하게 분리된다** - 이게
이 스크립트의 핵심 아이디어다. 로봇을 움직이지 않으므로 위험도 없고
1~2분이면 끝난다.

[무엇을 알아내는가]
  1. **양자화 계단(Δ)** - 관측된 값들이 이산적인 격자에 놓이는지, 그
     간격이 얼마인지. 양자화 노이즈는 [-Δ/2, +Δ/2] 균등분포로 **유계이고
     성질이 알려져** 있어서, 크기를 알면 "추측"이 아니라 거의 계산으로
     걷어낼 수 있다.
  2. **관절별 노이즈 표준편차** - 필터(칼만 등)의 측정노이즈 공분산 R을
     정하는 직접 근거가 된다. 지금은 이 값을 몰라서 필터 설계를 시작할
     수조차 없다.
  3. **드리프트 유무** - 정지 상태인데 값이 한 방향으로 흐르면 순수
     노이즈가 아니라 열/중력 처짐 같은 다른 현상이 있다는 뜻이다.
  4. **`P_FEEDBACK_DEADBAND_DEG=0.2` 검증** - §29.20에서 실험적으로
     고른 값인데, 노이즈 크기와 비교해 타당한지 이제야 근거를 갖고
     판단할 수 있다. 노이즈 3σ보다 데드밴드가 작으면 피드백이 노이즈를
     쫓고 있다는 뜻이다.

[로봇은 움직이지 않는다] 현재 자세에서 그대로 읽기만 한다. 다만
**서보에 힘이 들어간 상태(정지 유지 중)와 완전히 이완된 상태의 노이즈가
다를 수 있으므로**, [2]로 몇 가지 자세를 옮겨가며 재는 옵션도 뒀다
(이때만 로봇이 움직인다).

사용법:
    python3 mycobot_encoder_noise_test.py

    [1] 현재 자세에서 노이즈 측정 (권장 첫 실행 - 로봇 안 움직임)
    [2] 여러 자세에서 측정 (자세에 따라 노이즈가 다른지 - 로봇 이동함)
    [3] 샘플링 주기만 측정 (get_angles() 호출이 실제로 얼마나 걸리는가)
    [4] [10차 신규, §39] fast_get_angles(pymycobot 우회) 값 대조 + 속도비교.
        **이 검증을 통과하기 전에는 USE_FAST_GET_ANGLES를 켜지 말 것** -
        응답 프레임 형식이 아직 추정이다. 로봇은 움직이지 않는다.
    [q] 종료
"""

import sys
import time
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from mycobot_kinematics import (
    find_robot_port, JOINT_LIMITS_DEG, SPEED,
    SETTLE_DELAY_SEC, MOVE_TIMEOUT_SEC,
)

SETTLE_EXTRA_SEC = 1.0     # 이동 후 서보가 완전히 자리 잡을 때까지(다른 스크립트와 동일)
N_SAMPLES = 400            # 정지 상태 반복 읽기 횟수. 400이면 표준편차 추정이
                           # 충분히 안정되고(상대오차 ~3.5%), 48ms 주기 가정 시
                           # 20초 정도라 지루하지도 않다.
GRID_TOL = 1e-6            # 값이 같은 격자점인지 판정할 때의 부동소수 허용오차

# [2]번에서 훑을 자세들 - J2/J3를 움직여 중력부하가 크게 다른 상황을 만든다
# (정지 유지 토크가 다르면 노이즈도 다를 수 있다는 가설 확인용).
POSE_SET = [
    ("수직 정렬(부하 최소)", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("팔 뻗음(부하 큼)", [0.0, -60.0, 30.0, 0.0, 30.0, 0.0]),
    ("반대쪽 뻗음", [0.0, 30.0, -30.0, 0.0, -30.0, 0.0]),
]


def collect_samples(mc, n=N_SAMPLES):
    """정지 상태에서 get_angles()를 n번 읽는다.
    반환: (angles 배열 shape(n,6), 각 읽기 시각 배열 shape(n,))
    읽기가 실패하거나 6개가 아니면 그 샘플은 건너뛴다."""
    rows = []
    stamps = []
    t_start = time.time()
    fails = 0
    while len(rows) < n:
        a = mc.get_angles()
        if isinstance(a, list) and len(a) == 6:
            rows.append(list(a))
            stamps.append(time.time() - t_start)
        else:
            fails += 1
            if fails > n:      # 절반 이상 실패하면 포기 - 연결 문제
                break
    if fails:
        print(f"    ⚠️ 읽기 실패 {fails}회(무시하고 계속했습니다)")
    return np.asarray(rows, dtype=float), np.asarray(stamps, dtype=float)


def detect_quantization_step(values):
    """한 관절의 값 배열에서 양자화 계단 크기를 추정한다.

    방법: 서로 다른 값들을 정렬해 인접 차이를 구하고, 그 **최소 양수 차이**를
    본다. 값이 진짜 격자 위에 놓여 있다면 모든 인접 차이가 이 최소값의
    정수배여야 한다 - 그 정수배 여부까지 확인해서 격자 가설이 맞는지
    판정한다(그냥 최소차이만 보면 우연히 가까운 두 값에 속을 수 있다).

    반환: (계단추정값 또는 None, 고유값 개수, 격자설명 문자열)
    """
    uniq = np.unique(values)
    if len(uniq) < 2:
        return None, len(uniq), "값이 하나뿐 - 완전히 고정(노이즈 없음 또는 표본 부족)"

    diffs = np.diff(uniq)
    diffs = diffs[diffs > GRID_TOL]
    if len(diffs) == 0:
        return None, len(uniq), "유효한 차이 없음"

    step = float(np.min(diffs))
    # 모든 인접 차이가 step의 정수배인지 - 격자 가설 검정
    ratios = diffs / step
    residual = np.abs(ratios - np.round(ratios))
    grid_ok = bool(np.max(residual) < 0.05)   # 5% 이내면 정수배로 인정
    if grid_ok:
        desc = f"격자 확인됨(모든 간격이 {step:.4f}도의 정수배)"
    else:
        desc = (f"격자 아님(간격이 {step:.4f}도의 정수배가 아닌 게 섞임 - "
                f"양자화 외 다른 노이즈가 지배적일 수 있음)")
    return step, len(uniq), desc


def analyze(angles, stamps, label=""):
    """수집한 샘플을 관절별로 분석해 출력하고, 요약 dict를 반환한다."""
    n = len(angles)
    if n < 10:
        print("  ⚠️ 표본이 너무 적습니다.")
        return None

    print(f"\n{'='*74}")
    print(f"  결과{(' - ' + label) if label else ''}  (표본 {n}개)")
    print(f"{'='*74}")

    if len(stamps) > 1:
        dt = np.diff(stamps)
        print(f"  샘플링 주기: 평균 {np.mean(dt)*1000:.1f}ms  "
              f"중앙값 {np.median(dt)*1000:.1f}ms  최대 {np.max(dt)*1000:.1f}ms")
        print(f"  총 소요: {stamps[-1]:.1f}초")

    print(f"\n  {'관절':<5}{'평균(도)':>11}{'표준편차(도)':>13}{'p-p폭(도)':>11}"
          f"{'고유값수':>9}{'계단(도)':>10}{'드리프트(도)':>13}")

    summary = {}
    for j in range(6):
        v = angles[:, j]
        step, n_uniq, _desc = detect_quantization_step(v)
        # 드리프트: 앞 10%와 뒤 10%의 평균 차이 - 정지 상태인데 한 방향으로
        # 흐르면 순수 노이즈가 아니다.
        k = max(2, n // 10)
        drift = float(np.mean(v[-k:]) - np.mean(v[:k]))
        std = float(np.std(v))
        pp = float(np.max(v) - np.min(v))
        step_txt = f"{step:>10.4f}" if step is not None else f"{'-':>10}"
        print(f"  J{j+1:<4}{np.mean(v):>11.4f}{std:>13.4f}{pp:>11.4f}"
              f"{n_uniq:>9}{step_txt}{drift:>13.4f}")
        summary[j] = {"mean": float(np.mean(v)), "std": std, "pp": pp,
                      "n_uniq": n_uniq, "step": step, "drift": drift}

    # 격자 판정 상세 - 표가 좁아서 위에 못 넣은 설명을 따로 출력
    print("\n  --- 격자(양자화) 판정 ---")
    for j in range(6):
        _step, _n_uniq, desc = detect_quantization_step(angles[:, j])
        print(f"    J{j+1}: {desc}")

    interpret(summary)
    return summary


def interpret(summary):
    """측정값을 §38의 판단 기준에 비춰 해석해준다."""
    print("\n  --- 해석 ---")

    stds = [summary[j]["std"] for j in range(6)]
    max_std = max(stds)
    worst = int(np.argmax(stds)) + 1

    # 1) 데드밴드 타당성 - 현재 P_FEEDBACK_DEADBAND_DEG=0.2와 비교
    DEADBAND_NOW = 0.2
    print(f"  [데드밴드] 현재 P_FEEDBACK_DEADBAND_DEG = {DEADBAND_NOW}도")
    for j in range(6):
        s = summary[j]["std"]
        if s <= 0:
            continue
        three_sigma = 3.0 * s
        if three_sigma > DEADBAND_NOW:
            print(f"    ⚠️ J{j+1}: 노이즈 3σ={three_sigma:.3f}도 > 데드밴드 {DEADBAND_NOW}도 "
                  f"- 이 관절은 피드백이 노이즈를 쫓고 있을 수 있습니다.")
    if all(3.0 * summary[j]["std"] <= DEADBAND_NOW for j in range(6)):
        print(f"    ✅ 전 관절 3σ가 데드밴드 안 - 현재 설정이 노이즈를 잘 막고 있습니다.")

    # 2) 필터 설계용 R 값
    print(f"\n  [필터 설계] 칼만 필터를 쓴다면 측정노이즈 분산 R의 근거값:")
    print("    " + "  ".join(f"J{j+1}={summary[j]['std']**2:.5f}" for j in range(6))
          + "   (단위: 도²)")
    print(f"    가장 노이즈가 큰 관절: J{worst} (σ={max_std:.4f}도)")

    # 3) 드리프트 경고
    for j in range(6):
        d = abs(summary[j]["drift"])
        s = summary[j]["std"]
        if s > 0 and d > 3.0 * s:
            print(f"\n  ⚠️ [드리프트] J{j+1}이 정지 상태인데 {summary[j]['drift']:+.4f}도 "
                  f"흘렀습니다(노이즈 3σ={3*s:.4f}도보다 큼).")
            print("     순수 측정노이즈가 아니라 열/중력처짐/서보 미세이동일 수 있습니다 -")
            print("     이 경우 필터의 '정지 = 변화없음' 가정이 깨지므로 따로 조사해야 합니다.")

    # 4) 양자화가 지배적인지
    quantized = [j for j in range(6) if summary[j]["step"] is not None
                 and summary[j]["std"] > 0
                 and summary[j]["step"] > summary[j]["std"]]
    if quantized:
        names = ", ".join(f"J{j+1}" for j in quantized)
        print(f"\n  [양자화 지배] {names}: 계단 크기가 표준편차보다 큽니다 -")
        print("     노이즈의 주된 원인이 엔코더 양자화라는 뜻입니다. 이 경우")
        print("     단순 평균/저역통과보다 양자화를 명시적으로 모델링한 추정이")
        print("     효과적입니다(균등분포 [-Δ/2, +Δ/2]로 알려진 성질을 씁니다).")


def plot_samples(angles, stamps, label=""):
    """관절별 시계열 + 히스토그램. 격자에 놓이는지 눈으로 확인하는 게 목적."""
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    for j in range(6):
        v = angles[:, j]
        ax = axes[0][j]
        ax.plot(stamps, v, ".-", ms=3, lw=0.6)
        ax.set_title(f"J{j+1} 시계열")
        ax.set_xlabel("t (s)")
        if j == 0:
            ax.set_ylabel("각도 (도)")
        ax.grid(alpha=0.3)

        ax2 = axes[1][j]
        uniq = np.unique(v)
        bins = len(uniq) if 1 < len(uniq) < 40 else 30
        ax2.hist(v, bins=bins)
        ax2.set_title(f"J{j+1} 분포 (고유 {len(uniq)}개)")
        ax2.set_xlabel("각도 (도)")
        ax2.grid(alpha=0.3)

    fig.suptitle(f"정지 상태 get_angles() 노이즈{(' - ' + label) if label else ''} "
                 f"- 히스토그램이 이산 막대면 양자화 지배")
    fig.tight_layout()
    out = "/tmp/encoder_noise.png"
    try:
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"\n💾 그래프 저장: {out}")
    except Exception as e:
        print(f"\n⚠️ 그래프 저장 실패: {e}")
    fig.show()
    plt.pause(0.1)


def goto_and_settle(mc, angles):
    """다른 진단 스크립트(mycobot_tau_pose_dependency_test.py)와 같은 방식."""
    mc.send_angles(list(angles), SPEED)
    time.sleep(SETTLE_DELAY_SEC)
    t0 = time.time()
    while mc.is_moving():
        if time.time() - t0 > MOVE_TIMEOUT_SEC:
            break
        time.sleep(0.1)
    time.sleep(SETTLE_EXTRA_SEC)


def cmd_current_pose(mc):
    """[1] 지금 자세 그대로 - 로봇을 전혀 움직이지 않는다."""
    cur = mc.get_angles()
    print(f"\n현재 자세: {[round(a, 2) for a in cur] if cur else '읽기 실패'}")
    print(f"{N_SAMPLES}개 샘플을 읽습니다 (로봇은 움직이지 않습니다)...")
    angles, stamps = collect_samples(mc)
    summary = analyze(angles, stamps, label="현재 자세")
    if summary and input("\n그래프를 볼까요? (y/N): ").strip().lower() == "y":
        plot_samples(angles, stamps, label="현재 자세")
    return summary


def cmd_multi_pose(mc):
    """[2] 여러 자세 - 정지 유지 토크가 다르면 노이즈도 다른지 확인."""
    print("\n[주의] 로봇이 아래 자세들로 이동합니다. 주변을 비워주세요.")
    for name, pose in POSE_SET:
        print(f"    - {name}: {pose}")
    if input("계속할까요? (y/N): ").strip().lower() != "y":
        print("취소했습니다.")
        return

    results = {}
    for name, pose in POSE_SET:
        # 관절 한계 확인 - 다른 스크립트와 같은 원칙으로 미리 막는다
        bad = [i for i, a in enumerate(pose)
               if not (JOINT_LIMITS_DEG[i][0] <= a <= JOINT_LIMITS_DEG[i][1])]
        if bad:
            print(f"  ⚠️ [{name}] 관절한계 밖(J{[b+1 for b in bad]}) - 건너뜁니다.")
            continue
        print(f"\n--- [{name}] 이동 중... ---")
        goto_and_settle(mc, pose)
        angles, stamps = collect_samples(mc)
        results[name] = analyze(angles, stamps, label=name)

    if len(results) >= 2:
        print(f"\n{'='*74}")
        print("  자세별 노이즈 표준편차 비교 (도)")
        print(f"{'='*74}")
        print(f"  {'자세':<24}" + "".join(f"{'J'+str(j+1):>10}" for j in range(6)))
        for name, s in results.items():
            if s:
                print(f"  {name:<24}" + "".join(f"{s[j]['std']:>10.4f}" for j in range(6)))
        print("\n  → 자세에 따라 크게 다르면 노이즈가 부하의존이라는 뜻입니다")
        print("    (필터의 R을 상수로 두면 안 되고 자세별로 바꿔야 함).")
        print("    비슷하면 R을 상수로 둬도 됩니다 - 설계가 훨씬 간단해집니다.")


def cmd_sampling_rate(mc):
    """[3] get_angles() 호출 자체의 소요시간 분포.

    §35에서 '평균이 아니라 꼬리가 문제'라는 걸 배웠으므로, 여기서도
    평균만이 아니라 p99/최댓값을 같이 본다."""
    n = 200
    print(f"\nget_angles() {n}회 호출 시간을 잽니다 (로봇은 움직이지 않습니다)...")
    durations = []
    for _ in range(n):
        t0 = time.perf_counter()
        a = mc.get_angles()
        dt = time.perf_counter() - t0
        if isinstance(a, list) and len(a) == 6:
            durations.append(dt)
    if not durations:
        print("⚠️ 유효한 읽기가 없습니다.")
        return
    d = np.asarray(durations) * 1000.0
    print(f"\n  호출시간(ms): 평균 {d.mean():.2f}  중앙값 {np.median(d):.2f}  "
          f"p95 {np.percentile(d,95):.2f}  p99 {np.percentile(d,99):.2f}  최대 {d.max():.2f}")
    print(f"  10ms 초과: {int((d>10).sum())}/{len(d)}회   "
          f"20ms 초과: {int((d>20).sum())}/{len(d)}회")
    print("\n  → 스트리밍 루프의 목표주기는 48ms다(ABSORB_TARGET_PERIOD_SEC).")
    print("    '매번 실측' 모드는 매 사이클 이 호출을 하므로, 최댓값이 48ms에")
    print("    가까우면 그 자체로 주기미달의 원인이 될 수 있다 - §36.1에서")
    print("    끝내 못 밝힌 교란변수의 후보이기도 하다.")


def cmd_verify_fast_read(mc):
    """[4] [10차, §39] fast_get_angles()가 pymycobot get_angles()와
    **값이 정확히 일치하는지** 대조하고, 속도 이득을 잰다.

    6-b의 송신 우회는 pymycobot 출력과 바이트 단위로 대조해 검증했지만,
    이 수신 파서는 응답 프레임 형식이 추정이라 아직 그 대조를 안 거쳤다.
    이 메뉴가 그 대조다 - **통과하기 전에는 USE_FAST_GET_ANGLES를 켜지 말 것.**
    로봇은 움직이지 않는다."""
    try:
        from mycobot_kinematics import fast_get_angles
    except ImportError:
        print("⚠️ fast_get_angles를 찾을 수 없습니다 - mycobot_kinematics.py가 최신인지 확인하세요.")
        return

    n = 100
    print(f"\n[1/2] 값 대조 - 같은 정지 자세에서 두 방식을 번갈아 {n}회 읽습니다...")
    mismatches = []
    fails = 0
    for i in range(n):
        slow = mc.get_angles()
        fast = fast_get_angles(mc)
        if fast is None:
            fails += 1
            continue
        if not (isinstance(slow, list) and len(slow) == 6):
            continue
        # 정지 상태이므로 두 값은 정확히 같아야 한다(엔코더가 안 변하므로).
        if any(abs(a - b) > 1e-9 for a, b in zip(slow, fast)):
            mismatches.append((i, slow, fast))

    print(f"    읽기 실패(파싱 불가): {fails}/{n}회")
    print(f"    값 불일치: {len(mismatches)}/{n - fails}회")
    if mismatches[:3]:
        for idx, slow, fast in mismatches[:3]:
            print(f"      #{idx}  pymycobot={slow}")
            print(f"           fast     ={fast}")

    ok = (fails == 0 and not mismatches)
    if not ok:
        print("\n  ❌ 검증 실패 - USE_FAST_GET_ANGLES를 켜지 마세요.")
        if fails:
            print("     파싱 실패가 있다면 응답 프레임 형식(명령코드/길이)이 추정과")
            print("     다를 수 있습니다. mycobot_kinematics.py의 parse_angles_response()를")
            print("     실제 수신 바이트와 대조해 고쳐야 합니다.")
        return

    print("\n  ✅ 값 대조 통과 - 두 방식이 100% 동일한 값을 돌려줍니다.")

    print(f"\n[2/2] 속도 비교 - 각 {n}회씩...")
    t_slow = []
    for _ in range(n):
        t0 = time.perf_counter()
        mc.get_angles()
        t_slow.append(time.perf_counter() - t0)
    t_fast = []
    for _ in range(n):
        t0 = time.perf_counter()
        fast_get_angles(mc)
        t_fast.append(time.perf_counter() - t0)

    s = np.asarray(t_slow) * 1000
    f = np.asarray(t_fast) * 1000
    print(f"    pymycobot : 평균 {s.mean():6.2f}ms  p99 {np.percentile(s,99):6.2f}  최대 {s.max():6.2f}")
    print(f"    fast      : 평균 {f.mean():6.2f}ms  p99 {np.percentile(f,99):6.2f}  최대 {f.max():6.2f}")
    saved = s.mean() - f.mean()
    print(f"    절약: 사이클당 {saved:.2f}ms  ({s.mean()/max(f.mean(),1e-9):.1f}배 빠름)")

    budget = 48.0
    print(f"\n  [사이클 예산 영향] 목표주기 {budget:.0f}ms 기준")
    print(f"    이전: 읽기 {s.mean():.1f}ms -> 나머지 작업에 {budget - s.mean():.1f}ms")
    print(f"    이후: 읽기 {f.mean():.1f}ms -> 나머지 작업에 {budget - f.mean():.1f}ms")
    if saved > 3.0:
        print(f"\n  ✅ 의미 있는 개선입니다. mycobot_kinematics.py의")
        print("     USE_FAST_GET_ANGLES = True 로 바꾸면 적용됩니다.")
        print("     ⚠️ 바꾼 뒤 곡선 실행으로 주기미달/유효지연이 실제로")
        print("        나아지는지 확인할 것(§39).")
    else:
        print(f"\n  ⚠️ 절약이 {saved:.1f}ms뿐입니다 - 펌웨어 응답시간이 지배적이라")
        print("     라이브러리 우회로 얻을 게 별로 없다는 뜻입니다(§39의 '기대치'")
        print("     주석 참고). 켜도 큰 효과는 없을 수 있습니다.")


def main():
    print("🔌 로봇 포트 탐색 중...")
    mc, port = find_robot_port()
    if mc is None:
        print("❌ 로봇을 찾지 못했습니다. USB 연결/권한을 확인하세요.")
        sys.exit(1)
    print(f"✅ 연결됨: {port}")
    print("\n[1]/[3]번은 로봇을 전혀 움직이지 않습니다 - 안전합니다.")
    print("[2]번만 여러 자세로 이동하므로 주변을 확인하세요.")

    while True:
        print("\n--- 메뉴 ---")
        choice = input(
            "  [1] 현재 자세에서 노이즈 측정 (권장 첫 실행 - 로봇 안 움직임)\n"
            "  [2] 여러 자세에서 측정 (부하에 따라 노이즈가 다른지 - 로봇 이동함)\n"
            "  [3] 샘플링 주기만 측정 (get_angles() 호출 소요시간 - 로봇 안 움직임)\n"
            "  [4] [10차 신규, §39] fast_get_angles 검증 + 속도비교 (로봇 안 움직임)\n"
            "  [q] 종료\n"
            "> "
        ).strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            cmd_current_pose(mc)
        elif choice == "2":
            cmd_multi_pose(mc)
        elif choice == "3":
            cmd_sampling_rate(mc)
        elif choice == "4":
            cmd_verify_fast_read(mc)
        else:
            print("잘못된 입력입니다.")

    print("종료합니다.")


if __name__ == "__main__":
    main()
