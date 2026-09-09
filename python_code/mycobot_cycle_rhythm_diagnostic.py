# -*- coding: utf-8 -*-
"""
mycobot_cycle_rhythm_diagnostic.py
====================================
[9차 세션, 진단용 1회성 스크립트] "전송이 빠름-빠름-빠름-느림으로
반복되는 것 같다"는 관찰이 실제 주기적 리듬인지, 아니면 그렇게 느껴질
뿐인지(사람 눈으로 33ms 단위 리듬을 구분하는 건 원래 어렵다)를 숫자로
확정한다. mycobot_vibration_diagnostic.py와 같은 위치의
"GUI 없는 1회 실행용 스크립트" - 로봇 연결도 필요 없다. 순수하게
mycobot_cycle_log.py에 이미 쌓인 로그만 읽어서 분석한다.

**전제**: mycobot_cycle_log.py에 기록이 있으려면, 먼저 GUI로 스트리밍+
'매번 실측' 모드로 곡선을 한 번 이상 실행해야 한다(mycobot_path_editor.py의
execute_path가 실행 끝에 자동으로 남긴다).

세 가지를 본다:
  1) 시계열 그래프 - 눈으로 리듬의 존재/형태를 먼저 확인
  2) 히스토그램 - 사이클 시간의 분포 형태 (이중봉이면 "빠름팀/느림팀"이
     실제로 갈린다는 뜻)
  3) 자기상관함수(ACF) - 진짜 주기가 있다면 그 주기(N사이클)에서 뚜렷한
     봉우리가 뜬다. 이게 이 스크립트의 핵심 - 시계열을 육안으로 보고
     "느낌"으로 주기를 짚는 게 아니라, 통계적으로 확정한다.

사용법:
    python3 mycobot_cycle_rhythm_diagnostic.py

    [1] 최신 기록 분석
    [2] 최근 N개 기록을 나란히 비교 (STREAM_SEND_SPEED를 바꿔가며
        여러 번 실행했다면, 값이 리듬 강도에 미치는 영향을 한눈에 비교)
    [q] 종료

해석 가이드:
    - ACF가 lag=1 이후 빠르게 0 근처로 죽으면: 사이클끼리 독립적인 노이즈.
      "리듬"으로 느낀 것은 착시이거나 다른 원인(예: 방향전환 시 백래시
      계단점프처럼 훨씬 느린 주기)일 가능성이 크다.
    - ACF가 특정 lag(예: 4, 8)에서 뚜렷한 봉우리(관례상 ±1.96/sqrt(N)
      유의선 밖)를 반복적으로 보이면: 그 lag가 리듬의 실제 주기다.
    - 히스토그램이 뚜렷한 이중봉이면: "빠른 사이클"과 "느린 사이클"
      두 그룹이 실제로 나뉜다는 뜻 - STREAM_SEND_SPEED로 인한
      "도착 후 대기"가설(로봇이 목표에 먼저 도착해 다음 명령까지
      멈춰있다)과 부합한다.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 로컬 환경에 맞게 필요시 바꿀 것
import matplotlib.pyplot as plt

import mycobot_cycle_log as cycle_log


def autocorrelation(x, max_lag):
    """정규화된 자기상관함수. x는 이미 평균을 뺀 배열이 아니어도 된다
    (여기서 뺀다). 반환: lag 0..max_lag에 대한 상관계수 배열(lag 0은 항상 1.0)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var <= 1e-12:
        return np.zeros(max_lag + 1)
    acf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.dot(x[:-lag], x[lag:]) / ((n - lag) * var)
    return acf


def analyze_one(rec):
    label = rec.get("label", "?")
    ts = rec.get("timestamp", "?")
    cycles_ms = np.array(rec.get("cycle_times_ms", []), dtype=float)
    n = len(cycles_ms)
    send_speed = rec.get("send_speed")
    extra = rec.get("extra") or {}

    if n < 10:
        print(f"⚠️ [{ts}] 사이클 수가 너무 적습니다({n}개) - 분석 생략.")
        return

    print(f"\n{'='*70}")
    print(f"  [{ts}]  [{label}]  send_speed={send_speed}  사이클수={n}")
    print(f"  measure_mode={extra.get('measure_mode')}  "
          f"late_ratio={extra.get('late_ratio')}  "
          f"target_period={extra.get('target_period_ms')}ms")
    print(f"{'='*70}")
    print(f"  평균 {cycles_ms.mean():.2f}ms / 표준편차 {cycles_ms.std():.2f}ms / "
          f"최소 {cycles_ms.min():.2f}ms / 최대 {cycles_ms.max():.2f}ms")

    max_lag = min(30, n // 3)
    acf = autocorrelation(cycles_ms, max_lag)
    # 백색잡음(순수 노이즈)이라면 acf[lag>=1]이 대략 이 범위 안에 있어야 한다
    sig_level = 1.96 / np.sqrt(n)

    # [검증 중 발견, 3차] "가장 작은 유의 lag"도, "±1/±2 지역 극댓값"도
    # 합성 데이터 반복검증(시드 8개)에서 안정적으로 진짜 주기를 못 짚었다 -
    # 이 프로젝트의 파형(대부분 낮은 값 + 간헐적 튐, 비대칭 듀티사이클)은
    # 조화성분이 복잡해서 단순 규칙으로 "이게 그 주기다"라고 자동 판정하면
    # 틀린 확신을 줄 위험이 컸다. 대신 **믿을 수 있는 판정만 자동으로 한다** -
    # "유의미한 리듬이 있는가 없는가"는 8개 시드 전부 안정적으로 맞았다
    # (백색잡음은 유의 lag 0~4개, 주기적 신호는 항상 모든 lag가 유의).
    # 정확한 주기 숫자는 아래 ACF 막대그래프를 직접 보고 판단할 것 - 가장
    # 왼쪽에서 뚜렷하게 튀어나온 lag가 유력 후보고, 그 배수는 조화성분이다.
    sig_lags = [lag for lag in range(1, max_lag + 1) if abs(acf[lag]) > sig_level]
    if len(sig_lags) >= max_lag // 2:
        print(f"  📎 유의 lag가 {len(sig_lags)}/{max_lag}개 - **유의미한 리듬이 있습니다.**")
        print("     정확한 주기는 아래 ACF 막대그래프에서 가장 왼쪽의 뚜렷한 "
              "봉우리를 확인할 것 (그 배수는 조화성분).")
    elif sig_lags:
        print(f"  📎 유의 lag가 일부 있음({sig_lags}) - 약한 상관은 있으나 "
              "뚜렷한 주기적 리듬이라고 보기엔 근거가 약합니다.")
    else:
        print(f"  📎 유의한 자기상관 없음(유의선 ±{sig_level:.3f} 안) - "
              "사이클끼리 대체로 독립적입니다. 눈에 보이던 리듬은 착시이거나,")
        print("     이 시계열보다 더 느린(수백ms~초 단위) 다른 원인일 가능성이 큽니다.")

    # [9차, GC 조사] mycobot_stream_exec.run_streaming_dispatch가 gc.callbacks로
    # 관찰한 "GC가 발동한 사이클 인덱스"가 로그에 있으면, 실제 튀는 사이클과
    # 얼마나 겹치는지를 정량으로 본다. "튐"은 평균+2*표준편차보다 큰 사이클로
    # 정의한다(임의 기준이지만, 이 데이터의 이중봉 분포에서 두 그룹을 가르는
    # 데는 충분히 보수적인 문턱이다).
    gc_cycle_indices = set(extra.get("gc_cycle_indices", []) or [])
    spike_threshold = cycles_ms.mean() + 2 * cycles_ms.std()
    spike_indices = list(np.where(cycles_ms > spike_threshold)[0].tolist())

    if gc_cycle_indices and spike_indices:
        def _match_count(gc_set, tol=1):
            return sum(1 for s in spike_indices if any(abs(s - g) <= tol for g in gc_set))

        actual_matched = _match_count(gc_cycle_indices)

        # [검증 중 발견] 처음엔 "겹치는 비율이 60% 넘으면 원인"이라는 고정
        # 문턱을 썼는데, 합성검증(GC=무관 시나리오)에서 44%가 나와 "애매함"
        # 구간에 잘못 걸렸다 - GC가 자주 발동할수록(밀도가 높을수록) 무작위로
        # 배치돼도 우연히 겹칠 확률 자체가 올라가기 때문에, 밀도를 무시한
        # 고정 비율은 오해를 부른다. 그래서 **"진짜 GC 발동 위치를 무작위 위치로
        # 바꿨다면 몇 개나 우연히 겹쳤을까"**를 500번 시뮬레이션(순열검정)해서
        # 그 우연 분포와 실제값을 비교한다(z-score). 밀도가 얼마든 공정하게 비교된다.
        rng = np.random.default_rng(0)
        n_gc = len(gc_cycle_indices)
        null_matches = np.array([
            _match_count(set(rng.choice(n, size=min(n_gc, n), replace=False).tolist()))
            for _ in range(500)
        ])
        null_mean, null_std = null_matches.mean(), null_matches.std()
        z = (actual_matched - null_mean) / max(null_std, 1e-9)

        print(f"\n  🔬 GC 상관 분석 (순열검정 500회): GC 발동 {n_gc}회, "
              f"튀는 사이클(>{spike_threshold:.1f}ms) {len(spike_indices)}개")
        print(f"     실제 겹침 {actual_matched}개  vs  우연 겹침(무작위 배치 시) "
              f"평균 {null_mean:.1f}±{null_std:.1f}개  →  z={z:+.1f}")
        if z >= 3:
            print("     → 우연이라고 보기 매우 어렵습니다. GC가 리듬의 주요 원인일 가능성이 높습니다.")
        elif z >= 1.5:
            print("     → 우연보다는 유의하게 많이 겹치지만, 결정적이라 하기엔 약합니다.")
        else:
            print("     → 우연 수준과 구분되지 않습니다. GC는 원인이 아닐 가능성이 높습니다.")
    else:
        gc_cycle_indices = None  # 로그에 없음(구버전 기록) 또는 튀는 사이클 자체가 없음

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    axes[0].plot(cycles_ms, ".-", markersize=3, linewidth=0.7, color="#2166ac")
    axes[0].axhline(cycles_ms.mean(), color="gray", linestyle=":", linewidth=0.8)
    if gc_cycle_indices:
        gc_idx_in_range = [g for g in gc_cycle_indices if 0 <= g < n]
        if gc_idx_in_range:
            axes[0].scatter(gc_idx_in_range, cycles_ms[gc_idx_in_range],
                             marker="x", s=70, color="green", zorder=5,
                             label=f"GC 발동 사이클 ({len(gc_idx_in_range)}회)")
            axes[0].legend(fontsize=8)
    axes[0].set_title(f"사이클 시계열 - {label} (send_speed={send_speed})")
    axes[0].set_xlabel("사이클 인덱스")
    axes[0].set_ylabel("사이클 시간 (ms)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(cycles_ms, bins=min(40, max(10, n // 5)), color="#5aa9c9", edgecolor="white")
    axes[1].set_title("사이클 시간 분포 (이중봉이면 '빠름/느림' 두 그룹이 실재)")
    axes[1].set_xlabel("사이클 시간 (ms)")
    axes[1].set_ylabel("빈도")
    axes[1].grid(alpha=0.3)

    lags = np.arange(max_lag + 1)
    axes[2].bar(lags, acf, color="#c44e52", width=0.6)
    axes[2].axhline(sig_level, color="gray", linestyle="--", linewidth=0.8)
    axes[2].axhline(-sig_level, color="gray", linestyle="--", linewidth=0.8)
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_xticks(lags)
    axes[2].set_title("자기상관함수(ACF) - 점선 밖 lag가 리듬 후보 (가장 왼쪽 뚜렷한 lag를 주기로 볼 것)")
    axes[2].set_xlabel("lag (사이클)")
    axes[2].set_ylabel("자기상관")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out_path = f"/tmp/cycle_rhythm_{label.replace(' ', '_').replace('(', '').replace(')', '')}_{ts.replace(':', '')}.png"
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  💾 그래프 저장: {out_path}")
    except Exception as e:
        print(f"  ⚠️ 그래프 저장 실패: {e}")
    fig.show()
    plt.pause(0.1)


def compare_recent(n_records):
    records = cycle_log.list_recent(limit=n_records)
    if not records:
        print("⚠️ 로그가 비어있습니다 - GUI로 스트리밍+'매번 실측' 모드로 곡선을 "
              "한 번 이상 실행해야 기록이 쌓입니다.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for rec in reversed(records):  # 오래된 것부터 그려서 범례가 시간순
        cycles_ms = np.array(rec.get("cycle_times_ms", []), dtype=float)
        if len(cycles_ms) < 5:
            continue
        send_speed = rec.get("send_speed")
        ts = rec.get("timestamp", "?")[:19]
        ax.plot(cycles_ms, alpha=0.7, linewidth=0.9,
                label=f"{ts}  send_speed={send_speed}  (평균 {cycles_ms.mean():.1f}ms)")
    ax.set_xlabel("사이클 인덱스")
    ax.set_ylabel("사이클 시간 (ms)")
    ax.set_title(f"최근 {len(records)}개 실행 비교 - STREAM_SEND_SPEED를 바꿔가며 "
                 "여러 번 실행했다면 여기서 리듬 변화가 보입니다")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "/tmp/cycle_rhythm_compare.png"
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"💾 비교 그래프 저장: {out_path}")
    except Exception as e:
        print(f"⚠️ 그래프 저장 실패: {e}")
    fig.show()
    plt.pause(0.1)

    print("\n=== 요약 ===")
    for rec in records:
        cycles_ms = np.array(rec.get("cycle_times_ms", []), dtype=float)
        if len(cycles_ms) < 5:
            continue
        acf = autocorrelation(cycles_ms, min(30, len(cycles_ms) // 3))
        sig_level = 1.96 / np.sqrt(len(cycles_ms))
        n_sig = sum(1 for a in acf[1:] if abs(a) > sig_level)
        print(f"  {rec.get('timestamp','?')[:19]}  send_speed={rec.get('send_speed')}  "
              f"평균={cycles_ms.mean():.1f}ms  표준편차={cycles_ms.std():.1f}ms  "
              f"유의lag수={n_sig}")


def main():
    while True:
        print("\n--- 메뉴 ---")
        choice = input(
            "  [1] 최신 기록 분석\n"
            "  [2] 최근 N개 기록 비교\n"
            "  [q] 종료\n"
            "> "
        ).strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            records = cycle_log.list_recent(limit=1)
            if not records:
                print("⚠️ 로그가 비어있습니다 - GUI로 스트리밍+'매번 실측' 모드로 "
                      "곡선을 한 번 이상 실행해야 기록이 쌓입니다.")
                continue
            analyze_one(records[0])
        elif choice == "2":
            raw = input("비교할 개수 (Enter=5): ").strip()
            n = int(raw) if raw else 5
            compare_recent(n)
        else:
            print("잘못된 입력입니다.")

    print("종료합니다.")


if __name__ == "__main__":
    main()
