# -*- coding: utf-8 -*-
"""
mycobot_tau_floor_summary.py
==============================
[9차 세션, §23.16-3/§26.9-4/§25.7-4, τ 하한 실험 지원 도구]

**실험 프로토콜은 이미 있었다** - `mycobot_stream_exec.py`의
`STREAM_TCP_SPEED_MMS`를 손으로 여러 값(예: 45,35,25,15)으로 바꿔가며
**같은 저장된 곡선**을 매번 '매번 실측' 모드로 실행하면 된다(§26.6에서
이미 이 상수를 손으로 바꾸는 방식을 쓴 전례가 있다). 이 실험 자체가
안 된 이유는 도구가 없어서가 아니라 **결과가 흩어져서 나중에 비교하기
귀찮았기 때문**으로 보인다 - `joint_lag.txt`는 실행마다 새 폴더
(`everytime_trace_N`)에 떨어지고, 어느 폴더가 어느 속도였는지는 실행
순서를 기억해야만 알 수 있었다.

이번 세션에 두 가지를 고쳤다:
  1) `mycobot_result_plots.py`가 `joint_lag.txt` 맨 위에 그 실행의 목표
     속도(`STREAM_TCP_SPEED_MMS`)를 한 줄 남기도록 함(순수 추가, 로직
     무변경) - 이제 파일 하나만 봐도 "이게 몇 mm/s였지"를 알 수 있다.
  2) 이 스크립트 - 결과 폴더 전체를 훑어 그 한 줄(목표속도)과 관절별
     "유효지연(ms)"을 모아 속도별로 정리한다. **로봇 연결도, 로그
     생성도 필요 없다** - 순수하게 이미 쌓인 `joint_lag.txt` 파일들을
     읽기만 한다.

[τ 하한을 읽는 법] 속도를 낮출 때마다 유효지연이 계속 줄어들면 아직
바닥이 아니고, 어느 지점부터 더 안 줄고 평평해지면 그 지점이 이
하드웨어의 진짜 하한이다(§23.16-3 원래 가설).

사용법:
    python3 mycobot_tau_floor_summary.py [결과루트경로]

    결과루트경로 생략 시 기본값 `~/바탕화면/curve_line_test`(§10 환경정보
    기준). 하위의 모든 `{버전}_figure/*/joint_lag.txt`를 재귀적으로 찾는다.

    [1] 관절별 속도-지연 표 (텍스트)
    [2] 그래프로 보기 (관절별 선 - 평평해지는 지점을 육안으로)
    [q] 종료

[주의] 서로 다른 곡선으로 실행한 결과가 섞이면 §14/§27의 교훈("혼합
데이터로 회귀하면 오염된다")이 그대로 적용된다 - 가급적 **같은 저장된
곡선**(curve_log)으로 속도만 바꿔 실행한 결과들만 모아서 비교할 것.
이 스크립트는 그 구분을 자동으로 못 한다(각 실행이 어떤 곡선이었는지
`joint_lag.txt`엔 안 남는다) - 실험할 때 결과 폴더를 확인해가며 직접
관리하거나, 실험 전후로 결과 폴더를 따로 비워두는 걸 권장한다.
"""

import os
import re
import sys
import glob

import numpy as np

DEFAULT_RESULTS_ROOT = os.path.expanduser("~/바탕화면/curve_line_test")

SPEED_RE = re.compile(r"\[목표 TCP 속도\]\s*([\d.]+)\s*mm/s")
WAYPOINT_RE = re.compile(r"\[웨이포인트\]\s*(\d+)개,\s*평균 간격\s*([\d.]+)mm")
# "   J1  |     0.123  |     0.456  |      210.3  |         45.2"
JOINT_ROW_RE = re.compile(
    r"^\s*J(\d)\s*\|\s*[\d.\-]+\s*\|\s*[\d.\-]+\s*\|\s*([\d.\-]+|nan)\s*\|"
)


def parse_joint_lag_file(path):
    """joint_lag.txt 하나를 읽어
    (목표속도mm/s 또는 None, 평균간격mm 또는 None, {관절idx: 유효지연ms}) 반환.
    목표속도/간격 줄이 없는(이번 세션 이전에 만들어진) 구버전 파일은 None으로
    반환한다 - 호출자가 '정보없음' 그룹으로 따로 묶는다."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None, None, {}

    speed = None
    m = SPEED_RE.search(text)
    if m:
        speed = float(m.group(1))

    step_mm = None
    m2 = WAYPOINT_RE.search(text)
    if m2:
        step_mm = float(m2.group(2))

    lags = {}
    for line in text.splitlines():
        m3 = JOINT_ROW_RE.match(line)
        if m3:
            j = int(m3.group(1))
            val = m3.group(2)
            if val != "nan":
                lags[j] = float(val)
    return speed, step_mm, lags


def scan(results_root):
    paths = glob.glob(os.path.join(results_root, "*_figure", "*", "joint_lag.txt"))
    records = []   # [(path, speed_or_None, step_mm_or_None, {joint: lag_ms}), ...]
    for p in sorted(paths):
        speed, step_mm, lags = parse_joint_lag_file(p)
        if lags:
            records.append((p, speed, step_mm, lags))
    return records


def print_table(records):
    known = [r for r in records if r[1] is not None]
    unknown = [r for r in records if r[1] is None]

    if not records:
        print("⚠️ joint_lag.txt를 하나도 못 찾았습니다 - 경로를 확인하거나, "
              "곡선을 '매번 실측' 모드로 최소 한 번 실행해보세요.")
        return

    print(f"\n총 {len(records)}건 발견 (목표속도 기록됨 {len(known)}건, "
          f"구버전(속도 미기록) {len(unknown)}건)")

    if known:
        # 속도별로 묶는다 (완전히 같은 값끼리)
        by_speed = {}
        for _, speed, step_mm, lags in known:
            by_speed.setdefault(speed, []).append((step_mm, lags))

        print(f"\n{'속도(mm/s)':<12}{'건수':<6}{'평균간격(mm)':<14}"
              + "".join(f"J{j:<9}" for j in range(1, 7)))
        for speed in sorted(by_speed):
            runs = by_speed[speed]
            steps = [s for s, _ in runs if s is not None]
            step_str = f"{np.mean(steps):<14.3f}" if steps else f"{'-':<14}"
            row = f"{speed:<12.0f}{len(runs):<6}{step_str}"
            for j in range(1, 7):
                vals = [lags[j] for _, lags in runs if j in lags]
                cell = f"{np.mean(vals):7.1f}  " if vals else f"{'-':>7}  "
                row += cell
            print(row)

        print("\n→ 속도를 낮출 때마다 유효지연이 계속 줄면 아직 바닥이 아님.")
        print("  어느 지점부터 더 안 줄고 평평해지면 그 지점이 이 하드웨어의 τ 하한.")
        print("  [주의] '평균간격'이 속도와 거의 비례해서 커지는 건 정상(§4.4) -")
        print("  다만 특정 속도에서 간격이 유독 넓은데 그 지점만 τ가 갑자기 튀면,")
        print("  순수 서보지연이 아니라 코너컷 효과(§4.3)가 섞였을 수 있으니")
        print("  의심해볼 것.")

    if unknown:
        print(f"\n⚠️ 속도 미기록 {len(unknown)}건은 표에서 제외했습니다 "
              "(이번 세션 이전 실행 - mycobot_result_plots.py 갱신 전이라 "
              "목표속도가 안 남아있음). 필요하면 파일명/생성시각으로 직접 대조하세요:")
        for p, _, _, _ in unknown[:10]:
            print(f"   {p}")
        if len(unknown) > 10:
            print(f"   ... 외 {len(unknown)-10}건")


def plot(records):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    known = [r for r in records if r[1] is not None]
    if not known:
        print("⚠️ 목표속도가 기록된 결과가 없어 그래프를 못 그립니다.")
        return

    by_speed = {}
    for _, speed, _step_mm, lags in known:
        by_speed.setdefault(speed, []).append(lags)
    speeds = sorted(by_speed)

    fig, ax = plt.subplots(figsize=(9, 6))
    for j in range(1, 7):
        ys, xs = [], []
        for speed in speeds:
            vals = [r[j] for r in by_speed[speed] if j in r]
            if vals:
                xs.append(speed)
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, "o-", label=f"J{j}")
    ax.set_xlabel("목표 TCP 속도 (mm/s)")
    ax.set_ylabel("유효지연 (ms)")
    ax.set_title("속도별 관절 유효지연 - 평평해지는 지점이 τ 하한 후보")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "/tmp/tau_floor_summary.png"
    try:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"💾 그래프 저장: {out}")
    except Exception as e:
        print(f"⚠️ 그래프 저장 실패: {e}")
    fig.show()
    plt.pause(0.1)


def main():
    results_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_ROOT
    print(f"결과 루트: {results_root}")
    if not os.path.isdir(results_root):
        print("⚠️ 경로가 없습니다 - 인자로 실제 경로를 넘겨주세요.")
        return

    records = scan(results_root)

    while True:
        print("\n--- 메뉴 ---")
        choice = input("  [1] 속도-지연 표\n  [2] 그래프\n  [q] 종료\n> ").strip().lower()
        if choice == "q":
            break
        elif choice == "1":
            print_table(records)
        elif choice == "2":
            plot(records)
        else:
            print("잘못된 입력입니다.")

    print("종료합니다.")


if __name__ == "__main__":
    main()
