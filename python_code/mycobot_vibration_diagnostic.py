# -*- coding: utf-8 -*-
"""
mycobot_vibration_diagnostic.py
================================
[8차 세션, 진단용 1회성 스크립트] 정렬자세(혹은 임의 자세)에서 정지해
있는데 로봇이 떠는 증상을 조사한다. mycobot_latency_diagnostic.py /
mycobot_frame_verify.py와 같은 위치의 "GUI 없는 1회 실행용 스크립트".

목적: "떨림"이 사람 눈에는 애매해도, get_angles()를 촘촘히 찍으면
"어느 관절의 명령값이 실제로 흔들리는가"를 숫자로 잡을 수 있다.
관절오프셋보정(joint_offset_correction)은 이미 A/B로 꺼봐도 동일하다고
확인됐으므로(우리 쪽 코드가 재전송하는 루프가 아니라는 정황 증거), 이
스크립트는 순수하게 "로봇이 정지해 있는 동안 get_angles()가 보고하는
값 자체가 흔들리는가"만 잰다 - 서보/기구 레벨 문제인지 좁히는 용도.

사용법:
    python3 mycobot_vibration_diagnostic.py

    1) 로봇을 원하는 자세로 보낸다 - 정렬자세[1], 현재 자세 그대로[2],
       또는 J3 저부하 비교자세[3](§23.4, 정렬자세에서 J3만 0으로 낮춤).
    2) SAMPLE_SEC 동안 get_angles()를 최대한 빠르게 반복 호출해 시계열을
       모은다 (§3에 따르면 읽기 자체가 20~40ms/회 걸리므로 그게 실질
       샘플링 주기다 - 그보다 빠른 진동은 이 방법으로는 못 잡는다).
    3) 관절별 평균/표준편차/최대진폭(peak-to-peak)을 표로 찍고,
       그래프로 저장한다. 반복 측정 가능(Enter로 재측정) - "정렬자세"
       "곡선 실행 직후" "J3 저부하" 등 여러 상황을 이 스크립트 하나로
       비교할 수 있다.

    [주의, §23.4에서 실측 확인] 손으로 팔을 받쳐서 부하를 줄이는 방식은
    사람 손의 미세한 흔들림이 섞여 들어가(J1 진동이 오히려 심해짐)
    교란변수가 됐다. **손을 대지 않고** 자세 자체를 바꿔서(옵션 [3])
    비교하는 걸 권장한다.

    [8차 세션, §23.15 갱신] 측정할 때마다 실제 관절각(평균)을 공유 모듈
    `mycobot_pose_log.py`를 통해 `vibration_pose_log.jsonl`(스크립트와
    같은 폴더)에서 **조회만** 한다 - 이 스크립트는 로그에 아무것도 안
    쓴다. 이 로그는 `mycobot_path_editor.py`가 곡선 실행 직후마다 남기는
    기록이 유일한 출처다 - 즉 GUI로 실제 작업하다 "어, 이거 떠는데"
    싶었던 자세가 이미 로그에 남아있을 수 있다. 새 측정을 할 때마다
    로그에서 "관절별 오차가 전부 3도 이내인" 과거 기록을 자동으로
    찾아서 보여주기만 한다. (예전엔 이 스크립트도 자기 측정을 로그에
    같이 남겼는데, `[4]`로 로그 자세를 불러와 재측정하면 그 결과가 또
    쌓이고 그걸 또 불러와 재는 식으로 로그가 지저분해져서 뺐다.)

해석 가이드:
    - 표준편차가 로봇의 각도 분해능(~0.1도) 근처면: 그냥 읽기 노이즈일
      뿐 진짜 떨림이 아닐 수 있다.
    - 표준편차가 그보다 뚜렷이 크고(예: 0.3도 이상) 그래프가 규칙적인
      진동(사인파 비슷)으로 보이면: 서보 홀딩토크 헌팅 가능성.
    - 그래프가 계단형으로 두 값 사이를 왔다갔다 하면: 백래시 dither
      가능성 (§3 백래시 항목과 같은 메커니즘).
    - 여러 관절이 동시에 같은 타이밍에 튀면: 전원(브라운아웃) 의심.
    - 한두 관절만 독립적으로 떨면: 그 관절 서보/기어박스 개별 문제.
"""

import sys
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 로컬 환경에 맞게 필요시 바꿀 것
import matplotlib.pyplot as plt

from mycobot_kinematics import (
    find_robot_port, ALIGN_ANGLES, SPEED, SETTLE_DELAY_SEC,
)
import mycobot_pose_log as pose_log

SAMPLE_SEC = 8.0          # 1회 측정 길이 (초)
ANGLE_RESOLUTION_DEG = 0.1  # 로봇 각도 분해능 참고값(§3) - 판정 기준선으로 표시

# ---------------------------------------------------------------------------
# [8차 세션 추가] J3 저부하 비교자세
# ---------------------------------------------------------------------------
# 정렬자세(ALIGN_ANGLES=[0,-30,30,0,30,0])에서 J3만 0으로 낮췄다 - 나머지
# 관절은 그대로 둬서 "J3 하나만 바뀐" 효과를 깨끗하게 비교하기 위함이다.
# 손으로 팔을 받쳐서 부하를 줄이는 시도는 사람 손의 미세한 흔들림이 섞여
# 들어가(§23.4 참고 - 실측으로 확인됨, J1 진동이 오히려 심해짐) 교란변수가
# 됐다 - 이 자세는 사람이 손을 대지 않고도 J3의 중력부하 자체를 줄이는
# 시도다.
# [주의] 이 값이 실제로 J3의 중력모멘트를 충분히 줄이는지는 로봇의 정확한
# 기구학(DH/URDF) 세부사항에 달려 있어 이론적으로 100% 보장하지 못한다.
# 정렬자세보다 J3의 std/P2P가 뚜렷이 줄면 중력부하 가설 확인, 안 줄면
# 이 자세 자체(각도 조합)를 다음 세션에서 조정해야 할 수도 있다.
LOW_LOAD_TEST_ANGLES = [0, -30, 0, 0, 30, 0]


def sample_vibration(mc, duration_sec):
    """get_angles()를 duration_sec 동안 최대한 빠르게 반복 호출."""
    times, samples = [], []
    t0 = time.time()
    while time.time() - t0 < duration_sec:
        a = mc.get_angles()
        if isinstance(a, list) and len(a) == 6:
            samples.append(a)
            times.append(time.time() - t0)
    return np.array(times), np.array(samples, dtype=float)


def analyze_and_report(times, samples, label):
    if len(samples) < 5:
        print(f"⚠️ [{label}] 샘플이 너무 적습니다({len(samples)}개) - 측정 실패로 보고 재시도하세요.")
        return

    dt_mean = np.mean(np.diff(times)) * 1000 if len(times) > 1 else float("nan")
    print(f"\n=== [{label}] 관절별 정지오차/떨림 분석 (샘플 {len(samples)}개, 평균주기 {dt_mean:.1f}ms) ===")
    print("  관절 |  평균(도) | 표준편차(도) | 최대진폭 P2P(도) | 판정")
    ptp_list = []
    for j in range(6):
        col = samples[:, j]
        mean_v = col.mean()
        std_v = col.std()
        ptp = col.max() - col.min()
        ptp_list.append(ptp)
        if ptp <= ANGLE_RESOLUTION_DEG * 1.5:
            verdict = "정상범위(읽기노이즈 수준)"
        elif ptp <= 0.5:
            verdict = "경미한 흔들림 - 관찰 필요"
        else:
            verdict = "⚠️ 뚜렷한 떨림"
        print(f"   J{j+1}  | {mean_v:9.3f} | {std_v:12.3f} | {ptp:16.3f} | {verdict}")

    worst = int(np.argmax(ptp_list))
    print(f"  → 가장 크게 흔들리는 관절: J{worst+1} (P2P {ptp_list[worst]:.3f}도)")

    # [8차 세션, §23.15] 이 스크립트는 공유 자세 로그에 더 이상 기록하지
    # 않는다 - 조회(비슷한 과거 자세 찾기)만 한다. 원래는 log_pose()로
    # 기록도 같이 했는데, 그러면 mycobot_path_editor.py가 곡선 실행마다
    # 남기는 "진짜 실행 기록"과 이 스크립트가 "그냥 확인차 잰 것"이
    # 섞여서 로그가 지저분해졌다(예: from_log:curve_stop 자세를 불러와
    # 재보면 그 결과가 또 로그에 쌓이고, 그걸 또 불러와 재면 또 쌓이는
    # 식). 기록은 path_editor.py만 담당한다.
    joint_means = [float(samples[:, j].mean()) for j in range(6)]
    similar = pose_log.find_similar_poses(joint_means)
    if similar:
        print(f"📎 자세 로그: 비슷한 과거 자세 {len(similar)}건 발견"
              f" (관절별 오차 {pose_log.SIMILAR_POSE_TOLERANCE_DEG}도 이내, 조회만 - 이 측정은 기록 안 함):")
        for max_diff, rec in similar[:5]:
            extra_note = ""
            if "worst_joint" in rec:
                extra_note = (f" | 그때 최대흔들림: J{rec['worst_joint']}"
                              f" P2P {rec['worst_ptp']:.3f}도")
            print(f"   - {rec['timestamp']} [{rec['label']}]"
                  f" 최대관절오차 {max_diff:.2f}도{extra_note}")
    else:
        print("📎 자세 로그: 비슷한 과거 자세 없음 (이 측정은 로그에 기록되지 않습니다)")

    # 그래프: 6개 관절 시계열을 한 화면에
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for j in range(6):
        ax = axes[j // 3, j % 3]
        ax.plot(times, samples[:, j], ".-", markersize=3, linewidth=0.8)
        ax.axhline(samples[:, j].mean(), color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"Joint {j+1}  (P2P {ptp_list[j]:.3f}°, std {samples[:, j].std():.3f}°)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("angle (deg)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Vibration diagnostic - {label}")
    fig.tight_layout()
    out_path = f"/tmp/vibration_{label.replace(' ', '_')}_{int(time.time())}.png"
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"💾 그래프 저장: {out_path}")
    except Exception as e:
        print(f"⚠️ 그래프 저장 실패: {e}")
    fig.show()
    plt.pause(0.1)


def _pick_pose_from_log():
    """로그에서 최신 기록(최대 pose_log.DEFAULT_LIST_LIMIT개)을 보여주고
    하나를 고르게 한다. 선택된 record 또는 None."""
    records = pose_log.load_log()
    if not records:
        print("⚠️ 로그가 비어있습니다 - 아직 기록된 자세가 없습니다.")
        return None

    # 최신순 정렬 (타임스탬프 문자열은 ISO 형식이라 문자열 비교로도 시간순 정렬됨)
    records_sorted = sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)
    top_n = records_sorted[:pose_log.DEFAULT_LIST_LIMIT]

    print(f"\n=== 최근 자세 로그 (최신 {len(top_n)}개) ===")
    for i, rec in enumerate(top_n, start=1):
        jm = rec.get("joint_means", [])
        jm_str = ", ".join(f"{v:.1f}" for v in jm) if jm else "?"
        note = ""
        if "worst_joint" in rec:
            note = f"  [측정기록: J{rec['worst_joint']} P2P {rec['worst_ptp']:.3f}도]"
        print(f"  [{i}] {rec.get('timestamp', '?')}  [{rec.get('label', '?')}]"
              f"  ({jm_str}){note}")

    choice = input("불러올 번호를 선택하세요 (취소하려면 Enter): ").strip()
    if not choice:
        print("취소했습니다.")
        return None
    try:
        idx = int(choice)
    except ValueError:
        print("⚠️ 잘못된 입력입니다.")
        return None
    if not (1 <= idx <= len(top_n)):
        print("⚠️ 범위 밖 번호입니다.")
        return None
    return top_n[idx - 1]


def main():
    print("🔌 로봇 포트 탐색 중...")
    mc, port = find_robot_port()
    if mc is None:
        print("❌ 로봇을 찾지 못했습니다. USB 연결/권한을 확인하세요.")
        sys.exit(1)
    print(f"✅ 연결됨: {port}")

    while True:
        print("\n--- 새 측정 ---")
        mode = input(
            "측정할 상황을 고르세요:\n"
            "  [1] 정렬자세(ALIGN_ANGLES)로 이동 후 측정\n"
            "  [2] 현재 자세 그대로 측정 (곡선 실행 직후 등, 로봇을 먼저 원하는 상태로 만들어두세요)\n"
            "  [3] J3 저부하 비교자세로 이동 후 측정 (정렬자세에서 J3만 0으로 낮춤 - §23.4)\n"
            "  [4] 로그에서 자세 불러오기 (최근 5개 중 선택 후 그 자세로 이동)\n"
            "  [q] 종료\n"
            "> "
        ).strip().lower()

        if mode == "q":
            break
        elif mode == "1":
            print(f"정렬자세로 이동 중... {ALIGN_ANGLES}")
            mc.send_angles(ALIGN_ANGLES, SPEED)
            time.sleep(SETTLE_DELAY_SEC)
            t0 = time.time()
            while mc.is_moving():
                if time.time() - t0 > 30:
                    print("⚠️ 이동 완료 대기 타임아웃 - 그래도 측정은 진행합니다.")
                    break
            label = "align_pose"
        elif mode == "2":
            label = input("이 측정에 붙일 라벨을 입력하세요 (예: after_curve_stop): ").strip() or "current_pose"
        elif mode == "3":
            print(f"J3 저부하 비교자세로 이동 중... {LOW_LOAD_TEST_ANGLES}")
            mc.send_angles(LOW_LOAD_TEST_ANGLES, SPEED)
            time.sleep(SETTLE_DELAY_SEC)
            t0 = time.time()
            while mc.is_moving():
                if time.time() - t0 > 30:
                    print("⚠️ 이동 완료 대기 타임아웃 - 그래도 측정은 진행합니다.")
                    break
            label = "j3_low_load_pose"
        elif mode == "4":
            rec = _pick_pose_from_log()
            if rec is None:
                continue
            target = rec["joint_means"]
            print(f"로그된 자세로 이동 중... {[round(v, 1) for v in target]}")
            mc.send_angles(target, SPEED)
            time.sleep(SETTLE_DELAY_SEC)
            t0 = time.time()
            while mc.is_moving():
                if time.time() - t0 > 30:
                    print("⚠️ 이동 완료 대기 타임아웃 - 그래도 측정은 진행합니다.")
                    break
            label = f"from_log:{rec.get('label', '?')}"
        else:
            print("잘못된 입력입니다.")
            continue

        print(f"{SAMPLE_SEC}초 동안 측정합니다. 로봇을 건드리지 마세요...")
        times, samples = sample_vibration(mc, SAMPLE_SEC)
        analyze_and_report(times, samples, label)

    print("종료합니다.")


if __name__ == "__main__":
    main()
