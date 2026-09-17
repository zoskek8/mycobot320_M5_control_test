# -*- coding: utf-8 -*-
"""
mycobot_ab_control_adjust.py
==============================
[10차 세션, §41] §34.5~§36에서 5차례 A/B를 했는데도 J1/J4/J5/J6 피드포워드의
실제 효과를 확정하지 못했다 - **대조군(J2/J3, 피드포워드가 아예 안 걸리는
관절)까지 매번 "FF ON 라벨"을 따라 같은 방향으로 움직였다**(§35.1/§36.1).
순서를 뒤집어도(§35.1의 3차) 이 패턴은 안 깨졌다 - 즉 순서효과가 아니라
**세션/실행 단위로 6관절 전부에 비슷한 크기로 걸리는 정체불명의 공통
교란**이 있다는 뜻이다.

[이 스크립트의 아이디어] 원인을 못 밝혀도 **대조군 자체를 잣대로 쓰면
공통 교란을 지울 수 있다** - 통계학의 "차분의 차분"(difference-in-
differences)과 같은 원리다. 매 실행마다:

    조정값(Jx) = 원시_유효지연(Jx) - 그 실행의 대조군 평균(J2,J3)

으로 바꾸면, 그 실행에 공통으로 걸린 교란(원인이 뭐든）이 J2/J3에도
Jx에도 똑같이 실려 있었다는 전제 하에 상쇄된다. §34.5~§36 데이터를 보면
이 전제가 대체로 맞아 보인다(교란의 크기가 관절마다 비슷한 자릿수였다).
**전제가 틀렸을 가능성도 있으므로(교란이 관절마다 다른 크기일 수 있음),
이 조정이 만능은 아니다** - 그래서 원시값과 조정값을 항상 나란히 보여주고
스크립트가 자동으로 "확정"을 선언하지 않는다.

[반복 + 무작위 순서] 조정만으론 부족하다 - 매 실행 자체의 노이즈(로봇
상태·환경)가 여전히 남는다. 그래서 조건당 3회씩, 무작위 순서로 반복한다.

[10차, §43 갱신 - 두 번째 실험에도 재사용] 이 스크립트는 관절 역할을
파라미터로 받는다(`control_joints`/`treatment_joints`) - 실험마다 어느
관절이 "절대 안 건드려지는 대조군"이고 어느 게 "판정 대상"인지가 다르기
때문이다:
  - §41(오차모델 FF, J1/J5/J6에 적용): 대조군=(2,3), 판정대상=(1,4,5,6)
  - §43(자세지연보정, J2/J3/J4에 적용): 대조군=(1,5,6), 판정대상=(2,3,4)
아래 `RUN_ORDER`/`RUN_DATA`/`main()`은 **현재 진행 중인 실험**(§43, 자세
지연보정) 기준으로 맞춰져 있다 - 시드 20260944(=20260901+43)로 새로 뽑은
순서:

    실행1=ON  실행2=OFF  실행3=ON  실행4=OFF  실행5=ON  실행6=OFF

곡선/속도는 §41과 동일하게 유지한다(`curve_e433b126`, 35mm/s) - 바꾸면
그 자체가 새로운 교란변수가 된다. **다만 이번엔 GUI의 "오차모델 FF"
체크박스는 그대로 ON에 둔 채(§42에서 이미 배포됨), "자세지연보정"
체크박스만 토글한다** - 두 기능은 서로 다른 관절에 적용되므로 독립적으로
검증 가능하다.

[사용법] 6회 실행 후, 각 실행의 joint_lag.txt 원문을 그대로 `RUN_DATA`에
붙여넣고 이 스크립트를 실행하면 된다. 로봇 연결이 필요 없다 - 순수하게
텍스트를 분석한다.

    python3 mycobot_ab_control_adjust.py
"""

import re
import sys
from collections import defaultdict

import numpy as np

# joint_lag.txt의 "J1 | 평균오차 | 최대오차 | 유효지연 | 각속도범위" 행 파서.
# mycobot_tau_floor_summary.py의 JOINT_ROW_RE와 같은 열 구조를 쓰되, [10차,
# §51 준비] 세 숫자 컬럼을 전부 캡처해 metric으로 고를 수 있게 했다(기존엔
# 유효지연 하나만 뽑았음 - §49처럼 평균오차가 필요한 실험도 있어서).
JOINT_ROW_RE = re.compile(
    r"^\s*J(\d)\s*\|\s*([\d.\-]+|nan)\s*\|\s*([\d.\-]+|nan)\s*\|\s*([\d.\-]+|nan)\s*\|"
)
# metric 이름 -> JOINT_ROW_RE 캡처그룹 번호. 열 순서(평균오차,최대오차,유효지연)와 동일.
_METRIC_GROUP = {"mean_err": 2, "max_err": 3, "lag": 4}

# [10차, §51 준비] J3 중력처짐 모델(§50) 실기 A/B - 곡선마다 따로 6회씩 돌려서
# 이 파일 상단을 곡선별로 갈아끼우며 분석한다(§41~49와 같은 방식 - 곡선을
# 합쳐서 한 번에 분석하지 않는다. §49의 실수가 정확히 그거였다).
# 곡선별 순서(균형블록 무작위, ON 3+OFF 3):
#   e433b126(35mm/s): 시드 20260951 -> OFF,ON,ON,OFF,OFF,ON
#   e651e55d(35mm/s): 시드 20260952 -> ON,OFF,ON,ON,OFF,OFF
#   78dfad65(35mm/s): 시드 20260953 -> OFF,ON,OFF,ON,ON,OFF
# 아래는 첫 곡선(e433b126) 기준 - 다른 곡선으로 넘어갈 땐 RUN_ORDER/RUN_DATA를
# 위 표대로 바꿔서 다시 채운다.
RUN_ORDER = ["OFF", "ON", "ON", "OFF", "OFF", "ON"]   # e433b126, 시드 20260951

# [여기에 실행 결과를 채워 넣는다] 6개 원소, RUN_ORDER와 같은 순서.
# 각 원소는 joint_lag.txt 원문 전체(복붙) 또는 None(아직 안 돌림).
# ON/OFF는 mycobot_kinematics.py의 J3_CORRECTION_MODE를 "gravity"(ON) /
# "static"(OFF, §48 이전 정지모델)로 바꾼 것을 뜻한다. 예측 검증(§51 3~4단계)
# 에서는 "none"을 쓴다 - 모델이 예측하는 편향은 "보정이 없을 때의 처짐"이라
# 무보정 상태를 재야 예측과 실측을 직접 대조할 수 있다.
# 나머지(관절오프셋보정 ON, 자세지연보정 ON, 오차모델
# FF J1/J5/J6)는 전부 §49와 동일하게 고정.
# **주의**: status.txt에는 이 플래그 상태가 안 찍힌다 - 매 실행 전/후로
# 소스의 플래그 값을 직접 확인해서 라벨이 안 헷갈렸는지 검증할 것.
RUN_DATA = [
    None,   # 실행1 = OFF
    None,   # 실행2 = ON
    None,   # 실행3 = ON
    None,   # 실행4 = OFF
    None,   # 실행5 = OFF
    None,   # 실행6 = ON
]


def parse_joint_lag(text, metric="lag"):
    """joint_lag.txt 원문에서 {관절idx(1~6): 값} 를 뽑는다.

    metric: "lag"(유효지연,ms, 기본값 - 기존 호출부 그대로 호환) /
    "mean_err"(평균오차,도) / "max_err"(최대오차,도)."""
    group_idx = _METRIC_GROUP[metric]
    lags = {}
    for line in text.splitlines():
        m = JOINT_ROW_RE.match(line)
        if m:
            val = m.group(group_idx)
            if val != "nan":
                lags[int(m.group(1))] = float(val)
    return lags


def control_baseline(lags, control_joints):
    """그 실행의 대조군(control_joints) 평균 - 공통 교란의 추정치."""
    vals = [lags[j] for j in control_joints if j in lags]
    if not vals:
        return None
    return float(np.mean(vals))


def analyze(run_order, run_texts, control_joints=(2, 3), treatment_joints=(1, 4, 5, 6),
            metric="lag", unit="ms"):
    """조건별로 원시/조정 값을 모아 관절별 비교표를 만든다.

    control_joints/treatment_joints: [10차, §43에서 일반화] 실험마다 어느
    관절이 "절대 안 건드려지는 대조군"이고 어느 관절이 "이번에 판정할
    대상"인지가 다르다 - §41(오차모델 FF)은 대조군=J2/J3, 판정대상=
    J1/J4/J5/J6였고, §43(자세지연보정)은 반대로 대조군=J1/J5/J6, 판정대상=
    J2/J3/J4다. 기본값은 §41 그대로 둬서 기존 호출부(자체점검 포함)가
    안 깨지게 했다.

    metric/unit: [10차, §51 준비] 어느 컬럼을 볼지("lag"/"mean_err"/
    "max_err")와 출력용 단위 표시. 기본값은 §41~49 그대로(유효지연,ms).
    §51(J3 중력처짐 모델 검증)처럼 평균오차(도)가 필요하면
    metric="mean_err", unit="도"로 호출한다.

    로봇 데이터 없이도(합성 텍스트) 동작 - 자체 점검에서 이 성질을 쓴다."""
    parsed = []
    for cond, text in zip(run_order, run_texts):
        if text is None:
            continue
        lags = parse_joint_lag(text, metric=metric)
        if len(lags) < 6:
            print(f"  ⚠️ 관절 6개를 다 못 찾았습니다(찾은 것: {sorted(lags)}) - 이 실행은 건너뜁니다.")
            continue
        base = control_baseline(lags, control_joints)
        parsed.append({"cond": cond, "lags": lags, "base": base})

    if not parsed:
        return None

    n_on = sum(1 for r in parsed if r["cond"] == "ON")
    n_off = sum(1 for r in parsed if r["cond"] == "OFF")
    print(f"\n분석 대상: ON {n_on}회 / OFF {n_off}회 (총 {len(parsed)}회 입력됨)")
    if n_on == 0 or n_off == 0:
        print("  ⚠️ 한쪽 조건이 0회입니다 - 비교할 수 없습니다.")
        return None

    # 대조군 자체의 잔차 확인 - 조정이 대조군 스스로에게도 말이 되는지.
    # (base가 대조군 평균이므로 조정된 대조군은 서로 반대부호로 작지만 0은
    #  아니다 - 그 잔차 크기가 곧 "대조군 관절들 사이의 불일치", 즉
    #  조정법이 못 지우는 잔여 노이즈의 하한이다.)
    for cj in control_joints:
        resid = [r["lags"][cj] - r["base"] for r in parsed]
        print(f"대조군 자체 잔차(조정 후에도 남는 것) - J{cj}: {np.std(resid):.2f}{unit} "
              f"(조정법의 잔여 노이즈 하한 추정)")

    print(f"\n{'='*88}")
    print(f"  관절별 비교 - 원시(raw) vs 대조군 조정(adjusted)")
    print(f"{'='*88}")
    header = f"{'관절':<6}{'raw ON':>9}{'raw OFF':>9}{'raw 차이':>10}   " \
             f"{'adj ON':>9}{'adj OFF':>9}{'adj 차이':>10}"
    print(header)

    joint_names = {j: f"J{j}" for j in treatment_joints}
    joint_names.update({j: f"J{j}(대조)" for j in control_joints})
    all_joints = list(treatment_joints) + list(control_joints)
    verdicts = {}
    for j in all_joints:
        raw_on = [r["lags"][j] for r in parsed if r["cond"] == "ON"]
        raw_off = [r["lags"][j] for r in parsed if r["cond"] == "OFF"]
        adj_on = [r["lags"][j] - r["base"] for r in parsed if r["cond"] == "ON"]
        adj_off = [r["lags"][j] - r["base"] for r in parsed if r["cond"] == "OFF"]

        raw_diff = np.mean(raw_on) - np.mean(raw_off)
        adj_diff = np.mean(adj_on) - np.mean(adj_off)
        print(f"{joint_names[j]:<6}{np.mean(raw_on):>9.1f}{np.mean(raw_off):>9.1f}"
              f"{raw_diff:>+10.1f}   {np.mean(adj_on):>9.1f}{np.mean(adj_off):>9.1f}"
              f"{adj_diff:>+10.1f}")
        verdicts[j] = {"raw_diff": raw_diff, "adj_diff": adj_diff,
                       "adj_on_spread": np.std(adj_on) if len(adj_on) > 1 else None,
                       "adj_off_spread": np.std(adj_off) if len(adj_off) > 1 else None}

    print(f"\n  (음수 = ON일 때 {unit} 값이 더 작음 = ON이 유리한 방향)")

    treat_names = ",".join(f"J{j}" for j in treatment_joints)
    ctrl_names = ",".join(f"J{j}" for j in control_joints)
    print(f"\n{'='*88}")
    print(f"  해석 - {treat_names} ({ctrl_names}는 그 자체가 대조군이라 판정 대상이 아님)")
    print(f"{'='*88}")
    for j in treatment_joints:
        v = verdicts[j]
        spread = None
        if v["adj_on_spread"] is not None and v["adj_off_spread"] is not None:
            spread = max(v["adj_on_spread"], v["adj_off_spread"])
        note = ""
        if spread is not None and spread > 0:
            ratio = abs(v["adj_diff"]) / spread
            if ratio < 1.0:
                note = f"  (조정 차이가 반복 간 흩어짐보다 작음 - 근거 약함, ratio={ratio:.1f})"
            else:
                note = f"  (조정 차이가 반복 간 흩어짐보다 큼 - ratio={ratio:.1f})"
        direction = "ON이 유리(개선)" if v["adj_diff"] < 0 else "ON이 불리(악화)"
        print(f"  {joint_names[j]}: 조정 차이 {v['adj_diff']:+.1f}{unit} → {direction}{note}")

    print("\n  ⚠️ n=3 대 3은 통계적 유의성을 논할 표본이 아니다 - 여기서는 방향과")
    print("     대략적 크기만 본다. 그래도 raw와 adj가 같은 방향이면 신뢰도가")
    print("     올라가고, raw에서 봤던 차이가 adj에서 사라지면 그건 원래")
    print("     대조군과 함께 움직이던 공통교란이었다는 뜻이다(§35~36의 패턴).")

    return verdicts


def main():
    if all(t is None for t in RUN_DATA):
        print("⚠️ RUN_DATA가 전부 비어 있습니다 - 6회 실행 후 joint_lag.txt 원문을")
        print("   이 파일 상단의 RUN_DATA 리스트에 순서대로 붙여넣고 다시 실행하세요.")
        print(f"\n   실행 순서(무작위, §51): {RUN_ORDER}")
        sys.exit(0)

    filled = sum(1 for t in RUN_DATA if t is not None)
    print(f"입력된 실행: {filled}/6")
    if filled < 6:
        print("  (전부 안 채워져도 있는 것만으로 우선 분석합니다.)")

    # [§51] J3 중력처짐 모델은 J3 계수만 바꾸므로 판정대상=(3,), 나머지 5관절이
    # 대조군. §49와 같은 이유로 지표는 평균오차(도) - 이건 상수/모델 편향을
    # 고치는 보정이라 유효지연(ms)은 적절한 지표가 아니다.
    analyze(RUN_ORDER, RUN_DATA, control_joints=(1, 2, 4, 5, 6), treatment_joints=(3,),
            metric="mean_err", unit="도")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        # 로봇/실제 데이터 없이 로직만 검증 - 참 효과를 심어둔 합성 데이터로
        # 이 스크립트가 그 효과를 올바르게 복원하는지 확인한다.
        print("자체 점검 시작...\n")
        rng = np.random.default_rng(0)

        def make_run(cond, common_offset, true_effects, noise=0.3, vary="lag"):
            """common_offset: 그 실행에 6관절 전부 공통으로 실리는 교란(§35~36
            재현). true_effects: {joint: 값} - 그 조건이 실제로 그 관절에 주는
            효과(ON일 때만 적용). vary: 세 숫자 컬럼(mean_err/max_err/lag) 중
            어느 것에 값을 심을지 - [10차, §51] 평균오차 지표 검증에 재사용."""
            base_val = {1: 120.0, 2: 145.0, 3: 135.0, 4: 125.0, 5: 90.0, 6: 90.0}
            lines = ["[목표 TCP 속도] 35mm/s", "[웨이포인트] 531개, 평균 간격 1.679mm",
                     "  관절 | 평균오차(도) | 최대오차(도) | 유효지연(ms) | 각속도범위(도/s)"]
            cols = {"mean_err": "1.000", "max_err": "3.000", "lag": "135.0"}
            for j in range(1, 7):
                val = base_val[j] + common_offset + rng.normal(0, noise)
                if cond == "ON":
                    val += true_effects.get(j, 0.0)
                cols[vary] = f"{val:.1f}"
                lines.append(f"   J{j}  |       {cols['mean_err']}  |       {cols['max_err']}"
                             f"  |       {cols['lag']}  |           20.0")
            return "\n".join(lines)

        # 참 효과: J1=-10ms(FF가 실제로 도움), J4=0(효과 없음), J5=-5, J6=0
        TRUE = {1: -10.0, 4: 0.0, 5: -5.0, 6: 0.0}
        order = ["OFF", "OFF", "ON", "ON", "ON", "OFF"]
        # 공통교란을 조건에 강하게 실어서(§35~36처럼) raw만 보면 다 좋아 보이게 만든다
        common = {"OFF": 8.0, "ON": -8.0}
        texts = [make_run(c, common[c], TRUE) for c in order]

        print("--- raw로만 보면(조정 전) J4/J6도 좋아 보여야 정상(공통교란 때문) ---")
        v = analyze(order, texts)

        assert v[1]["adj_diff"] < -5, f"J1 참효과(-10) 복원 실패: {v[1]['adj_diff']}"
        assert abs(v[4]["adj_diff"]) < 5, f"J4 참효과(0) 오판: {v[4]['adj_diff']}"
        assert v[5]["adj_diff"] < -1, f"J5 참효과(-5) 복원 실패: {v[5]['adj_diff']}"
        assert abs(v[6]["adj_diff"]) < 5, f"J6 참효과(0) 오판: {v[6]['adj_diff']}"
        print("\n✅ 합성 데이터에서 조정 후 J1(-10)/J5(-5)는 검출, J4/J6(0)은 정확히")
        print("   '효과없음'으로 남았습니다 - raw만 봤다면 공통교란 때문에 넷 다")
        print("   좋아 보였을 상황인데, 조정이 그 함정을 피했습니다.")

        # [10차, §43] 역할을 뒤집은 경우(대조군=J1/J5/J6, 판정대상=J2/J3/J4 -
        # 자세지연보정 실험) 파라미터가 실제로 반영되는지 별도 검증. 참효과를
        # J2/J3/J4쪽에 심고 대조군은 J1/J5/J6로 바꿔서 같은 로직을 재사용.
        print("\n--- 역할반전 검증(§43, 대조군=J1/J5/J6, 판정대상=J2/J3/J4) ---")
        TRUE2 = {2: -8.0, 3: 0.0, 4: -3.0}   # J2=효과있음, J3=없음, J4=약간
        common2 = {"OFF": 6.0, "ON": -6.0}
        texts2 = [make_run(c, common2[c], TRUE2) for c in order]
        v2 = analyze(order, texts2, control_joints=(1, 5, 6), treatment_joints=(2, 3, 4))
        assert v2[2]["adj_diff"] < -3, f"J2 참효과(-8) 복원 실패: {v2[2]['adj_diff']}"
        assert abs(v2[3]["adj_diff"]) < 5, f"J3 참효과(0) 오판: {v2[3]['adj_diff']}"
        assert v2[4]["adj_diff"] < -0.5, f"J4 참효과(-3) 복원 실패: {v2[4]['adj_diff']}"
        print("✅ 역할반전(대조군/판정대상 교체)에서도 참효과 정확히 복원")

        # [10차, §51] metric="mean_err" 검증 - J3만 판정대상, 나머지 5관절 대조군.
        # vary="mean_err"로 평균오차 컬럼에 참효과를 심어, lag 컬럼이 아니라
        # 정말 mean_err 컬럼을 읽는지까지 함께 확인한다.
        print("\n--- 평균오차 지표 검증(§51, 대조군=J1,J2,J4,J5,J6, 판정대상=J3) ---")
        TRUE3 = {3: -0.35}   # J3=효과있음(중력처짐 보정이 편향을 줄임), 나머지는 0(대조군)
        common3 = {"OFF": 0.15, "ON": -0.15}
        texts3 = [make_run(c, common3[c], TRUE3, noise=0.03, vary="mean_err") for c in order]
        v3 = analyze(order, texts3, control_joints=(1, 2, 4, 5, 6),
                     treatment_joints=(3,), metric="mean_err", unit="도")
        assert v3[3]["adj_diff"] < -0.2, f"J3 참효과(-0.35) 복원 실패: {v3[3]['adj_diff']}"
        print("✅ metric=\"mean_err\"로 평균오차 컬럼을 정확히 읽고 참효과 복원")

        # 파서 자체 검증 - 세 metric이 각각 올바른 컬럼을 읽는지 확인
        sample = "   J3  |       1.234  |       5.678  |      142.5  |           30.1"
        got = parse_joint_lag("header\n" + sample)   # 기본값 metric="lag"
        assert got == {3: 142.5}, got
        got_mean = parse_joint_lag("header\n" + sample, metric="mean_err")
        assert got_mean == {3: 1.234}, got_mean
        got_max = parse_joint_lag("header\n" + sample, metric="max_err")
        assert got_max == {3: 5.678}, got_max
        print("✅ parse_joint_lag 단일행 파싱 정상 (lag/mean_err/max_err 전부)")

        print("\n모든 자체 점검 통과.")
    else:
        main()
