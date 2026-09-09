# -*- coding: utf-8 -*-
"""
mycobot_error_model_fit.py
===========================
[9차 세션, §30, 진단/학습용 스크립트] `mycobot_error_model.py`가 쌓아둔
실행 기록으로 관절별 오차모델을 학습하고, **곡선 간 일반화가 되는지**를
검증한다.

**로봇 연결이 필요 없다** - 이미 저장된 `error_model_train.jsonl`만 읽는다.
GUI도 안 띄운다(mycobot_cycle_rhythm_diagnostic.py / mycobot_tau_floor_summary.py
와 같은 위치의 "GUI 없는 1회 실행용 스크립트").

**전제**: `mycobot_path_editor.py`로 **스트리밍 + '매번 실측'** 모드 실행을
최소 2개 이상의 **서로 다른 곡선**에서 해야 한다 - 곡선이 하나뿐이면
[2]번 교차검증(이 방식의 존재 이유)을 할 수 없다.

메뉴:
    [1] 전체 데이터로 학습 + 계수 해석 (관절별로 무엇이 오차를 지배하는가)
    [2] **곡선 간 교차검증** - 이 스크립트의 핵심.
        한 곡선을 빼고 학습 -> 뺀 곡선에서 평가. 곡선이 바뀌어도 통하는지가
        여기서 결정된다. 이게 통과해야 §30-3(실제 보정 적용)으로 갈 수 있다.
    [3] τ(위상지연) 설정 - §27 스텝응답 실측값 사용 (탐색은 불가능 - 코드 주석 참고)
    [q] 종료

**판정 기준(§30에서 정한 것)**:
    홀드아웃 곡선에서 RMSE가 20% 이상 줄면 -> 일반화가 실증됐다고 본다.
    5~20%면 애매(특징을 더 보태거나 데이터를 더 모아야 함).
    5% 미만이거나 음수면 -> 이 특징 집합으로는 안 된다는 뜻.
"""

import sys
from collections import defaultdict

import numpy as np

import mycobot_error_model as em

GENERALIZE_GOOD = 0.20      # 홀드아웃 RMSE 20% 이상 감소 = 일반화 실증
GENERALIZE_WEAK = 0.05

# [§30.7 실기 결정, §34에서 확장] 곡선 5종 교차검증 결과 J1/J4/J6은 계속
# 일관되게 좋고, [10차, §34] 관절별 특징셋 도입 후 J5도 39%(5홀드아웃 전부
# 양수)로 합류했다. J2/J3는 여전히 문턱 미달/불안정(가라비티·백래시,
# §4.8·§27.6.1)이라 제외 유지.
# [10차 세션, §42] §41의 대조군조정 A/B(6회, 무작위순서)에서 J4만 뚜렷한
# 신호가 나왔는데 **나쁜 방향**이었다(+11ms, 반복간 흩어짐의 2.3배 -
# raw/조정 거의 그대로라 공통교란이 아니라 J4 자체 효과로 판단). §32~34의
# 오프라인 교차검증은 J4가 24~32% 개선될 거라 예측했지만 실기가 정반대로
# 나온 것 - 선형 특징으로는 못 잡는 비선형성이거나 extended 특징셋이 J4에서
# 과적합됐을 가능성. J1/J5/J6은 조정 후 전부 0에 가까워(무효과, 해롭지도
# 않음) 그대로 유지한다.
SELECTED_JOINTS = [0, 4, 5]   # 0-based: J1, J5, J6 (J4 §42에서 제외)


def group_by_curve(runs):
    """곡선 라벨별로 실행을 묶는다(교차검증의 분할 단위)."""
    g = defaultdict(list)
    for r in runs:
        g[r.get("curve_label", "?")].append(r)
    return dict(g)


def search_tau(runs, joint):
    """[9차 세션, §30, 폐기됨 - 호출하지 말 것]

    **τ는 이 회귀에서 원리적으로 식별할 수 없다.** 처음엔 R²가 최대가 되는
    τ를 격자탐색하려 했는데, 합성검증에서 참값 130ms인 데이터에 320~400ms를
    "확신을 갖고" 돌려주는 걸 발견했다. 원인은 우연이 아니라 구조적이다:

        위상을 τ만큼 잘못 잡으면 오차에 (τ_참 - τ_추정)·ω 항이 생긴다.
        그런데 **ω는 이미 특징 벡터에 들어있다**(점성마찰 항). 그래서
        모델이 그 항을 ω 계수로 완벽히 흡수해버리고, 어떤 τ를 넣어도
        R²가 똑같이 좋게 나온다 - τ와 ω계수가 완전히 교락(confounded)된다.

    바꿔 말하면 **위상지연으로 생기는 오차는 ω 항이 이미 알아서 잡고 있다** -
    이 모델에 한해서는 τ 정렬이 (ω에 비례하는 성분에 대해서는) 불필요하다는
    뜻이기도 하다. 다만 α·sinθ·cosθ 항은 τ가 틀리면 잘못된 시점의 상태에서
    평가되므로, τ를 아예 무시하는 것보다는 §27에서 **스텝응답으로 독립 측정한**
    값을 쓰는 게 낫다(그건 이 회귀와 무관한 별개 실험이라 교락이 없다).

    이 함수는 같은 실수를 반복하지 않도록 설명만 남기고 비활성화한다.
    """
    raise NotImplementedError(
        "τ 격자탐색은 원리적으로 불가능합니다(ω 특징과 교락) - "
        "위 docstring 참고. §27 스텝응답 실측값을 쓰세요.")


# [§27 실측] 스텝응답으로 독립 측정한 관절별 τ(ms). 이 회귀와 무관한 별개
# 실험이라 교락 문제가 없다. 곡선/속도에 따라 달라지므로 대표값일 뿐이고,
# 정확한 값이 필요하면 mycobot_tau_pose_dependency_test.py로 다시 잰다.
TAU_FROM_STEP_RESPONSE_MS = [130, 150, 137, 132, 99, 95]


def cmd_set_tau(runs):
    """τ를 설정한다 - 탐색이 아니라 **외부 실측값 사용**이다(위 설명 참고)."""
    print("\nτ(위상지연) 설정")
    print("  [주의] R²를 최대화하는 τ 탐색은 원리적으로 불가능합니다 -")
    print("         위상 오정렬 오차 (τ차이)·ω 를 ω 특징이 그대로 흡수해서,")
    print("         어떤 τ든 똑같이 잘 맞는 것처럼 보입니다(합성검증에서")
    print("         참값 130ms에 320ms를 확신 있게 돌려주는 걸 확인했습니다).")
    print("         그래서 스텝응답으로 독립 측정한 값을 씁니다.\n")
    print(f"  [1] §27 스텝응답 실측 대표값 사용: {TAU_FROM_STEP_RESPONSE_MS}")
    print("  [2] 직접 입력 (쉼표로 6개, 예: 130,150,137,132,99,95)")
    print("  [3] τ 무시(전부 0) - ω 항이 흡수하므로 이것도 크게 나쁘진 않습니다")
    sel = input("  > ").strip()
    if sel == "2":
        raw = input("  6개 값(ms): ").strip()
        try:
            vals = [float(x) for x in raw.split(",")]
            if len(vals) != 6:
                raise ValueError("6개가 아님")
            return vals
        except Exception as e:
            print(f"  ⚠️ 입력 오류({e}) - §27 실측값을 씁니다.")
            return list(TAU_FROM_STEP_RESPONSE_MS)
    if sel == "3":
        return [0] * 6
    return list(TAU_FROM_STEP_RESPONSE_MS)


def cmd_fit_all(runs, taus=None):
    if taus is not None:
        for r in runs:
            r["tau_ms"] = taus
    models = em.fit_from_runs(runs)
    if not models:
        print("⚠️ 학습할 수 있는 데이터가 없습니다.")
        return None

    print(f"\n{'='*74}")
    print(f"  전체 학습 결과 ({len(runs)}개 실행)")
    print(f"{'='*74}")
    print(f"  {'관절':<5}{'R²':>8}{'샘플':>8}   " +
          "".join(f"{n:>11}" for n in em.FEATURE_NAMES))
    for j in sorted(models):
        m = models[j]
        print(f"  J{j+1:<4}{m['r2']:>8.3f}{m['n']:>8}   " +
              "".join(f"{v:>11.4f}" for v in m["w"]))

    print("\n  --- 계수 해석 (관절별로 무엇이 오차를 지배하는가) ---")
    for j in sorted(models):
        w = np.array(models[j]["w"])
        # 각 항의 기여 크기를 비교하려면 계수만으로는 안 되고 특징의 스케일을
        # 곱해야 한다 - 여기서는 학습 데이터의 표준편차를 곱해 정규화한다.
        contrib = {}
        for r in runs[:1]:
            rows = em.rows_from_run(r)
            if j in rows:
                X = rows[j][0]
                for i, name in enumerate(em.FEATURE_NAMES[:-1]):   # bias 제외
                    contrib[name] = abs(w[i]) * float(np.std(X[:, i]))
        if contrib:
            top = sorted(contrib.items(), key=lambda kv: -kv[1])[:3]
            desc = ", ".join(f"{k}({v:.3f}도)" for k, v in top)
            print(f"    J{j+1}: {desc}")
    print("\n    alpha=관성 / omega=점성마찰 / sign_omega=쿨롱마찰·백래시 /")
    print("    sin·cos_theta=중력부하  (전부 §4.8·§7.1·§9·§29.20에서 실측된 현상)")
    return models


def cmd_save_selected(runs, taus=None):
    """[§30.7 결정] J1/J4/J6만 **전체 데이터로** 다시 학습해서 저장한다.

    교차검증([2])은 "일반화가 되는가"를 확인하는 용도라 항상 한 곡선을
    빼고 학습했다 - 실제로 쓸 모델은 **가진 데이터를 전부 써서** 다시
    학습하는 게 맞다(검증 끝난 뒤에 홀드아웃을 계속 버릴 이유가 없다).
    """
    if taus is not None:
        for r in runs:
            r["tau_ms"] = taus
    models = em.fit_from_runs(runs)
    missing = [j for j in SELECTED_JOINTS if j not in models]
    if missing:
        print(f"\n⚠️ J{[j+1 for j in missing]} 학습 데이터가 없습니다 - 저장을 건너뜁니다.")
        return
    path = em.save_model(models, joints=SELECTED_JOINTS)
    print(f"\n✅ 저장 완료: {path}")
    print(f"   관절: {[f'J{j+1}' for j in SELECTED_JOINTS]}")
    for j in SELECTED_JOINTS:
        m = models[j]
        print(f"   J{j+1}: 전체데이터 R²={m['r2']:.3f} (n={m['n']})")
    print("\n   mycobot_stream_exec.py의 APPLY_ERROR_MODEL_FEEDFORWARD를 켜면")
    joint_names = "/".join(f"J{j+1}" for j in SELECTED_JOINTS)
    print(f"   다음 실행부터 이 파일을 읽어 {joint_names}에 피드포워드를 적용합니다.")
    print("   ⚠️ 실기 미검증 - 반드시 A/B로 개선을 확인한 뒤에 기본 ON으로 바꿀 것.")


def cmd_cross_validate(runs, taus=None):
    """**이 스크립트의 핵심** - 곡선을 빼고 학습해서 뺀 곡선에서 평가."""
    if taus is not None:
        for r in runs:
            r["tau_ms"] = taus

    groups = group_by_curve(runs)
    if len(groups) < 2:
        print(f"\n⚠️ 곡선이 {len(groups)}종뿐입니다 - 교차검증을 하려면 **서로 다른**")
        print("   곡선으로 최소 2종 이상 실행해야 합니다. 이 검증이야말로 이 방식이")
        print("   ILC 대신 선택된 이유(곡선 간 일반화)를 확인하는 유일한 방법입니다.")
        return

    print(f"\n{'='*74}")
    print(f"  곡선 간 교차검증 (곡선 {len(groups)}종)")
    print(f"{'='*74}")
    print("  한 곡선을 빼고 나머지로 학습 -> 뺀 곡선에서 평가합니다.\n")

    all_reduction = []
    for held_out, held_runs in groups.items():
        train_runs = [r for lbl, rs in groups.items() if lbl != held_out for r in rs]
        if not train_runs:
            continue
        models = em.fit_from_runs(train_runs)
        ev = em.evaluate(models, held_runs)
        if not ev:
            continue
        print(f"  [홀드아웃: {held_out}]  (학습 {len(train_runs)}회 / 평가 {len(held_runs)}회)")
        print(f"    {'관절':<6}{'R²':>8}{'RMSE 전':>10}{'RMSE 후':>10}{'감소':>9}")
        for j in sorted(ev):
            e = ev[j]
            red = 1 - e["rmse_after"] / max(e["rmse_before"], 1e-9)
            all_reduction.append(red)
            print(f"    J{j+1:<5}{e['r2']:>8.3f}{e['rmse_before']:>10.3f}"
                  f"{e['rmse_after']:>10.3f}{red*100:>8.0f}%")
        print()

    if not all_reduction:
        print("  ⚠️ 평가 결과가 없습니다.")
        return

    mean_red = float(np.mean(all_reduction))
    print(f"{'='*74}")
    print(f"  전체 평균 RMSE 감소: {mean_red*100:.1f}%")
    if mean_red >= GENERALIZE_GOOD:
        print("  ✅ **일반화 실증** - 학습에 없던 곡선에서도 오차가 뚜렷이 줄었습니다.")
        print("     -> §30-3(실제 피드포워드 보정 적용)으로 진행할 근거가 됩니다.")
    elif mean_red >= GENERALIZE_WEAK:
        print("  ⚠️ 애매합니다 - 효과는 있으나 약합니다. 데이터를 더 모으거나")
        print("     (다양한 곡선/속도) 특징을 보태는 것(예: α², ω², 관절 간 결합항)을")
        print("     검토하세요.")
    else:
        print("  ❌ 일반화가 안 됩니다 - 이 특징 집합으로는 곡선 간 전이가 어렵습니다.")
        print("     이 경우 ILC(곡선별 학습)로 되돌아가는 것도 합리적인 선택입니다")
        print("     (효과는 확실하나 곡선마다 다시 배워야 함).")


def main():
    runs = em.load_runs()
    print(f"저장된 실행 기록: {len(runs)}건")
    if not runs:
        print("\n⚠️ 학습 데이터가 없습니다.")
        print("   mycobot_path_editor.py에서 **스트리밍 + '매번 실측'** 모드로")
        print("   곡선을 실행하면 자동으로 쌓입니다.")
        print("   교차검증을 하려면 **서로 다른 곡선 2종 이상**이 필요합니다.")
        sys.exit(0)

    groups = group_by_curve(runs)
    print(f"곡선 종류: {len(groups)}종")
    for lbl, rs in groups.items():
        speeds = sorted({r.get("tcp_speed_mms") for r in rs})
        print(f"  - {lbl}: {len(rs)}회 (속도 {speeds})")

    # [9차, §30.9] 오차모델 피드포워드를 이미 켠 채로 쌓인 데이터가 섞여
    # 있으면 경고한다 - 그 실행의 오차는 "피드포워드가 이미 잡고 남은
    # 잔차"라 순수 PD 시절 데이터와 성격이 다르다. 섞어서 학습하면 이미
    # 보정된 몫까지 다시 배우려 들어 모델이 왜곡될 수 있다.
    contaminated = [r for r in runs if r.get("extra", {}).get("error_model_ff_applied")]
    if contaminated:
        joints = sorted({j for r in contaminated
                         for j in r["extra"].get("error_model_ff_joints", [])})
        print(f"\n⚠️ {len(contaminated)}건은 오차모델 피드포워드가 이미 켜진 채로 "
              f"실행됐습니다(관절: {[f'J{j+1}' for j in joints]}).")
        print("   그 관절들은 이 데이터로 재학습하면 이미 보정된 몫까지 다시 배우려")
        print("   들어 모델이 왜곡될 수 있습니다 - 재학습 전에 걸러내는 걸 권장합니다.")

    taus = None
    while True:
        print("\n--- 메뉴 ---")
        choice = input(
            "  [1] 전체 학습 + 계수 해석\n"
            "  [2] 곡선 간 교차검증 (핵심)\n"
            "  [3] τ 설정 (§27 스텝응답 실측값 - 탐색은 원리적으로 불가)\n"
            "  [4] [§30.7 결정] 선택 관절(J1/J4/J6)만 모델 저장 - stream_exec가 읽음\n"
            "  [q] 종료\n"
            "> "
        ).strip().lower()
        if choice == "q":
            break
        elif choice == "1":
            cmd_fit_all(runs, taus)
        elif choice == "2":
            cmd_cross_validate(runs, taus)
        elif choice == "3":
            taus = cmd_set_tau(runs)
            print(f"\n  -> 이후 [1]/[2]/[4]에 이 τ가 적용됩니다: {taus}")
        elif choice == "4":
            cmd_save_selected(runs, taus)
        else:
            print("잘못된 입력입니다.")
    print("종료합니다.")


if __name__ == "__main__":
    main()
