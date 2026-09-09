"""
정지 상태 정상상태 오차 측정 + 관절 오프셋 보정 검증  (1회성 진단 스크립트)
================================================================================

왜 필요한가
-----------
Figure 1의 t=0 을 보면 로봇이 **완전히 멈춰 있는데도** commanded 와 measured 가
어긋나 있었다. 속도가 0이므로 이건 추종지연도, 경로계획 문제도 아니다. 서보의
정상상태 오차(백래시 · 감속기 유격 · 중력 처짐 · 각도 분해능)다.

**이 값이 이 시스템 정확도의 바닥이다.** 파라미터 튜닝으로는 이 아래로 못
내려간다. `mycobot_kinematics.joint_offset_correction()`이 이 스크립트의
10자세 실측을 기반으로 관절별 보정 테이블을 만들어 `mycobot_path_editor.py`가
명령을 보내기 전에 자동 적용하도록 이미 반영돼 있다(§9 static_error_test 참고,
PROJECT_HANDOFF.md §12).

이 스크립트가 하는 일 (6가지, 메뉴로 선택)
-------------------------------------------
1. **기본 실측** (`run_static_error_test`) - 10개 자세로 이동해 명령각-실측각
   잔차를 잰다. 처음 만들었던 그 실험 그대로.
2. **J2 단독 스윕** (`run_j2_sweep_test`) - 다른 관절(J1,J3,J4,J5,J6)은 고정해두고
   J2만 0→-90도를 10도씩 훑는다. 기본 실측 10개 자세는 여러 관절이 동시에
   바뀌어서 J2 하나만의 순수 효과를 완전히 분리하지 못했다(회귀 R²=0.61,
   중간 신뢰도) - 이 스윕은 그 오염을 없애 J2 보정계수를 더 정확히 뽑는다.
3. **방향이력(hysteresis) 테스트 - 목표 1개** (`run_hysteresis_test`) - 같은
   목표 자세를 서로 다른 방향에서 반복 접근시켜 오차 부호가 뒤집히는지
   확인한다. 4차에서 이걸로 J1,J3,J4,J5의 방향이력을 확인했다.
4. **보정 검증** (`run_correction_validation_test`) - `joint_offset_correction`을
   실제로 적용해서 다시 측정 - 이론(1차 근사)이 아니라 실물로 개선폭을 확인.
5. **[5차 신규] 방향이력 테스트 - 다중 목표** (`run_hysteresis_multipose_test`)
   - PROJECT_HANDOFF.md §9 이슈7. 3번 테스트는 목표 자세 **1개**에서만
   방향이력을 쟀는데, 그 크기(A방향-B방향 차이)가 자세마다 다른지는
   아직 몰랐다. 이 테스트는 같은 방향이력 측정을 서로 다른 목표 자세
   5개에 대해 반복해서, 지금 일괄 적용 중인
   `_HYSTERESIS_SAFETY_SCALE=0.5`가 모든 자세에 적정한 절충값인지,
   아니면 자세별/관절별로 다른 계수가 필요한지 확인한다.
6. **[5차 신규] J6 세션 시작 재확인** (`run_j6_quick_recheck`) - §9 이슈8.
   J6는 방향이력은 없지만 세션 간 값 드리프트가 확인됐다(+0.26°→+0.08°).
   J6 기준 자세 1개만 짧게 재확인해 지금 세션의 실제 J6 오차를 재고,
   소스코드 상수와 크게 다르면 `update_j6_offset()`으로 이번 프로세스
   한정으로 즉시 갱신한다. 세션 시작 시 가장 먼저(다른 테스트보다 먼저)
   돌리는 걸 권장 - 30초 안팎이면 끝난다.
7. **[5차 신규] J6 단독 스윕** (`run_j6_sweep_test`) - 메뉴 6을 실제로 돌려본
   결과 소스 상수(+0.197°, J6=0 데이터로만 만들어짐)와 J6=20°에서의 실측
   (-0.230°) 사이에 **부호까지 뒤집히는 큰 차이**가 나왔다. 이게 "시간
   드리프트"인지 "J6 오차가 사실 J2처럼 명령값에 선형 의존하는데 여태
   0에서만 재서 상수로 착각한 것"인지 구분하기 위한 스윕(J2 스윕과 동일
   방식). J6를 -90→+90도까지 15도씩 훑어 회귀 R²을 본다.

실행:
    python3 mycobot_static_error_test.py
"""

import math
import time
import numpy as np

from mycobot_kinematics import (
    chain, active_indices, find_robot_port,
    SETTLE_DELAY_SEC, MOVE_TIMEOUT_SEC, SPEED, ALIGN_ANGLES, JOINT_LIMITS_DEG,
    joint_offset_correction, update_j6_offset,
)

# 테스트 자세들 - 팔을 편 자세/접은 자세/좌우로 돌린 자세를 골고루.
# 중력 부하가 다른 자세를 섞어야 '자세에 따라 오차가 달라지는가'를 볼 수 있다.
TEST_POSES = [
    ALIGN_ANGLES,
    [  0, -60,  60,   0,  30,  0],
    [  0, -20,  20,   0,  30,  0],
    [ 45, -45,  45,   0,  30,  0],
    [-45, -45,  45,   0,  30,  0],
    [ 90, -60,  30, -30,  60,  0],
    [-90, -60,  30, -30,  60,  0],
    [  0, -90,  90, -30,  30,  0],
    [  0, -10,  10,   0,  60,  0],
    [ 60, -30,  20,  20,  90,  0],
]

SETTLE_EXTRA_SEC = 1.0     # is_moving() 이 False가 된 뒤 추가로 더 기다릴 시간
REPEATS = 3                # 같은 자세를 몇 번 읽을지 (읽기 노이즈 분리용)


def fk_mm(angles_deg):
    """관절각(도, 6개) -> TCP 위치(mm, 3)"""
    q = [0.0] * len(chain.links)
    for i, idx in enumerate(active_indices):
        q[idx] = math.radians(angles_deg[i])
    return chain.forward_kinematics(q)[:3, 3] * 1000.0


def goto_and_settle(mc, angles):
    mc.send_angles(list(angles), SPEED)
    time.sleep(SETTLE_DELAY_SEC)
    t0 = time.time()
    while mc.is_moving():
        if time.time() - t0 > MOVE_TIMEOUT_SEC:
            break
        time.sleep(0.1)
    time.sleep(SETTLE_EXTRA_SEC)     # 서보가 완전히 자리 잡을 때까지


def run_static_error_test(mc):
    print("\n" + "=" * 74)
    print("[기본 실측] 10개 자세 - 관절각도/TCP 오차")
    print("=" * 74)
    rows = []
    for pi, pose in enumerate(TEST_POSES):
        goto_and_settle(mc, pose)

        reads = []
        for _ in range(REPEATS):
            a = mc.get_angles()
            if isinstance(a, list) and len(a) == 6:
                reads.append(a)
            time.sleep(0.15)
        if not reads:
            print(f"  자세 {pi+1}: 읽기 실패, 건너뜀")
            continue

        reads = np.array(reads, dtype=float)
        meas = reads.mean(axis=0)
        read_noise = reads.std(axis=0).max()      # 읽기 자체의 흔들림

        joint_err = meas - np.array(pose, dtype=float)
        tcp_err_vec = fk_mm(meas) - fk_mm(pose)
        tcp_err = float(np.linalg.norm(tcp_err_vec))

        rows.append((pose, joint_err, tcp_err_vec, tcp_err, read_noise))
        print(f"자세 {pi+1:2d} {np.array(pose)}")
        print("   관절오차(도) : " + " ".join(f"{e:+6.2f}" for e in joint_err)
              + f"   |최대| {np.abs(joint_err).max():.2f}")
        print(f"   TCP오차(mm)  : X{tcp_err_vec[0]:+7.2f} Y{tcp_err_vec[1]:+7.2f} "
              f"Z{tcp_err_vec[2]:+7.2f}   |합| {tcp_err:6.2f}")
        print(f"   읽기 흔들림  : {read_noise:.3f}도\n")

    print("=" * 74)
    if rows:
        tcp_all = np.array([r[3] for r in rows])
        jerr_all = np.abs(np.array([r[1] for r in rows]))
        print(f"정지 상태 TCP 오차 : 평균 {tcp_all.mean():.2f}mm / "
              f"중앙 {np.median(tcp_all):.2f}mm / 최대 {tcp_all.max():.2f}mm")
        print("관절별 |오차| 평균 : " +
              "  ".join(f"J{j+1}={jerr_all[:, j].mean():.2f}도" for j in range(6)))
        print()
        print(">> 이 TCP 오차 평균이 곧 '경로 추종으로 달성 가능한 정확도의 바닥'입니다.")
        print("   이 결과를 mycobot_kinematics.py의 _STATIC_ERR_POSES_DEG에 반영해")
        print("   joint_offset_correction()의 계수를 갱신하는 걸 고려하세요")
        print("   (지금 그 상수는 2026-08-12 실측값이 이미 들어가 있습니다).")

    goto_and_settle(mc, ALIGN_ANGLES)
    return rows


# ---------------------------------------------------------------------------
# [신규] J2 단독 스윕 - 다른 관절을 고정해 J2 하나만의 순수 효과를 잰다
# ---------------------------------------------------------------------------
J2_SWEEP_FIXED_J1J3J4J5J6 = (0, 30, 0, 30, 0)   # J1,J3,J4,J5,J6 고정값
J2_SWEEP_RANGE_DEG = list(range(0, -91, -10))   # 0,-10,...,-90


def run_j2_sweep_test(mc):
    print("\n" + "=" * 74)
    print("[J2 단독 스윕] J1,J3,J4,J5,J6 고정, J2만 0→-90도를 10도씩 훑는다")
    print("=" * 74)
    j1, j3, j4, j5, j6 = J2_SWEEP_FIXED_J1J3J4J5J6
    rows = []
    for j2 in J2_SWEEP_RANGE_DEG:
        pose = [j1, j2, j3, j4, j5, j6]
        goto_and_settle(mc, pose)
        reads = []
        for _ in range(REPEATS):
            a = mc.get_angles()
            if isinstance(a, list) and len(a) == 6:
                reads.append(a)
            time.sleep(0.15)
        if not reads:
            print(f"  J2={j2:+4d}°: 읽기 실패, 건너뜀")
            continue
        meas = np.array(reads, dtype=float).mean(axis=0)
        err = meas - np.array(pose, dtype=float)
        rows.append((j2, float(err[1])))
        print(f"  J2={j2:+4d}°   오차={err[1]:+.3f}°   (참고: J3오차={err[2]:+.3f}°)")

    if len(rows) >= 3:
        j2v = np.array([r[0] for r in rows], dtype=float)
        ev = np.array([r[1] for r in rows], dtype=float)
        a, b = np.polyfit(j2v, ev, 1)
        pred = a * j2v + b
        resid = ev - pred
        ss_tot = np.sum((ev - ev.mean()) ** 2)
        r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 1e-9 else float("nan")
        print(f"\nJ2 단독 회귀: err = {a:.5f}*J2 + {b:.5f}   R²={r2:.3f}")
        print("이 값을 mycobot_kinematics.py의 _fit_offset_models() 결과(10자세")
        print("혼합 회귀, R²=0.61)와 비교하세요. 다른 관절 부하가 안 섞였으므로")
        print("여기 R²가 더 높아야 정상입니다. 차이가 크면 kinematics.py의")
        print("_STATIC_ERR_POSES_DEG에 이 스윕 데이터를 추가하거나, J2 계수를")
        print("이 값으로 직접 교체하는 걸 권장합니다.")
    else:
        print("\n표본이 3개 미만이라 회귀를 못 냅니다.")

    goto_and_settle(mc, ALIGN_ANGLES)
    return rows


# ---------------------------------------------------------------------------
# [신규] 방향이력(hysteresis) 테스트 - 같은 목표를 반대 방향에서 반복 접근
# ---------------------------------------------------------------------------
HYSTERESIS_TARGET = [0, -45, 45, 0, 30, 0]
HYSTERESIS_FROM_A = [70, -20, 20, 30, 60, 0]     # 목표보다 전반적으로 '큰 값' 쪽에서 접근
HYSTERESIS_FROM_B = [-70, -70, 70, -30, 0, 0]    # 목표보다 전반적으로 '작은 값' 쪽에서 접근
HYSTERESIS_REPEATS = 4


def _approach_and_measure(mc, via_pose, target_pose):
    goto_and_settle(mc, via_pose)      # 이 경유점을 거쳐서
    goto_and_settle(mc, target_pose)   # 목표로 접근 (방향이 via_pose에 의해 결정됨)
    reads = []
    for _ in range(REPEATS):
        a = mc.get_angles()
        if isinstance(a, list) and len(a) == 6:
            reads.append(a)
        time.sleep(0.15)
    if not reads:
        return None
    meas = np.array(reads, dtype=float).mean(axis=0)
    return meas - np.array(target_pose, dtype=float)


def run_hysteresis_test(mc):
    print("\n" + "=" * 74)
    print("[방향이력 테스트] 같은 목표를 서로 다른 방향에서 반복 접근 - 오차 부호 확인")
    print(f"목표: {HYSTERESIS_TARGET}")
    print(f"A방향 경유점: {HYSTERESIS_FROM_A}")
    print(f"B방향 경유점: {HYSTERESIS_FROM_B}")
    print("=" * 74)

    errs_a, errs_b = [], []
    for rep in range(HYSTERESIS_REPEATS):
        ea = _approach_and_measure(mc, HYSTERESIS_FROM_A, HYSTERESIS_TARGET)
        eb = _approach_and_measure(mc, HYSTERESIS_FROM_B, HYSTERESIS_TARGET)
        if ea is not None:
            errs_a.append(ea)
        if eb is not None:
            errs_b.append(eb)
        print(f"  반복 {rep+1}: A방향 오차={np.round(ea, 3) if ea is not None else '읽기실패'}")
        print(f"           B방향 오차={np.round(eb, 3) if eb is not None else '읽기실패'}")

    if errs_a and errs_b:
        A = np.array(errs_a)
        B = np.array(errs_b)
        print("\n관절별 방향이력 분석:")
        print(f"{'':4s} {'A방향 평균':>12s} {'B방향 평균':>12s} {'차이':>9s}  방향이력?")
        any_flip = False
        for j in range(6):
            ma, mb = float(A[:, j].mean()), float(B[:, j].mean())
            flip = (ma > 0) != (mb > 0) and abs(ma) > 0.05 and abs(mb) > 0.05
            any_flip = any_flip or flip
            print(f"J{j+1:<3d} {ma:+11.3f}° {mb:+11.3f}° {ma-mb:+8.3f}°  "
                  f"{'예 - 방향이력 있음' if flip else '아니오'}")
        print()
        if any_flip:
            print(">> 방향이력이 확인된 관절은 목표각만으로 만든 상수/선형 보정 테이블이")
            print("   틀릴 수 있습니다 (mycobot_kinematics.py의 joint_offset_correction).")
            print("   그 관절은 접근방향까지 키에 넣거나, 두 방향 평균으로 절충하거나,")
            print("   보정 자체를 포기하는 게 안전합니다. J1이 유력한 후보였습니다")
            print("   (기본 실측에서 자세1~3 vs 자세8~9 사이 부호가 뒤집혔음).")
        else:
            print(">> 방향이력이 뚜렷하지 않습니다 - 지금의 상수/선형 보정을 그대로")
            print("   써도 좋을 근거가 됩니다.")
    else:
        print("\n측정 실패로 분석 불가.")

    goto_and_settle(mc, ALIGN_ANGLES)
    return errs_a, errs_b


# ---------------------------------------------------------------------------
# [신규] 보정 검증 - joint_offset_correction을 실제로 적용해 재측정
# ---------------------------------------------------------------------------
def run_correction_validation_test(mc):
    print("\n" + "=" * 74)
    print("[보정 검증] joint_offset_correction 적용 후 재측정 - 실측 개선폭 확인")
    print("(1차 근사가 아니라 실제로 보정된 명령을 보내고 다시 잰다)")
    print("=" * 74)
    rows = []
    for pi, pose in enumerate(TEST_POSES):
        corrected = joint_offset_correction(pose)
        goto_and_settle(mc, corrected)
        reads = []
        for _ in range(REPEATS):
            a = mc.get_angles()
            if isinstance(a, list) and len(a) == 6:
                reads.append(a)
            time.sleep(0.15)
        if not reads:
            print(f"  자세 {pi+1}: 읽기 실패, 건너뜀")
            continue
        meas = np.array(reads, dtype=float).mean(axis=0)
        # [중요] 오차는 '보정된 명령(corrected)' 대비가 아니라 '원래 목표(pose)' 대비로 잰다 -
        # 우리가 실제로 원하는 건 pose에 도달하는 것이므로 이게 진짜 성공 지표다.
        err_after = meas - np.array(pose, dtype=float)
        rows.append((pose, err_after))
        print(f"자세 {pi+1:2d} {np.array(pose)}  보정후 |오차|최대={np.abs(err_after).max():.3f}°"
              f"  (관절오차: " + " ".join(f"{e:+.2f}" for e in err_after) + ")")

    if rows:
        worst = np.array([np.abs(r[1]).max() for r in rows])
        allerr = np.array([r[1] for r in rows])
        print(f"\n보정 적용 후 |관절오차| 평균={worst.mean():.3f}°  최대={worst.max():.3f}°")
        print("관절별 |오차| 평균(보정후): " +
              "  ".join(f"J{j+1}={np.abs(allerr[:, j]).mean():.2f}도" for j in range(6)))
        print("\n(보정 전 원본 실측 기준: 평균 |오차| 약 0.83도, 최대 1.17도였습니다 -")
        print(" 여기서 뚜렷이 줄었으면 보정이 실제로 먹힌다는 뜻이고, 안 줄었거나")
        print(" 오히려 늘었으면 joint_offset_correction의 모델을 재검토해야 합니다.)")

    goto_and_settle(mc, ALIGN_ANGLES)
    return rows


# ---------------------------------------------------------------------------
# [5차 신규] 방향이력 테스트 - 다중 목표자세 (§9 이슈7)
# ---------------------------------------------------------------------------
# 목표 자세 5개 - 워크스페이스 여러 영역(중앙/좌/우/접힘/폄)을 골고루.
# 3번 테스트(HYSTERESIS_TARGET 1개)와 겹치지 않게 다른 자세를 우선했다.
HYSTERESIS_MULTIPOSE_TARGETS = [
    [  0, -45,  45,   0,  30,  0],   # 중앙, 기본 높이
    [ 60, -30,  20,  20,  30,  0],   # 오른쪽으로 튼 자세
    [-60, -30,  20,  20,  30,  0],   # 왼쪽으로 튼 자세
    [  0, -75,  75, -20,  60,  0],   # 많이 접은 자세 (중력부하 큼)
    [  0, -15,  15,  10,  60,  0],   # 많이 편 자세 (중력부하 작음)
]

# 각 목표에 대해 '큰 값 쪽'/'작은 값 쪽'에서 접근하는 경유점을 절차적으로
# 만든다. 목표에 이 델타를 더하고/빼서 만들고, 관절한계를 벗어나면 clip한다.
HYSTERESIS_MULTIPOSE_VIA_DELTA_DEG = [40, 25, 25, 25, 25, 0]
# [6차 갱신, §9 이슈7 후속] 2회였을 때 J3(변동비 1.23)/J4(0.68)가 노이즈인지
# 실제 자세의존 패턴인지 구분이 안 됐다 - 두 관절의 평균|A-B|(0.23°/0.41°)
# 자체가 작아 반복 2회로는 표본노이즈에 취약했다. 4회로 늘려 재확인한다.
# (J1/J5는 이미 변동비 0.06/0.01로 명백해서 반복을 더 늘려도 결론이 안
# 바뀔 것 - 총 실행시간 증가를 감안해 J3/J4 재확인이 주 목적이지만 5개
# 목표 전부 4회로 도는 게 구조상 더 간단해 전체를 늘린다.)
HYSTERESIS_MULTIPOSE_REPEATS = 4   # 목표당 반복 횟수


def _clip_to_joint_limits(pose_deg):
    out = []
    for v, (lo, hi) in zip(pose_deg, JOINT_LIMITS_DEG):
        out.append(min(max(v, lo + 1.0), hi - 1.0))   # 한계 바로 위/아래 1도 여유
    return out


def _via_poses_for_target(target):
    delta = HYSTERESIS_MULTIPOSE_VIA_DELTA_DEG
    via_a = _clip_to_joint_limits([t + d for t, d in zip(target, delta)])   # '큰 값' 쪽에서 접근
    via_b = _clip_to_joint_limits([t - d for t, d in zip(target, delta)])   # '작은 값' 쪽에서 접근
    return via_a, via_b


def run_hysteresis_multipose_test(mc):
    print("\n" + "=" * 74)
    print("[방향이력 - 다중 목표] 서로 다른 목표자세 5개에서 방향이력 크기를 비교")
    print("(PROJECT_HANDOFF.md §9 이슈7 - 지금의 일괄 50% 보정 강도가 자세에")
    print(" 상관없이 적정한지, 아니면 자세별 계수가 필요한지 확인)")
    print("=" * 74)

    per_pose_diff = []   # [(target, {joint_idx: A평균-B평균}), ...]
    for ti, target in enumerate(HYSTERESIS_MULTIPOSE_TARGETS):
        via_a, via_b = _via_poses_for_target(target)
        print(f"\n-- 목표 {ti+1}/5: {target}")
        print(f"   A방향 경유점: {[round(v,1) for v in via_a]}")
        print(f"   B방향 경유점: {[round(v,1) for v in via_b]}")

        errs_a, errs_b = [], []
        for rep in range(HYSTERESIS_MULTIPOSE_REPEATS):
            ea = _approach_and_measure(mc, via_a, target)
            eb = _approach_and_measure(mc, via_b, target)
            if ea is not None:
                errs_a.append(ea)
            if eb is not None:
                errs_b.append(eb)
            print(f"   반복 {rep+1}: A={np.round(ea, 3) if ea is not None else '읽기실패'}  "
                  f"B={np.round(eb, 3) if eb is not None else '읽기실패'}")

        if errs_a and errs_b:
            A = np.array(errs_a)
            B = np.array(errs_b)
            diff = {}
            for j in range(6):
                ma, mb = float(A[:, j].mean()), float(B[:, j].mean())
                diff[j] = ma - mb
            per_pose_diff.append((target, diff))
        else:
            print("   측정 실패 - 이 목표는 분석에서 제외")

    if len(per_pose_diff) < 2:
        print("\n표본(성공한 목표 수)이 2개 미만이라 자세간 비교를 못 합니다.")
        goto_and_settle(mc, ALIGN_ANGLES)
        return per_pose_diff

    print("\n" + "=" * 74)
    print("관절별 방향이력 크기(A-B) - 목표자세별 비교")
    header = "     " + "".join(f"목표{i+1:>8d}" for i in range(len(per_pose_diff)))
    print(header)
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
    from mycobot_kinematics import _HYSTERESIS_CONFIRMED  # noqa: local import for reporting only
    for j in range(6):
        vals = [diff[j] for _, diff in per_pose_diff]
        row = f"{joint_names[j]:<4s} " + "".join(f"{v:+9.3f}" for v in vals)
        confirmed = " (4차에서 방향이력 확인됨)" if j in _HYSTERESIS_CONFIRMED else ""
        print(row + confirmed)

    print("\n관절별 변동성(자세 간 A-B 값의 표준편차 / 평균|A-B|):")
    for j in range(6):
        vals = np.array([diff[j] for _, diff in per_pose_diff])
        mean_abs = np.abs(vals).mean()
        std = vals.std()
        ratio = std / mean_abs if mean_abs > 1e-6 else float("nan")
        note = ""
        if j in _HYSTERESIS_CONFIRMED:
            if ratio < 0.3:
                note = " -> 자세와 거의 무관하게 일정. 지금의 일괄 50% 상수로 충분해 보임."
            elif ratio < 0.7:
                note = " -> 어느 정도 자세에 따라 달라짐. 완전 무시하긴 애매함."
            else:
                note = " -> 자세마다 크게 달라짐. 일괄 상수 대신 자세별/구간별 계수 검토 권장."
        print(f"  {joint_names[j]}: 평균|A-B|={mean_abs:.3f}°  표준편차={std:.3f}°  "
              f"변동비={ratio:.2f}{note}")

    print("\n>> 위 '변동비'가 작은 관절은 현재의 _HYSTERESIS_SAFETY_SCALE=0.5 상수")
    print("   근사가 안전합니다. 변동비가 크게 나온 관절이 있다면")
    print("   mycobot_kinematics.py에 그 관절만 자세 구간별 보정 강도를 추가하는")
    print("   걸 고려하세요 (J3가 이미 |J2|>=55 기준 2단계 모델을 쓰는 것과 같은 방식).")

    goto_and_settle(mc, ALIGN_ANGLES)
    return per_pose_diff


# ---------------------------------------------------------------------------
# [5차 신규] J6 세션 시작 재확인 (§9 이슈8 - 세션간 드리프트 대응)
# ---------------------------------------------------------------------------
J6_RECHECK_POSE = [0, -45, 45, 0, 30, 20]     # J6=20도로 보내 오차를 재는 기준 자세
J6_RECHECK_REPEATS = 3
J6_DRIFT_WARN_DEG = 0.10   # 소스 상수와 이 이상 차이나면 경고
def run_j6_quick_recheck(mc):
    print("\n" + "=" * 74)
    print("[J6 세션 시작 재확인] J6는 방향이력은 없지만 세션간 드리프트가")
    print("확인된 관절입니다(§9 이슈8, +0.26°->+0.08° 사례). 30초 안팎 소요.")
    print("=" * 74)

    from mycobot_kinematics import _OFFSET_MODEL   # 최신 소스 상수와 비교하기 위해 직접 참조
    source_const = _OFFSET_MODEL["j6_mean"]

    goto_and_settle(mc, J6_RECHECK_POSE)
    reads = []
    for _ in range(J6_RECHECK_REPEATS):
        a = mc.get_angles()
        if isinstance(a, list) and len(a) == 6:
            reads.append(a)
        time.sleep(0.15)

    if not reads:
        print("  읽기 실패 - 재확인 불가. 소스 상수를 그대로 씁니다:"
              f" j6_mean={source_const:+.3f}°")
        goto_and_settle(mc, ALIGN_ANGLES)
        return None

    meas = np.array(reads, dtype=float).mean(axis=0)
    now_err = float(meas[5] - J6_RECHECK_POSE[5])
    diff = now_err - source_const
    print(f"  소스코드 상수(j6_mean) : {source_const:+.3f}°")
    print(f"  지금 세션 실측 오차    : {now_err:+.3f}°")
    print(f"  차이                   : {diff:+.3f}°")

    if abs(diff) >= J6_DRIFT_WARN_DEG:
        update_j6_offset(now_err, note="세션 시작 재확인")
        print(f"  -> 드리프트 확인({abs(diff):.3f}° >= {J6_DRIFT_WARN_DEG}°). 이번 세션은")
        print("     방금 잰 값으로 즉시 갱신했습니다(런타임 한정, 소스코드는 그대로).")
        print("     이 드리프트가 여러 세션에서 반복 재현되면 mycobot_kinematics.py의")
        print("     j6_mean 소스 상수를 영구적으로 갱신하는 걸 고려하세요(§9 이슈8).")
    else:
        print(f"  -> 소스 상수와 큰 차이 없음(< {J6_DRIFT_WARN_DEG}°). 갱신 없이 그대로 사용.")

    goto_and_settle(mc, ALIGN_ANGLES)
    return now_err


# ---------------------------------------------------------------------------
# [5차 신규] J6 단독 스윕 - §9 이슈8 후속. 재확인 테스트(메뉴 6)에서 소스 상수
# (+0.197°, J6=0에서 측정)와 J6=20° 실측(-0.230°) 사이에 부호까지 뒤집히는
# 큰 차이(-0.427°)가 나왔다. `_STATIC_ERR_POSES_DEG`를 보면 지금까지 J6은
# **항상 명령값 0에서만** 측정됐다 - 그래서 그 차이가 "시간에 따른 드리프트"인지
# 처음부터 "J6 오차가 J2처럼 명령값에 선형 의존하는데 여태 0에서만 재서 상수로
# 착각한 것"인지 구분이 안 됐다. 이 스윕(J2 스윕과 동일한 방식, 다른 관절 고정)이
# 그 둘을 가른다.
# ---------------------------------------------------------------------------
J6_SWEEP_FIXED_J1J2J3J4J5 = (0, -30, 30, 0, 30)   # [8차, §23.10] 더 이상 ALIGN_ANGLES와 동일하지 않음 -
# ALIGN_ANGLES가 진동 문제로 [0,-30,0,0,30,0]로 바뀌었지만(§23.10), 이
# 스윕은 예전 정렬자세와 같은 조건에서 J6 드리프트를 잰 과거 데이터와
# 비교하려는 목적이라 값을 일부러 그대로 뒀다. 여기를 ALIGN_ANGLES를
# 참조하도록 바꾸면 과거 J6 드리프트 측정과의 비교가 깨진다.
J6_SWEEP_RANGE_DEG = list(range(-90, 91, 15))     # -90,-75,...,90 (13단계)


def run_j6_sweep_test(mc):
    print("\n" + "=" * 74)
    print("[J6 단독 스윕] J1,J2,J3,J4,J5 고정, J6만 -90→+90도를 15도씩 훑는다")
    print("(§9 이슈8 후속 - J6 오차가 상수인지 명령값 의존적인지 확인)")
    print("=" * 74)
    j1, j2, j3, j4, j5 = J6_SWEEP_FIXED_J1J2J3J4J5
    rows = []
    for j6 in J6_SWEEP_RANGE_DEG:
        pose = [j1, j2, j3, j4, j5, j6]
        goto_and_settle(mc, pose)
        reads = []
        for _ in range(REPEATS):
            a = mc.get_angles()
            if isinstance(a, list) and len(a) == 6:
                reads.append(a)
            time.sleep(0.15)
        if not reads:
            print(f"  J6={j6:+4d}°: 읽기 실패, 건너뜀")
            continue
        meas = np.array(reads, dtype=float).mean(axis=0)
        err = meas - np.array(pose, dtype=float)
        rows.append((j6, float(err[5])))
        print(f"  J6={j6:+4d}°   오차={err[5]:+.3f}°")

    if len(rows) >= 3:
        j6v = np.array([r[0] for r in rows], dtype=float)
        ev = np.array([r[1] for r in rows], dtype=float)
        a_coef, b_coef = np.polyfit(j6v, ev, 1)
        pred = a_coef * j6v + b_coef
        resid = ev - pred
        ss_tot = np.sum((ev - ev.mean()) ** 2)
        r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 1e-9 else float("nan")
        const_rmse = float(np.sqrt(np.mean((ev - ev.mean()) ** 2)))
        linear_rmse = float(np.sqrt(np.mean(resid ** 2)))

        from mycobot_kinematics import _OFFSET_MODEL
        print(f"\nJ6 단독 회귀: err = {a_coef:.5f}*J6 + {b_coef:.5f}   R²={r2:.3f}")
        print(f"상수모델 RMSE={const_rmse:.3f}°  vs  선형모델 RMSE={linear_rmse:.3f}°")
        print(f"현재 소스 상수 j6_mean={_OFFSET_MODEL['j6_mean']:+.3f}°"
              "  (주의: 이 상수는 전부 명령 J6=0 데이터로만 만들어졌었다)")

        if abs(a_coef) > 0.01 and r2 > 0.3:
            print("\n>> 기울기가 유의미하고 R²도 준수합니다. J6 오차는 상수가 아니라")
            print("   명령값에 선형으로 의존하는 것으로 보입니다 - J2와 같은 패턴입니다.")
            print("   mycobot_kinematics.py의 joint_offset_correction()에서 J6을 상수")
            print("   (m['j6_mean'])이 아니라 이 회귀식(err_j6 = a*J6 + b)으로 바꾸는 걸")
            print("   권장합니다. 재확인 테스트(메뉴 6)에서 본 큰 드리프트(-0.427°)는")
            print("   시간 드리프트가 아니라 애초에 상수 모델 자체가 틀렸던 것일 가능성이 큽니다.")
        else:
            print("\n>> 뚜렷한 선형 추세가 보이지 않습니다(R² 낮음 또는 기울기 작음).")
            print("   재확인 테스트에서 본 큰 차이(-0.427°)는 명령값 의존성보다는")
            print("   진짜 시간에 따른 드리프트이거나 그날의 이상치일 가능성이 더 큽니다.")
            print("   여러 세션에 걸쳐 메뉴 6(재확인)을 반복해 패턴을 더 쌓아보세요.")
    else:
        print("\n표본이 3개 미만이라 회귀를 못 냅니다.")

    goto_and_settle(mc, ALIGN_ANGLES)
    return rows


def main():
    mc, port = find_robot_port()
    if mc is None:
        print("❌ 로봇을 찾지 못했습니다. PROJECT_HANDOFF.md §10 체크리스트를 확인하세요.")
        return
    print(f"✅ 연결됨: {port}\n")

    print("실행할 테스트를 고르세요 (쉼표로 여러 개, 예: 1,3  /  엔터 = 1만):")
    print("  1) 기본 실측 (10자세, ~2분)")
    print("  2) J2 단독 스윕 (10단계, ~2분)")
    print("  3) 방향이력 테스트 - 목표 1개 (4회 반복, ~3분)")
    print("  4) 보정 검증 (10자세, joint_offset_correction 적용, ~2분)")
    print("  5) 방향이력 테스트 - 다중 목표 5개 (§9 이슈7, ~10분 - 6차에서 반복 2→4로 증량)")
    print("  6) J6 세션 시작 재확인 (§9 이슈8, ~30초 - 다른 테스트보다 먼저 권장)")
    print("  7) [신규] J6 단독 스윕 (13단계, ~3분 - §9 이슈8 후속)")
    print("  a) 전부 실행 (~23분)")
    choice = input(">>> ").strip().lower()
    if choice == "":
        choice = "1"
    picks = set(choice.replace(" ", "").split(",")) if choice != "a" else {"1", "2", "3", "4", "5", "6", "7"}

    if "6" in picks:
        run_j6_quick_recheck(mc)
    if "7" in picks:
        run_j6_sweep_test(mc)
    if "1" in picks:
        run_static_error_test(mc)
    if "2" in picks:
        run_j2_sweep_test(mc)
    if "3" in picks:
        run_hysteresis_test(mc)
    if "4" in picks:
        run_correction_validation_test(mc)
    if "5" in picks:
        run_hysteresis_multipose_test(mc)

    print("\n모든 선택 테스트 완료. 정렬자세 복귀 완료.")


if __name__ == "__main__":
    main()
