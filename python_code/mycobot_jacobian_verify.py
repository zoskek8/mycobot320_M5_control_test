# -*- coding: utf-8 -*-
"""
mycobot_jacobian_verify.py
===========================
[9차 세션, §24, 진단용 1회성 스크립트] mycobot_kinematics.py에 새로 추가한
analytic_jacobian_6d()(해석적/기하 야코비안, FK 1회)가 기존
numerical_jacobian_6d()(유한차분, FK 7회)와 **이 로봇의 실제 URDF**에서도
일치하는지 확인한다.

**로봇 연결이 필요 없다** - 순수하게 chain.forward_kinematics()만 돌린다.
GUI도 안 띄운다.

배경: 프로파일링(mycobot_validate_profile.py) 결과 numerical_jacobian_6d가
검증시간의 56%(1.99초/3.55초)를 먹고 있었다. 해석적 공식으로 바꾸면
FK가 7회->1회로 줄어 이론상 훨씬 빨라지지만, 근사가 아니라 "닫힌 형태
공식이 이 URDF의 관절축 파싱과 실제로 맞는가"를 실기에서 확인 전에는
믿을 수 없다 - 일반적인 임의 체인(myCobot과 무관) 2종에서는 무작위
자세 200개씩 최대오차 2.6e-7(유한차분 자체의 이산화 오차 수준)로
검증됐지만, 이 로봇의 실제 URDF에서는 아직이다.

이 스크립트가 통과하면: mycobot_kinematics.py 맨 위쪽의
USE_ANALYTIC_JACOBIAN = False 를 True로 바꾸기만 하면 된다.
(solve_pose_ik는 이미 그 스위치를 보고 골라 쓰도록 되어 있다.)

사용법:
    python3 mycobot_jacobian_verify.py

    [1] 무작위 자세 N개로 정확도 검증 (권장, 기본 500개)
    [2] 저장된 경로(curve_path_log.jsonl)의 실제 웨이포인트로 검증
        (실전과 가장 가까운 자세 분포 - 무작위보다 대표성 높음)
    [3] 속도 비교만 (이미 정확도는 확인했고 배속만 다시 보고 싶을 때)
    [q] 종료
"""

import time

import numpy as np
import mycobot_kinematics as mk
import mycobot_curve_log as curve_log
from mycobot_curve_math import PathPoint

TOL_MM_PER_S = 0.5     # 위치 야코비안 성분 오차 허용선 (라디안/초 단위 입력 기준 mm급)
TOL_DEG_PER_S = 0.5    # 자세 야코비안 성분 오차 허용선 (deg급, 회전벡터라 근사)
N_RANDOM_DEFAULT = 500


def random_valid_full_q(rng):
    """관절한계 안에서 무작위 자세를 뽑는다. 자가충돌 등은 안 걸러도 된다 -
    야코비안 공식 자체는 자세의 물리적 타당성과 무관하게 항상 계산되고,
    두 방법이 일치하는지만 보는 것이므로."""
    full_q = [0.0] * len(mk.chain.links)
    for i, idx in enumerate(mk.active_indices):
        lo, hi = mk.JOINT_LIMITS_DEG[i]
        deg = rng.uniform(lo, hi)
        full_q[idx] = np.radians(deg)
    return full_q


def run_random_check(n):
    rng = np.random.default_rng(0)
    print(f"\n무작위 자세 {n}개로 analytic vs numerical 대조 중...")
    max_pos_err = 0.0
    max_ori_err = 0.0
    worst_q = None
    t_analytic = 0.0
    t_numeric = 0.0
    for i in range(n):
        full_q = random_valid_full_q(rng)

        t0 = time.perf_counter()
        Ja = mk.analytic_jacobian_6d(full_q)
        t_analytic += time.perf_counter() - t0

        t0 = time.perf_counter()
        Jn = mk.numerical_jacobian_6d(full_q)
        t_numeric += time.perf_counter() - t0

        err_pos = np.max(np.abs(Ja[:3, :] - Jn[:3, :]))
        err_ori = np.max(np.abs(np.degrees(Ja[3:, :]) - np.degrees(Jn[3:, :])))
        if err_pos > max_pos_err:
            max_pos_err = err_pos
            worst_q = list(full_q)
        max_ori_err = max(max_ori_err, err_ori)

        if (i + 1) % max(1, n // 10) == 0:
            print(f"  {i+1}/{n}  현재까지 최대오차: 위치 {max_pos_err:.2e}, 자세 {max_ori_err:.2e}deg")

    report_result(max_pos_err, max_ori_err, worst_q, t_analytic, t_numeric, n)


def run_logged_check():
    records = curve_log.list_recent(limit=curve_log.DEFAULT_LIST_LIMIT)
    if not records:
        print("⚠️ 저장된 경로가 없습니다 (curve_path_log.jsonl). [1]번 무작위 검증을 쓰세요.")
        return

    print("\n=== 저장된 경로 (최신 5개) ===")
    for i, rec in enumerate(records, start=1):
        print(f"  [{i}] {rec.get('timestamp', '?')}  [{rec.get('label', '?')}]"
              f"  ({rec.get('n_points', '?')}점)")
    choice = input("검증할 번호 (Enter=1번): ").strip() or "1"
    try:
        idx = int(choice)
    except ValueError:
        idx = 1
    if not (1 <= idx <= len(records)):
        idx = 1
    rec = records[idx - 1]
    pts = [PathPoint.from_snapshot(t) for t in curve_log.snapshots_as_tuples(rec)]

    # 각 점에서 natural_pose_at으로 실제 도달 가능한 자세를 구해 그 지점에서 비교한다.
    # (임의 좌표를 그냥 IK 없이 쓰면 로봇이 실제로 쓸 일 없는 부자연스러운 자세라
    #  대표성이 떨어진다 - 실제로 웨이포인트 검증 때 거치는 자세와 최대한 비슷하게.)
    print(f"\n[{rec.get('label','?')}] {len(pts)}개 점에서 자세를 구해 검증합니다...")
    q_seed = mk.ALIGN_ANGLES
    full_q_seed = [0.0] * len(mk.chain.links)
    for i, idx2 in enumerate(mk.active_indices):
        full_q_seed[idx2] = np.radians(q_seed[i])

    tested = []
    for p in pts:
        q, _R = mk.natural_pose_at(p.coord(), full_q_seed)
        if q is None:
            print(f"  ⚠️ ({p.x:.0f},{p.y:.0f},{p.z:.0f}) 도달 불가 - 건너뜀")
            continue
        tested.append(q)
        full_q_seed = q

    if not tested:
        print("⚠️ 도달 가능한 점이 하나도 없었습니다 - [1]번 무작위 검증을 쓰세요.")
        return

    max_pos_err = 0.0
    max_ori_err = 0.0
    worst_q = None
    t_analytic = t_numeric = 0.0
    for full_q in tested:
        t0 = time.perf_counter()
        Ja = mk.analytic_jacobian_6d(full_q)
        t_analytic += time.perf_counter() - t0
        t0 = time.perf_counter()
        Jn = mk.numerical_jacobian_6d(full_q)
        t_numeric += time.perf_counter() - t0
        err_pos = np.max(np.abs(Ja[:3, :] - Jn[:3, :]))
        err_ori = np.max(np.abs(np.degrees(Ja[3:, :]) - np.degrees(Jn[3:, :])))
        if err_pos > max_pos_err:
            max_pos_err = err_pos
            worst_q = list(full_q)
        max_ori_err = max(max_ori_err, err_ori)

    report_result(max_pos_err, max_ori_err, worst_q, t_analytic, t_numeric, len(tested))


def report_result(max_pos_err, max_ori_err, worst_q, t_analytic, t_numeric, n):
    print(f"\n{'='*60}")
    print(f"  결과 ({n}개 자세)")
    print(f"{'='*60}")
    print(f"  위치 야코비안 최대오차: {max_pos_err:.6e}  (허용선 {TOL_MM_PER_S})")
    print(f"  자세 야코비안 최대오차: {max_ori_err:.6e} deg  (허용선 {TOL_DEG_PER_S})")
    print(f"  해석적 누적시간: {t_analytic:.4f}s")
    print(f"  수치미분 누적시간: {t_numeric:.4f}s")
    if t_analytic > 0:
        print(f"  배속: {t_numeric/t_analytic:.1f}배")

    ok = max_pos_err < TOL_MM_PER_S and max_ori_err < TOL_DEG_PER_S
    if ok:
        print("\n✅ 통과 - analytic_jacobian_6d가 이 로봇 URDF에서도 numerical_jacobian_6d와 일치합니다.")
        print("   mycobot_kinematics.py 상단의 USE_ANALYTIC_JACOBIAN = False 를")
        print("   True로 바꾸면 됩니다. 바꾼 뒤에는 mycobot_static_error_test.py나")
        print("   실제 곡선 실행으로 정확도(cross-track 등)가 그대로인지 한 번 더 확인하세요")
        print("   (야코비안은 특이점 판정/IK 스텝 방향에만 쓰이지 결과 자세 자체를")
        print("    바꾸는 게 아니라서 정확도 영향은 없어야 하지만, 확인 없이 넘어가지 않는 게 원칙입니다).")
    else:
        print("\n❌ 불일치 - 오차가 허용선을 넘었습니다. USE_ANALYTIC_JACOBIAN을 켜지 마세요.")
        print("   최악 오차가 나온 자세(라디안, full_q):")
        print(f"   {worst_q}")
        print("   원인 후보: URDF의 축 방향 정의(rotation 부호), 또는 여러 관절이 같은")
        print("   원점을 공유하는 특수 구조. analytic_jacobian_6d()의 축/원점 추출 부분을")
        print("   이 자세로 디버깅해보세요.")


def main():
    print("✅ mycobot_kinematics 로드 완료 (로봇 연결 없음)")
    print("   analytic_jacobian_6d / numerical_jacobian_6d 둘 다 사용 가능")
    print(f"   현재 USE_ANALYTIC_JACOBIAN = {mk.USE_ANALYTIC_JACOBIAN}")

    while True:
        print("\n--- 메뉴 ---")
        choice = input(
            "  [1] 무작위 자세로 정확도 검증 (권장)\n"
            "  [2] 저장된 경로의 실제 웨이포인트로 검증\n"
            "  [3] 속도 비교만 (무작위 200개)\n"
            "  [q] 종료\n"
            "> "
        ).strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            raw = input(f"자세 개수 (Enter={N_RANDOM_DEFAULT}): ").strip()
            n = int(raw) if raw else N_RANDOM_DEFAULT
            run_random_check(n)
        elif choice == "2":
            run_logged_check()
        elif choice == "3":
            run_random_check(200)
        else:
            print("잘못된 입력입니다.")

    print("종료합니다.")


if __name__ == "__main__":
    main()
