# -*- coding: utf-8 -*-
"""
mycobot_natural_pose_ab_check.py
===================================
[9차 세션, §24 후속 진단] natural_pose_at을 solve_position_ik 기반으로
바꾼 뒤, 같은 곡선인데 검증 결과가 달라지는 사례가 나왔다. 원인 후보는
"앵커에서 고르는 자세 자체가 달라져서, build_orientations()의
RotationSpline 보간을 통해 곡선 전체의 자세 궤적이 바뀌었다"는 것.

이걸 직접 확인하려면 원래 곡선을 검증 통과시켜 실행까지 가야 하는데,
지금은 검증 자체가 막혀서 그 경로로는 비교가 안 된다. 그럴 필요 없다 -
궁금한 건 "자세가 다른가"뿐이므로, **검증/실행을 거치지 않고
natural_pose_at만 두 설정(USE_FAST_NATURAL_POSE_AT True/False)으로 직접
불러 비교**한다. 로봇 연결도 필요 없다.

build_orientations()와 정확히 같은 순서로 돈다 - 앵커뿐 아니라 모든 점
(제어점 포함)에서 natural_pose_at을 호출하고, 그 결과가 다음 점의 시드로
이어진다(mycobot_curve_math.py의 실제 구현 그대로).

사용법:
    python3 mycobot_natural_pose_ab_check.py

    [1] 저장된 경로에서 곡선을 불러와 비교 (권장)
    [2] 이번 세션에서 문제 됐던 곡선을 좌표로 직접 입력해 비교
    [q] 종료

해석 가이드:
    - 자세 차이(도)가 점마다 작고 고르면: 두 방식이 비슷한 자세를 고른다는
      뜻 - 이번 문제의 원인이 아닐 가능성이 높다.
    - 특정 점(특히 그 이후 급커브 구간)에서 갑자기 크게 벌어지면: 그 점의
      자세 선택이 두 방식 사이에서 갈렸고, 그 뒤로 계속 다른 시드를 쓰게
      되어 하류 전체가 달라졌다는 뜻 - 유력한 원인.
    - 한쪽만 실패(None)하는 점이 있으면: 그 자체로 결정적 차이.
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import mycobot_kinematics as mk
import mycobot_curve_log as curve_log
from mycobot_curve_math import PathPoint, ik_seed_q


def rotation_angle_deg(R1, R2):
    """두 회전행렬 사이의 각도 차이(도). 회전벡터 노름."""
    dR = np.asarray(R1) @ np.asarray(R2).T
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec())))


def build_orientations_with_switch(points, q_seed, use_fast):
    """mycobot_curve_math.build_orientations()와 정확히 같은 순서 - 모든 점에서
    natural_pose_at을 호출하고, 결과를 다음 점의 시드로 이어붙인다.
    실행/검증 없이 이 함수만 따로 재현해서 두 설정을 비교하기 위함."""
    orig = mk.USE_FAST_NATURAL_POSE_AT
    mk.USE_FAST_NATURAL_POSE_AT = use_fast
    try:
        orientations = []
        q = list(q_seed)
        fail_at = None
        for i, p in enumerate(points):
            qn, Rm = mk.natural_pose_at((p.x, p.y, p.z), q)
            if qn is None:
                orientations.append(None)
                if p.role == "anchor" and fail_at is None:
                    fail_at = i
                continue
            orientations.append(Rm)
            q = qn
        return orientations, fail_at
    finally:
        mk.USE_FAST_NATURAL_POSE_AT = orig   # 비교 끝나면 반드시 원상복구


def adjacent_jumps_deg(orientations):
    """[검증 중 정정] 옛 방식(ikpy)과 "얼마나 다른가"는 사실 잘못된 기준이었다 -
    ikpy도 이 곡선에서 점6에 실패했으니 '정답'이 아니다. RotationSpline이
    실제로 보간하는 건 인접한 점끼리의 자세 차이이므로, 그 방식 자기 자신의
    궤적 안에서 이웃한 점 사이 점프가 얼마나 큰지가 진짜 봐야 할 지표다.
    반환: [((i,i+1) 인덱스쌍, 각도차이), ...] - None이 낀 구간은 건너뜀."""
    jumps = []
    for i in range(len(orientations) - 1):
        R1, R2 = orientations[i], orientations[i + 1]
        if R1 is None or R2 is None:
            continue
        jumps.append((i, rotation_angle_deg(R1, R2)))
    return jumps


def compare(points, label):
    q_seed = ik_seed_q()

    print(f"\n{'='*70}")
    print(f"  [{label}]  점 {len(points)}개")
    print(f"{'='*70}")

    ori_new, fail_new = build_orientations_with_switch(points, q_seed, True)
    ori_old, fail_old = build_orientations_with_switch(points, q_seed, False)

    print(f"  새 방식(solve_position_ik) 실패 앵커: {fail_new}")
    print(f"  옛 방식(ikpy)              실패 앵커: {fail_old}")

    # [정정] 예전엔 여기서 new vs old를 점별로 비교했다 - 아래 self-consistency
    # 비교가 실제로 중요한 지표다. 신구 비교는 참고용으로만 남긴다.
    print("\n  --- 참고용: 신구 방식 점별 차이 (절대적 기준 아님) ---")
    print(f"  {'idx':<5}{'role':<8}{'신구차이(도)'}")
    for i, (p, Rn, Ro) in enumerate(zip(points, ori_new, ori_old)):
        if Rn is None or Ro is None:
            note = "새방식 실패" if Rn is None else "옛방식 실패"
            print(f"  {i:<5}{p.role:<8}{note}")
            continue
        print(f"  {i:<5}{p.role:<8}{rotation_angle_deg(Rn, Ro):.3f}")

    # --- 진짜 중요한 지표: 각 방식 '자기 궤적' 안에서 인접점 점프 크기 ---
    print("\n  --- 핵심: 방식별 인접점 간 자세 점프 (RotationSpline이 실제로 보간하는 값) ---")
    jumps_new = adjacent_jumps_deg(ori_new)
    jumps_old = adjacent_jumps_deg(ori_old)

    print(f"  {'구간':<10}{'새방식(도)':<14}{'옛방식(도)'}")
    for k in range(max(len(jumps_new), len(jumps_old))):
        seg_new = f"{jumps_new[k][0]}->{jumps_new[k][0]+1}" if k < len(jumps_new) else "-"
        val_new = f"{jumps_new[k][1]:.2f}" if k < len(jumps_new) else "-"
        val_old = f"{jumps_old[k][1]:.2f}" if k < len(jumps_old) else "-"
        flag = "  <-- 급점프" if (k < len(jumps_new) and jumps_new[k][1] > 30) else ""
        print(f"  {seg_new:<10}{val_new:<14}{val_old}{flag}")

    if jumps_new:
        max_new = max(j[1] for j in jumps_new)
        print(f"\n  📎 새 방식 최대 인접점프: {max_new:.2f}도")
    if jumps_old:
        max_old = max(j[1] for j in jumps_old)
        print(f"  📎 옛 방식 최대 인접점프: {max_old:.2f}도 (단, 이 방식은 점6에서 실패해 곡선을 못 끝냈다는 점 감안)")
    print("\n  → 이 값(인접점프)이 작아야 RotationSpline이 매끄럽게 보간한다.")
    print("     '신구 차이'가 커도 인접점프 자체가 작으면 문제 없을 수 있다.")


def main():
    while True:
        print("\n--- 메뉴 ---")
        choice = input(
            "  [1] 저장된 경로에서 곡선을 불러와 비교 (권장)\n"
            "  [2] 좌표 직접 입력해 비교\n"
            "  [q] 종료\n"
            "> "
        ).strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            records = curve_log.list_recent(curve_log.DEFAULT_LIST_LIMIT)
            if not records:
                print("⚠️ 저장된 경로가 없습니다.")
                continue
            print("\n=== 저장된 경로 ===")
            for i, rec in enumerate(records, start=1):
                print(f"  [{i}] {rec.get('timestamp','?')}  [{rec.get('label','?')}]"
                      f"  ({rec.get('n_points','?')}점)")
            sel = input("비교할 번호 (Enter=1번): ").strip() or "1"
            try:
                idx = int(sel)
            except ValueError:
                idx = 1
            if not (1 <= idx <= len(records)):
                idx = 1
            rec = records[idx - 1]
            pts = [PathPoint.from_snapshot(t) for t in curve_log.snapshots_as_tuples(rec)]
            compare(pts, rec.get("label", "?"))
        elif choice == "2":
            print("좌표를 'x,y,z,role' 형식으로 한 줄에 하나씩 입력하세요")
            print("(role은 anchor 또는 ctrl). 빈 줄 입력하면 종료.")
            pts = []
            while True:
                line = input(f"  점 {len(pts)+1}: ").strip()
                if not line:
                    break
                try:
                    x, y, z, role = line.split(",")
                    pts.append(PathPoint(float(x), float(y), float(z), role=role.strip()))
                except Exception as e:
                    print(f"  ⚠️ 파싱 실패: {e}")
            if len(pts) < 2:
                print("⚠️ 점이 너무 적습니다.")
                continue
            compare(pts, "직접입력")
        else:
            print("잘못된 입력입니다.")

    print("종료합니다.")


if __name__ == "__main__":
    main()
