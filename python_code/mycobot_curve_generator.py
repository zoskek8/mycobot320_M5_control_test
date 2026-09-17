# -*- coding: utf-8 -*-
"""
mycobot_curve_generator.py
===========================
[10차 세션, §51 준비] J3 중력처짐 모델(§50)을 **새 곡선**에서 검증하기 위한
곡선 생성기.

왜 필요한가
------------
§50 모델은 곡선 6종(e433b126, e651e55d, fff98c63, fc043529, ea5354ed,
78dfad65)으로 적합했다. 그 6종으로 실기 A/B를 하면 학습 데이터로 검증하는
셈이라, "임의의 새 곡선을 따라간다"는 이 프로젝트의 목표를 확인하지 못한다
(§49가 1종으로 적합해 같은 1종으로 검증한 실수의 완화판일 뿐이다).

그런데 사람이 GUI에서 곡선을 그릴 때 "J3 평균자세가 이만큼 되게" 그리는 건
불가능하다 - 화면에 보이는 건 TCP 좌표지 관절각이 아니기 때문이다. 이
스크립트가 그 역할을 대신한다: 작업공간 안에서 곡선을 무작위로 만들어보고,
**실제 검증과 똑같은 알고리즘**으로 IK를 돌려 J3 자세 분포를 재본 뒤,
J3 평균자세가 서로 충분히 다른 것들만 골라낸다.

무엇을 하지 않는가
-------------------
- 로봇을 움직이지 않는다. IK/충돌검사만 하는 순수 오프라인 계산이다.
- 기존 코드를 고치지 않는다. `curve_path_log.jsonl`에 곡선을 추가할 뿐이라,
  GUI의 기존 "저장된 경로 불러오기"로 그대로 불러올 수 있다.

사용법
-------
    python3 mycobot_curve_generator.py              # 탐색만 (파일 안 건드림)
    python3 mycobot_curve_generator.py --save       # 고른 곡선을 로그에 저장
    python3 mycobot_curve_generator.py --tries 200  # 통과가 적으면 시도를 늘린다
    python3 mycobot_curve_generator.py --from-existing   # 기존 곡선 변형 (통과율 높음)

저장 후 GUI에서 '저장된 경로 불러오기' -> `gen_*` 라벨을 고르면 된다.

첫 판에서 0/60이 나왔던 이유 (같은 실수 반복 방지)
----------------------------------------------------
후보를 큰 직육면체 안에 무작위로 뿌렸더니 60개 전부 떨어졌다. 실패의
대부분(44/60)은 앵커가 아니라 **중간 웨이포인트의 IK 미도달**이었다 -
앵커는 닿는데 그 사이가 안 닿는 형태다. 원인은 앵커가 사방에 흩어지면
`natural_pose_at`이 앵커마다 전혀 다른 자세를 주고, 그 사이를
RotationSpline으로 이을 때 도달 불가능한 자세가 생기기 때문이다. 사람이
GUI에서 그린 곡선에 이 문제가 없는 건, 검증을 통과할 때까지 드래그로
고친 결과가 곧 "자세 보간이 성립하는 곡선"이기 때문이다.

그래서 지금은 **좁은 방위각 구역 안의 랜덤워크**로 만들고, 반지름·고도각·
앵커 간격을 전부 `curve_path_log.jsonl`의 실측 분포에 맞춘다. 특히 고도각은
실제로 41.6~82.6°(중앙 65°)인데 처음엔 -10~60°로 추측해 넣어서 후보
대부분이 생성 단계에서 버려졌었다 - **이 영역 상수들은 추측하지 말고
실측에서 뽑을 것.**

탐색은 2단계다: 거친 간격(10mm)으로 싸게 훑어 떨어질 곡선을 먼저 걸러내고,
살아남은 것만 실제 실행 간격(1.68mm)으로 정밀 재검증한다. 최종 판정은 항상
정밀 쪽이라 기준이 느슨해지지는 않는다.

[중요] 출력되는 **예측 편향은 실행 전에 기록해 둘 것.** §51의 검증 방식은
"예측을 먼저 적고 실측과 대조"하는 것이라, 실행 후에 예측을 계산하면
사후해석이 되어 검증의 의미가 사라진다.
"""

import argparse
import hashlib
import math
import sys

import numpy as np

from mycobot_kinematics import (
    solve_pose_ik, within_joint_limits, check_self_collision,
    analytic_jacobian_6d, jacobian_condition_number,
    active_indices as _ACTIVE,
    CONDITION_NUMBER_MAX, DESK_SAFETY_MARGIN_MM,
    _J3_GRAVITY_A, _J3_GRAVITY_B,
)
from mycobot_curve_math import (
    bezier_chain_eval, build_orientations, resample_by_chord, ik_seed_q,
)
import mycobot_curve_log as curve_log

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
# 검증 허용오차/간격은 **path_editor와 같은 값**을 써야 한다 - 여기서 통과한
# 곡선이 GUI '경로 검증'에서 떨어지면 이 스크립트는 쓸모가 없다. 손으로 베껴
# 적으면 나중에 한쪽만 바뀌어 어긋나므로 실제 모듈에서 계산해 가져온다.
from mycobot_stream_exec import (
    STREAM_TCP_SPEED_MMS, STREAM_STEP_MM,
    MEASURE_CYCLE_SEC, ABSORB_SPIKES_TEST, ABSORB_TARGET_PERIOD_SEC,
)

POS_TOL_MM = max(0.05, STREAM_STEP_MM / 10.0)      # path_editor.CURVE_POS_TOL_MM
ORIENT_TOL_DEG = 0.5                                # path_editor.CURVE_ORIENT_TOL_DEG
# path_editor._stream_step_mm(measure_mode="full")와 동일 - §51 실험은 '매번 실측'
_PERIOD = ABSORB_TARGET_PERIOD_SEC if ABSORB_SPIKES_TEST else MEASURE_CYCLE_SEC["full"]
STEP_MM = float(min(6.0, max(0.5, STREAM_TCP_SPEED_MMS * _PERIOD)))

# 거친 1차 선별용 간격 - 실패할 곡선을 싸게 걸러낸다(IK 호출이 6배 줄어든다).
# 여기 통과한 것만 위 STEP_MM으로 정밀 재검증하므로 판정 기준은 안 느슨해진다.
COARSE_STEP_MM = 10.0

# 후보를 뿌릴 영역 - 기존 성공 곡선 15개의 실측 통계에서 뽑았다.
# 여기 숫자를 추측으로 정하면 후보가 통째로 버려진다(처음 0/60이 그랬다) -
# 전부 curve_path_log.jsonl에서 직접 잰 값이다:
#   인접 앵커 간격 중앙 134mm / 제어점 중점이탈 중앙 98mm
#   반지름 340~518mm(중앙 417) / 고도각 41.6~82.6°(중앙 65.1)
REACH_MIN_MM, REACH_MAX_MM = 330.0, 520.0
ELEV_RANGE_DEG = (42.0, 82.0)    # ★ 실측 고도각 범위. 팔이 대체로 위를 향한다.
ANCHOR_STEP_MM = (100.0, 170.0)  # 인접 앵커 간격

# 제어점을 앵커 간격의 몇 배까지 밀지. **절대거리가 아니라 비율**이다.
#
# 왜 실측 중앙값(98mm)보다 작게 잡는가:
# `bezier_chain_eval`에서 t=1(제어점 인덱스)의 실제 곡선 위치는 제어점이
# 아니라 베지어 중점(0.25*p0+0.5*pc+0.25*p2)이다. 그런데 `build_orientations`는
# 그 자리의 목표 자세를 **제어점 위치에서** 구한다 - 로봇이 실제로 가지 않는
# 지점의 자세가 스플라인 기준점이 되는 구조다. 그래서 제어점을 멀리 밀수록
# 도달 불가능한 자세가 요구되고 IK가 깨진다(2판에서 IK 미도달 47/60의 주원인).
# 기존 검증(run_curve_check)도 똑같이 동작하므로 이건 고칠 대상이 아니라
# 맞춰야 할 제약이다. 실측 곡선이 98mm로도 통과하는 건 사람이 검증을 통과할
# 때까지 드래그로 고친 결과지, 자동 생성이 따라할 수 있는 조건이 아니다.
CTRL_OFFSET_FRAC = (0.10, 0.40)

# [핵심] 앵커들을 좁은 각도 구역 안에 모은다.
# 0/60 실패의 주원인이 이것이었다 - 앵커가 사방에 흩어지면 natural_pose_at이
# 앵커마다 전혀 다른 자세를 주고, 그 사이를 RotationSpline으로 이으면 중간에
# 도달 불가능한 자세가 생겨 IK가 깨진다(앵커는 닿는데 중간이 안 닿는 형태).
# 실제 곡선들이 한 구역 안에 머무는 것도 사람이 검증 통과할 때까지 고친 결과다.
AZIMUTH_SPAN_DEG = 45.0          # 곡선 전체가 차지하는 방위각 폭
N_ANCHORS = 4                    # 앵커 4개 = 구간 3개

# 첫 앵커의 방위각 범위. --az-min/--az-max로 좁힐 수 있지만, 보통은 쓸 일이
# 없다 - 출발점은 --j3-min/--j3-max와 viable_start_cells()로 고르는 편이 낫다.
#
# [기각된 가설, 기록용] 기존 6종에서는 첫 앵커 방위각 약 90°를 경계로 J3
# 부호가 정확히 갈려서 분기 경계처럼 보였다. 그러나 --scan-start로 격자
# 1620점을 훑어보니 >90°에서 양(+)이 16/23, 0~90°에서 음(-)이 131/148로
# 경계가 깨끗하지 않았고, +120°는 도달 가능 점이 0인데 +110/+130은 있는
# 식으로 영역이 끊겨 있었다. **곡선 6개로 본 분리는 표본이 적어서 생긴
# 착시였다.** 각도 규칙으로 J3 자세를 통제하려 하지 말 것.
AZIMUTH_START_RANGE = (-180.0, 180.0)


def _sph_to_xyz(r, az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([r * math.cos(el) * math.cos(az),
                     r * math.cos(el) * math.sin(az),
                     r * math.sin(el)])


def make_candidate(rng, start_cell=None):
    """앵커/제어점이 번갈아 오는 점 목록을 만든다 (짝수=앵커, 홀수=제어점).

    무작위 박스가 아니라 **좁은 방위각 구역 안의 랜덤워크**로 만든다 - 위
    AZIMUTH_SPAN_DEG 주석 참고. 앵커를 구면좌표(반지름/방위각/고도각)로 잡으면
    "팔이 뻗은 방향"이 직접 통제되므로, 인접 앵커의 자연자세가 비슷해져
    자세 보간이 깨지지 않는다.

    start_cell: viable_start_cells()가 준 출발점. 주면 첫 앵커를 거기에
    고정한다 - 정렬자세에서 출발 가능한 것이 이미 확인된 자리라 출발점
    탈락(전체 실패의 대부분이었다)이 원천적으로 사라진다.
    """
    if start_cell is not None:
        az0, el0 = start_cell["az"], start_cell["el"]
        r = start_cell["r"]
    else:
        az0 = rng.uniform(*AZIMUTH_START_RANGE)      # 이 곡선이 놓일 방향
        el0 = rng.uniform(*ELEV_RANGE_DEG)           # 이 곡선의 대략적 높이
        r = rng.uniform(REACH_MIN_MM, REACH_MAX_MM)
    span = AZIMUTH_SPAN_DEG

    anchors = []
    az, el = az0, el0
    for i in range(N_ANCHORS):
        for _ in range(40):                          # 조건 맞을 때까지 재시도
            if i == 0:
                cand_az, cand_el, cand_r = az, el, r
            else:
                # 한 걸음이 ANCHOR_STEP_MM쯤 되도록 각도 증분을 역산한다.
                step = rng.uniform(*ANCHOR_STEP_MM)
                d_ang = math.degrees(step / max(r, 1.0))
                phi = rng.uniform(0.0, 360.0)
                cand_az = az + d_ang * math.cos(math.radians(phi))
                cand_el = el + d_ang * math.sin(math.radians(phi))
                cand_r = float(np.clip(r + rng.normal(0.0, 25.0),
                                       REACH_MIN_MM, REACH_MAX_MM))
            if abs(cand_az - az0) > span / 2.0:      # 구역 밖이면 다시
                continue
            if not (ELEV_RANGE_DEG[0] <= cand_el <= ELEV_RANGE_DEG[1]):
                continue
            az, el, r = cand_az, cand_el, cand_r
            anchors.append(_sph_to_xyz(cand_r, cand_az, cand_el))
            break
        else:
            return None, None                        # 이 후보는 포기

    if len(anchors) < N_ANCHORS:
        return None, None

    pts, roles = [], []
    for i, a in enumerate(anchors):
        if i > 0:
            gap = float(np.linalg.norm(a - anchors[i - 1]))
            mid = (anchors[i - 1] + a) / 2.0
            d = rng.normal(0.0, 1.0, 3)
            d /= max(np.linalg.norm(d), 1e-9)
            ctrl = mid + d * gap * rng.uniform(*CTRL_OFFSET_FRAC)
            # 제어점도 도달권 안에 있어야 자세 보간이 안정적이다.
            rc = np.linalg.norm(ctrl)
            if rc > REACH_MAX_MM:
                ctrl *= REACH_MAX_MM / rc
            elif rc < REACH_MIN_MM:
                ctrl *= REACH_MIN_MM / max(rc, 1e-9)
            pts.append(ctrl)
            roles.append(False)
        pts.append(a)
        roles.append(True)
    return [list(map(float, p)) for p in pts], roles


def trace_curve(pts_xyz, roles, q_seed, step_mm=None):
    """**run_curve_check와 같은 알고리즘**으로 곡선을 훑으면서 관절각을 모은다.

    run_curve_check는 통과여부만 돌려주므로 관절각을 볼 수 없어, 같은 절차를
    따라가되 각 웨이포인트의 해를 모아서 반환한다. 자세 보간은 그쪽과 동일하게
    앵커 자세 -> RotationSpline(없으면 Slerp) 전역 보간이다 - 여기가 어긋나면
    실제 실행과 다른 자세를 재게 되어 J3 분포 자체가 틀린다(§6.3과 같은 함정).

    step_mm: 웨이포인트 간격. None이면 STEP_MM(실제 실행값). 거친 선별에서는
    COARSE_STEP_MM을 넘겨 싸게 걸러낸다.

    반환: (관절각 배열 (n,6) 도 단위, 실패사유 또는 None)
    """
    from scipy.spatial.transform import Rotation, Slerp
    try:
        from scipy.spatial.transform import RotationSpline
        has_spline = True
    except ImportError:
        RotationSpline = None
        has_spline = False

    if step_mm is None:
        step_mm = STEP_MM

    n = len(pts_xyz)
    arr = np.asarray(pts_xyz, dtype=float)
    xs, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]
    ts = np.arange(n, dtype=float)

    orientations, bad_i = build_orientations(pts_xyz, roles, q_seed)
    if orientations is None:
        return None, f"{bad_i + 1}번 앵커에 도달 자세 없음"

    # [출발점 사전검사용] 점이 하나면 보간할 게 없다 - Slerp/RotationSpline은
    # 회전을 2개 이상 요구하므로 그쪽으로 보내면 터진다. 그 한 점을 정렬자세
    # 시드에서 실제로 풀 수 있는지만 확인하고 끝낸다.
    if n == 1:
        tgt = np.asarray(pts_xyz[0], dtype=float)
        if tgt[2] < DESK_SAFETY_MARGIN_MM:
            return None, "책상 충돌 [0% 지점]"
        sol, pe, oe = solve_pose_ik(list(q_seed), tgt, orientations[0],
                                    pos_tol_mm=POS_TOL_MM, orient_tol_deg=ORIENT_TOL_DEG)
        if sol is None:
            return None, f"IK 미도달(위치{pe:.1f}mm/자세{oe:.1f}°) [0% 지점]"
        ok, bad = within_joint_limits(sol)
        if not ok:
            return None, f"J{bad + 1} 관절한계 [0% 지점]"
        if check_self_collision(sol):
            return None, "자가충돌 [0% 지점]"
        return np.array([[math.degrees(sol[i]) for i in _ACTIVE]]), None

    R_ctrl = Rotation.from_matrix(np.stack(orientations))
    if has_spline and n >= 3:
        _ri = RotationSpline(ts, R_ctrl)
        def _R_at(t_):
            return _ri(t_).as_matrix()
    else:
        _sl = Slerp(ts, R_ctrl)
        def _R_at(t_):
            return _sl(np.clip(t_, ts[0], ts[-1])).as_matrix()

    t_probe = np.linspace(0.0, float(n - 1), 4000)
    px, py, pz = bezier_chain_eval(xs, ys, zs, t_probe)
    P_probe = np.stack([px, py, pz], axis=1)
    tq = resample_by_chord(t_probe, P_probe, step_mm)
    sx, sy, sz = bezier_chain_eval(xs, ys, zs, tq)
    R_all = _R_at(tq)

    q = list(q_seed)
    out = []
    n_wp = len(tq)
    for k in range(n_wp):
        where = f" [{100.0 * k / max(n_wp - 1, 1):.0f}% 지점]"
        tgt = np.array([sx[k], sy[k], sz[k]])
        if tgt[2] < DESK_SAFETY_MARGIN_MM:
            return None, "책상 충돌" + where
        sol, pe, oe = solve_pose_ik(q, tgt, R_all[k],
                                    pos_tol_mm=POS_TOL_MM, orient_tol_deg=ORIENT_TOL_DEG)
        if sol is None:
            return None, f"IK 미도달(위치{pe:.1f}mm/자세{oe:.1f}°)" + where
        ok, bad = within_joint_limits(sol)
        if not ok:
            return None, f"J{bad + 1} 관절한계" + where
        cond = jacobian_condition_number(analytic_jacobian_6d(sol)[:3, :])
        if cond > CONDITION_NUMBER_MAX:
            return None, "특이점" + where
        if check_self_collision(sol):
            return None, "자가충돌" + where
        q = sol
        out.append([math.degrees(sol[i]) for i in _ACTIVE])
    return np.array(out), None


def predict_j3_bias(q_deg):
    """§50 중력처짐 모델이 이 곡선에 대해 예측하는 J3 평균 편향(도).

    err_J3 = a*sin(θ2+θ3) + b 를 실제로 지나가는 웨이포인트마다 계산해
    평균낸다 - 이게 "보정을 켜면 없어질 것으로 예상되는 편향"이다.
    """
    theta23 = np.radians(q_deg[:, 1] + q_deg[:, 2])
    return float(np.mean(_J3_GRAVITY_A * np.sin(theta23) + _J3_GRAVITY_B))


def fingerprint(pts_xyz, roles):
    """path_editor와 동일한 8자리 지문 - 같은 좌표면 같은 값이 나와야 한다."""
    src = "|".join(
        repr((round(p[0], 3), round(p[1], 3), round(p[2], 3),
              "anchor" if r else "ctrl", None))
        for p, r in zip(pts_xyz, roles)
    )
    return hashlib.md5(src.encode("utf-8")).hexdigest()[:8]


def perturbed_from_existing(rng, jitter_mm=70.0):
    """기존 성공 곡선 하나를 골라 모든 점을 흔들어 새 곡선을 만든다.

    무작위 생성이 잘 안 될 때의 대안(--from-existing). 기존 곡선은 이미 자세
    보간이 성립하는 형태라 통과율이 훨씬 높다. 흔드는 폭이 크므로 모양도
    J3 자세분포도 원본과 달라지고 지문도 당연히 달라진다 - 다만 **완전히
    독립적인 곡선은 아니므로**, 무작위 생성으로 얻은 곡선이 하나라도 있으면
    그쪽을 우선하는 편이 검증으로서 강하다.
    """
    recs = [r for r in curve_log.load_log()
            if str(r.get("label", "")).startswith("curve_") and "stop" not in r["label"]]
    if not recs:
        return None, None
    rec = recs[rng.integers(len(recs))]
    pts, roles = [], []
    for p in rec["points"]:
        v = np.array(p[:3], dtype=float) + rng.normal(0.0, jitter_mm, 3)
        r = np.linalg.norm(v)
        if r > REACH_MAX_MM:
            v *= REACH_MAX_MM / r
        elif r < REACH_MIN_MM:
            v *= REACH_MIN_MM / max(r, 1e-9)
        pts.append([float(v[0]), float(v[1]), float(v[2])])
        roles.append(p[3] == "anchor")
    return pts, roles


# §50.2에 기록된 곡선별 J3 평균자세 - 이 스크립트가 재는 값이 여기와 맞는지
# 대조하는 데 쓴다(--check-known). 안 맞으면 trace_curve가 실제 실행과 다른
# 것을 재고 있다는 뜻이므로, 생성기가 뽑아낸 숫자도 전부 못 믿는다.
KNOWN_J3_MEAN = {
    "e433b126": +70.4, "ea5354ed": +63.6, "78dfad65": -38.0,
    "fff98c63": -57.8, "fc043529": -74.4, "e651e55d": -82.7,
}


def check_known():
    """저장된 기존 곡선들을 trace_curve로 훑어 §50.2 값과 대조한다.

    **생성기를 믿기 전에 먼저 통과해야 하는 검사다.** 여기서 값이 어긋나면
    자세 보간이든 관절 인덱스든 어딘가가 실제 실행과 다르다는 뜻이고, 그러면
    생성된 곡선의 'J3 평균/예측 편향'도 전부 틀린 값이다.
    """
    q_seed = ik_seed_q()
    recs = curve_log.load_log()
    print(f"{'곡선':<12}{'§50.2':>9}{'측정값':>10}{'차이':>9}   비고")
    n_ok = n_bad = 0
    for fp, expect in KNOWN_J3_MEAN.items():
        rec = next((r for r in recs if fp in str(r.get("label", ""))), None)
        if rec is None:
            print(f"{fp:<12}{expect:>+9.1f}{'---':>10}{'---':>9}   로그에 없음")
            continue
        pts = [p[:3] for p in rec["points"]]
        roles = [p[3] == "anchor" for p in rec["points"]]
        q_deg, reason = trace_curve(pts, roles, q_seed)
        if q_deg is None:
            print(f"{fp:<12}{expect:>+9.1f}{'실패':>10}{'---':>9}   {reason}")
            n_bad += 1
            continue
        got = float(q_deg[:, 2].mean())
        diff = got - expect
        mark = "일치" if abs(diff) < 5.0 else "★ 불일치"
        if abs(diff) < 5.0:
            n_ok += 1
        else:
            n_bad += 1
        print(f"{fp:<12}{expect:>+9.1f}{got:>+10.1f}{diff:>+9.1f}   {mark}")

    print()
    if n_bad == 0 and n_ok > 0:
        print(f"✅ {n_ok}개 전부 §50.2와 일치 - trace_curve가 실제 실행과 같은 것을")
        print("   재고 있습니다. 생성된 곡선의 J3/예측편향도 믿을 수 있습니다.")
    else:
        print(f"⚠️ 불일치/실패 {n_bad}개 - 생성기가 뽑은 숫자를 쓰면 안 됩니다.")
        print("   trace_curve의 자세 보간이나 관절 인덱스(_ACTIVE)부터 확인하세요.")
    return n_bad == 0


def scan_start(rng=None):
    """정렬자세에서 **출발 가능한** (방위각, 고도각, 반지름) 영역을 직접 훑는다.

    출발점 탈락이 97/100까지 나오면서 "어디서 시작할 수 있는가"를 추측으로
    좁히는 게 무의미해졌다. 점 하나당 IK 1회면 되므로 격자 전체를 훑어도
    금방 끝난다 - 곡선을 만들기 전에 지도를 먼저 그린다.

    덤으로 §51의 분기 가설(첫 앵커 방위각 ~90°에서 J3 부호가 갈린다)을
    곡선 생성과 무관하게, 훨씬 많은 표본으로 직접 확인할 수 있다.
    """
    q_seed = ik_seed_q()
    azs = np.arange(-180, 180, 10.0)
    els = np.arange(42, 83, 5.0)
    rads = np.arange(340, 521, 45.0)

    per_az = {}
    total = ok_total = 0
    for az in azs:
        hits = []
        for el in els:
            for r in rads:
                total += 1
                p = _sph_to_xyz(float(r), float(az), float(el))
                q_deg, _ = trace_curve([[float(p[0]), float(p[1]), float(p[2])]],
                                       [True], q_seed)
                if q_deg is not None:
                    ok_total += 1
                    hits.append((float(el), float(r), predict_j3_bias(q_deg)))
        per_az[float(az)] = hits

    print(f"격자 {total}점 중 출발 가능 {ok_total}점 "
          f"(방위각 {len(azs)} x 고도각 {len(els)} x 반지름 {len(rads)})\n")
    print(f"{'방위각':>7}{'가능':>6}{'예측편향 평균':>13}{'예측편향 범위':>20}   고도각 / 반지름")
    for az in azs:
        hits = per_az[float(az)]
        if not hits:
            print(f"{az:>+7.0f}{0:>6}{'---':>13}{'---':>20}   (없음)")
            continue
        bv = np.array([h[2] for h in hits])
        el_v = [h[0] for h in hits]
        r_v = [h[1] for h in hits]
        print(f"{az:>+7.0f}{len(hits):>6}{bv.mean():>+12.3f}°"
              f"{bv.min():>+11.3f}~{bv.max():>+8.3f}"
              f"   {min(el_v):.0f}~{max(el_v):.0f}° / {min(r_v):.0f}~{max(r_v):.0f}mm")

    allb = np.array([h[2] for hs in per_az.values() for h in hs])
    print()
    if len(allb):
        print(f"전체 도달 가능 출발점의 예측편향: {allb.min():+.3f} ~ {allb.max():+.3f}° "
              f"(모델상 가능 범위 -0.360 ~ +0.584)")
    print("\n위 표에서 원하는 예측편향 대역을 --bias-min/--bias-max로 지정하면")
    print("그 영역에서 출발하는 곡선만 생성합니다.")


def viable_start_cells(bias_range=None, fa_range=None, verbose=True):
    """정렬자세에서 출발 가능한 격자점을 모아 (좌표, 예측편향) 목록으로 준다.

    [왜 이렇게 바뀌었나 - 두 번의 방향 전환]
    (1) 처음엔 방위각/고도각/반지름 범위를 실측 곡선 통계에서 추측해 넣었는데,
        그건 "사람이 그려서 통과한 곡선이 어디 있었나"지 "어디서 출발
        가능한가"가 아니었다. 방위각을 95~160으로 옮기자 출발점 탈락이
        97/100까지 났다. 그래서 격자를 직접 훑는 방식으로 바꿨다.
    (2) 그 다음엔 J3 자세로 출발점을 골랐는데, 이것도 틀린 변수였다.
        §50 모델의 입력은 J3가 아니라 **팔뚝 절대각(θ2+θ3)**이다. 실제로
        J3 평균 +82.5°인 생성 곡선의 예측편향이 -0.033인 반면, J3 평균
        +70.4°인 기존 e433b126은 실측편향 +0.544였다 - J3가 비슷해도 θ2가
        다르면 sin(θ2+θ3)의 부호까지 뒤집힌다. J3로 고르면 예측편향이
        -0.356~+0.058에 갇혀서, 모델을 가장 세게 시험할 수 있는 양(+) 편향
        영역(e433b126의 +0.544 같은)이 통째로 빠진다.

    그래서 통제 변수를 **예측 편향 자체**로 잡는다. 곧바로 해석되는 값이고
    (그 곡선에서 보정이 없앨 것으로 예상되는 편향), 모델의 입력과 정확히
    같은 것을 통제한다.

    bias_range: (lo, hi) 이 범위의 예측 편향을 가진 출발점만 남긴다.
      모델상 가능한 전 범위는 -0.360 ~ +0.584도.
    """
    q_seed = ik_seed_q()
    cells = []
    for az in np.arange(-180, 180, 10.0):
        for el in np.arange(42, 83, 5.0):
            for r in np.arange(340, 521, 45.0):
                p = _sph_to_xyz(float(r), float(az), float(el))
                q_deg, _ = trace_curve([[float(p[0]), float(p[1]), float(p[2])]],
                                       [True], q_seed)
                if q_deg is None:
                    continue
                bias = predict_j3_bias(q_deg)
                fa = float(q_deg[0, 1] + q_deg[0, 2])
                if bias_range is not None and not (bias_range[0] <= bias <= bias_range[1]):
                    continue
                # [10차, §51] 팔뚝각 직접 지정. 예측편향은 sin이라 +60°와 +120°가
                # 같은 값이 나온다 - 특정 자세대를 겨냥할 땐 각도로 걸러야 한다.
                if fa_range is not None and not (fa_range[0] <= fa <= fa_range[1]):
                    continue
                cells.append({"xyz": [float(p[0]), float(p[1]), float(p[2])],
                              "az": float(az), "el": float(el), "r": float(r),
                              "j3": float(q_deg[0, 2]),
                              "fa": fa,
                              "bias": bias})
    if verbose:
        if cells:
            bv = np.array([c["bias"] for c in cells])
            fv = np.array([c["fa"] for c in cells])
            print(f"  출발 가능 격자점 {len(cells)}개 "
                  f"(팔뚝각 {fv.min():+.1f}~{fv.max():+.1f}°, "
                  f"예측편향 {bv.min():+.3f}~{bv.max():+.3f}°)")
        else:
            print("  조건에 맞는 출발점이 없습니다.")
    return cells


def search(n_try, rng, verbose=True, from_existing=False, start_cells=None,
           fa_range=None):
    """후보를 만들어보고 통과한 것들의 J3 통계를 모은다.

    2단계로 거른다: 거친 간격으로 싸게 훑어 실패할 곡선을 먼저 떨구고,
    살아남은 것만 실제 실행 간격(STEP_MM)으로 정밀 재검증한다. 최종 판정은
    항상 정밀 쪽이므로 기준이 느슨해지지는 않는다.
    """
    q_seed = ik_seed_q()
    found, n_coarse_fail, n_fine_fail, n_skip = [], 0, 0, 0
    n_start_fail = 0
    n_fa_out = 0
    reasons = {}
    fail_at = []          # IK가 깨진 지점(곡선의 몇 %) - 원인 좁히기용

    def _note(reason):
        key = reason.split("(")[0].split(" [")[0]
        reasons[key] = reasons.get(key, 0) + 1
        if "[" in reason and "%" in reason:
            try:
                fail_at.append(float(reason.split("[")[1].split("%")[0]))
            except (ValueError, IndexError):
                pass

    for i in range(n_try):
        if from_existing:
            pts, roles = perturbed_from_existing(rng)
        elif start_cells:
            cell = start_cells[int(rng.integers(len(start_cells)))]
            pts, roles = make_candidate(rng, start_cell=cell)
        else:
            pts, roles = make_candidate(rng)
        if pts is None:
            n_skip += 1
            continue

        # [1차 관문] 출발점만 먼저 본다. 실패의 36/44가 곡선 0-20% 구간에
        # 몰려 있었는데, 원인은 trace_curve(그리고 run_curve_check)가 정렬자세
        # 시드에서 국소 반복으로 푸는 반면 build_orientations는 그리드 시드
        # 폴백까지 쓴다는 불일치다 - 시작점의 자연자세가 정렬자세와 다른 IK
        # 분기에 있으면 첫 웨이포인트부터 깨진다. 이걸 IK 1회로 걸러내면
        # 거친선별(10여 회)까지 갈 후보가 크게 줄어 같은 시도횟수로 훨씬
        # 많은 곡선을 훑을 수 있다.
        q_start, reason = trace_curve(pts[:1], [True], q_seed, step_mm=COARSE_STEP_MM)
        if q_start is None:
            n_start_fail += 1
            _note("출발점: " + reason)
            continue

        q_coarse, reason = trace_curve(pts, roles, q_seed, step_mm=COARSE_STEP_MM)
        if q_coarse is None:
            n_coarse_fail += 1
            _note(reason)
            continue

        q_deg, reason = trace_curve(pts, roles, q_seed)      # 정밀 = 실제 실행 간격
        if q_deg is None:
            n_fine_fail += 1
            _note("정밀재검증: " + reason)
            continue

        j3 = q_deg[:, 2]
        fa = q_deg[:, 1] + q_deg[:, 2]        # 팔뚝 절대각 - §50 모델의 실제 입력
        # 출발점 팔뚝각과 곡선 전체 평균은 다르다 - 곡선을 따라 자세가 바뀌기
        # 때문. 겨냥한 자세대의 곡선을 얻으려면 여기서 한 번 더 걸러야 한다.
        if fa_range is not None and not (fa_range[0] <= float(fa.mean()) <= fa_range[1]):
            n_fa_out += 1
            continue
        first = np.array(pts[0], dtype=float)
        found.append({
            "pts": pts, "roles": roles,
            "fp": fingerprint(pts, roles),
            "n_wp": len(q_deg),
            "az0": float(np.degrees(np.arctan2(first[1], first[0]))),
            "j3_mean": float(j3.mean()),
            "fa_mean": float(fa.mean()),
            "j3_lo": float(j3.min()), "j3_hi": float(j3.max()),
            "pred_bias": predict_j3_bias(q_deg),
        })
        if verbose:
            f = found[-1]
            print(f"  [{len(found):2d}] {f['fp']}  예측편향 {f['pred_bias']:+.3f}°  "
                  f"팔뚝각 {f['fa_mean']:+7.1f}°  J3평균 {f['j3_mean']:+7.1f}°  "
                  f"웨이포인트 {f['n_wp']}")

    if verbose:
        print(f"\n  시도 {n_try}회 -> 통과 {len(found)}개 "
              f"(출발점 탈락 {n_start_fail}, 거친선별 탈락 {n_coarse_fail}, "
              f"정밀재검증 탈락 {n_fine_fail}, 팔뚝각 범위밖 {n_fa_out}, "
              f"생성실패 {n_skip})")
        if reasons:
            print("  실패 사유:", ", ".join(f"{k} {v}회" for k, v in
                                        sorted(reasons.items(), key=lambda x: -x[1])))
        if fail_at:
            fa = np.array(fail_at)
            hist = np.histogram(fa, bins=[0, 20, 40, 60, 80, 100.01])[0]
            print("  깨진 지점 분포(곡선 위치별): " +
                  " ".join(f"{lo}-{hi}%:{c}" for (lo, hi), c in
                           zip([(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)], hist)))
            print(f"    -> 한 구간에 몰려 있으면 그 부근 형태가 문제, "
                  f"고르게 퍼져 있으면 곡선 전반이 과하게 휜 것 "
                  f"(CTRL_OFFSET_FRAC를 줄여볼 것)")
    return found


def pick_spread(found, k=3):
    """예측 편향이 서로 최대한 멀리 떨어진 k개를 고른다.

    [주의] 예전엔 J3 평균자세로 벌렸는데 그건 틀린 기준이었다 - §50 모델의
    입력은 J3가 아니라 팔뚝 절대각(θ2+θ3)이고, J3가 비슷해도 θ2가 다르면
    예측편향이 정반대가 된다. 검증에서 벌어져야 하는 건 **모델이 내놓는
    예측값**이다. 예측 범위가 좁으면 "모델이 맞다"와 "모델이 상수를 내놓을
    뿐이다"를 구분할 수 없다.
    """
    if len(found) <= k:
        return list(found)
    pool = sorted(found, key=lambda f: f["pred_bias"])
    picked = [pool[0], pool[-1]]           # 양 극단 먼저
    while len(picked) < k:
        best, best_d = None, -1.0
        for c in pool:
            if c in picked:
                continue
            d = min(abs(c["pred_bias"] - p["pred_bias"]) for p in picked)
            if d > best_d:
                best, best_d = c, d
        if best is None:
            break
        picked.append(best)
    return sorted(picked, key=lambda f: f["pred_bias"])


def main():
    ap = argparse.ArgumentParser(description="§51 검증용 새 곡선 생성기")
    ap.add_argument("--tries", type=int, default=60, help="후보 시도 횟수 (기본 60)")
    ap.add_argument("--pick", type=int, default=3, help="최종 선택 개수 (기본 3)")
    ap.add_argument("--seed", type=int, default=20260951, help="난수 시드")
    ap.add_argument("--save", action="store_true",
                    help="고른 곡선을 curve_path_log.jsonl에 저장 (GUI에서 불러오기 가능)")
    ap.add_argument("--from-existing", action="store_true",
                    help="무작위 생성 대신 기존 곡선을 크게 흔들어 만든다 (통과율 높음)")
    ap.add_argument("--check-known", action="store_true",
                    help="기존 곡선의 J3를 §50.2와 대조해 이 스크립트 자체를 검증한다")
    ap.add_argument("--az-min", type=float, default=None,
                    help="첫 앵커 방위각 하한(도). 90 이상이면 J3 양(+) 자세대 예상 - 위 가설 참고")
    ap.add_argument("--az-max", type=float, default=None,
                    help="첫 앵커 방위각 상한(도)")
    ap.add_argument("--scan-start", action="store_true",
                    help="정렬자세에서 출발 가능한 방위각/고도각/반지름 영역을 훑어 표로 보여준다")
    ap.add_argument("--bias-min", type=float, default=None,
                    help="출발점의 예측 편향 하한(도). 모델상 가능 범위는 -0.360~+0.584")
    ap.add_argument("--bias-max", type=float, default=None,
                    help="출발점의 예측 편향 상한(도)")
    ap.add_argument("--fa-min", type=float, default=None,
                    help="팔뚝각(θ2+θ3) 하한(도). 예측편향은 sin이라 +60°와 +120°를 "
                         "구분 못 하므로, 특정 자세대를 겨냥할 땐 이쪽을 쓴다")
    ap.add_argument("--fa-max", type=float, default=None,
                    help="팔뚝각(θ2+θ3) 상한(도)")
    args = ap.parse_args()

    if args.scan_start:
        print("출발 가능 영역 스캔 - 점 하나당 IK 1회, 곡선 생성 전에 지도를 그린다\n")
        scan_start()
        sys.exit(0)

    if args.az_min is not None or args.az_max is not None:
        lo = AZIMUTH_START_RANGE[0] if args.az_min is None else args.az_min
        hi = AZIMUTH_START_RANGE[1] if args.az_max is None else args.az_max
        if lo >= hi:
            print(f"--az-min({lo})이 --az-max({hi}) 이상입니다.")
            sys.exit(1)
        globals()["AZIMUTH_START_RANGE"] = (lo, hi)

    if args.check_known:
        print("기존 곡선 대조 - trace_curve가 실제 실행과 같은 것을 재는지 확인\n")
        sys.exit(0 if check_known() else 1)

    mode = "기존 곡선 변형" if args.from_existing else "무작위 생성"
    print(f"후보 탐색 중... ({mode}, 시도 {args.tries}회, 시드 {args.seed})")
    print(f"  간격 {STEP_MM:.2f}mm / 위치허용 {POS_TOL_MM:.3f}mm / 자세허용 {ORIENT_TOL_DEG}°")
    print("  ※ IK를 웨이포인트마다 돌리므로 수십 초 걸릴 수 있습니다.\n")
    rng = np.random.default_rng(args.seed)

    # 출발점을 먼저 확정한다 - 정렬자세에서 출발 가능한 것이 확인된 자리에서만
    # 곡선을 시작하면, 실패의 대부분이었던 출발점 탈락이 사라진다.
    start_cells = None
    far = None
    if args.fa_min is not None or args.fa_max is not None:
        far = (args.fa_min if args.fa_min is not None else -1e9,
               args.fa_max if args.fa_max is not None else 1e9)
        print(f"  팔뚝각 조건: {far[0]:+.0f} ~ {far[1]:+.0f}°")
    if not args.from_existing:
        br = None
        if args.bias_min is not None or args.bias_max is not None:
            br = (args.bias_min if args.bias_min is not None else -1e9,
                  args.bias_max if args.bias_max is not None else 1e9)
            print(f"  출발점 예측편향 조건: {br[0]:+.3f} ~ {br[1]:+.3f}°")
        print("  출발 가능 격자 스캔 중...")
        start_cells = viable_start_cells(bias_range=br, fa_range=far)
        if not start_cells:
            print("\n조건에 맞는 출발점이 없습니다 - 조건을 넓히거나")
            print("--scan-start로 실제 도달 가능한 범위를 먼저 확인하세요.")
            sys.exit(1)
        print()

    found = search(args.tries, rng, from_existing=args.from_existing,
                   start_cells=start_cells, fa_range=far)

    found = search(args.tries, rng, from_existing=args.from_existing,
                   start_cells=start_cells)

    if not found and not args.from_existing:
        print("\n무작위 생성으로는 통과한 곡선이 없습니다 - 기존 곡선 변형으로 재시도합니다.")
        print("(기존 곡선은 자세 보간이 이미 성립하는 형태라 통과율이 높습니다)\n")
        found = search(args.tries, rng, from_existing=True)

    if not found:
        print("\n통과한 곡선이 없습니다 - --tries를 늘리거나 AZIMUTH_SPAN_DEG를 줄여보세요.")
        sys.exit(1)

    picked = pick_spread(found, args.pick)

    print(f"\n{'='*78}")
    print(f"  선택된 {len(picked)}개 - 예측 편향이 서로 벌어지도록 골랐습니다")
    print(f"{'='*78}")
    print(f"{'지문':<10}{'예측 편향':>11}{'팔뚝각':>10}{'J3평균':>10}{'웨이포인트':>10}")
    for f in picked:
        print(f"{f['fp']:<10}{f['pred_bias']:>+10.3f}°{f['fa_mean']:>+9.1f}°"
              f"{f['j3_mean']:>+9.1f}°{f['n_wp']:>10}")

    if found:
        bv = np.array([f["pred_bias"] for f in found])
        print(f"\n  통과 곡선의 예측편향 분포: {bv.min():+.3f} ~ {bv.max():+.3f}° "
              f"(모델상 가능 범위 -0.360 ~ +0.584)")
        if bv.max() - bv.min() < 0.2:
            print("   -> ★ 예측 폭이 좁습니다. 이대로 실기 검증하면 '모델이 맞다'와")
            print("      '모델이 그냥 상수를 내놓는다'를 구분할 수 없습니다.")
            print("      --bias-min/--bias-max로 반대쪽 영역을 따로 확보하세요.")

    print(f"\n  [기존 6종 참고 - §50.2] e433b126 +70.4° / ea5354ed +63.6° /")
    print(f"   78dfad65 -38.0° / fff98c63 -57.8° / fc043529 -74.4° / e651e55d -82.7°")
    print(f"\n  ⚠️ 위 '예측 편향'을 지금 기록해 두십시오. 실행 후에 계산하면")
    print(f"     사후해석이 되어 §51 검증의 의미가 사라집니다.")

    if args.save:
        for f in picked:
            snaps = [(p[0], p[1], p[2], "anchor" if r else "ctrl", None)
                     for p, r in zip(f["pts"], f["roles"])]
            curve_log.save_curve(
                label=f"gen_{f['fp']} (§51후보)",
                points_snapshots=snaps,
                extra={"generated_by": "mycobot_curve_generator.py",
                       "seed": args.seed,
                       "j3_mean_deg": round(f["j3_mean"], 2),
                       "forearm_angle_mean_deg": round(f["fa_mean"], 2),
                       "j3_lo_deg": round(f["j3_lo"], 2),
                       "j3_hi_deg": round(f["j3_hi"], 2),
                       "predicted_j3_bias_deg": round(f["pred_bias"], 4)},
            )
        print(f"\n✅ {len(picked)}개를 curve_path_log.jsonl에 저장했습니다 - GUI의")
        print("   '저장된 경로 불러오기'에서 gen_* 라벨을 고르면 됩니다.")
        print("   불러온 뒤 반드시 '경로 검증'을 눌러 GUI 쪽에서도 통과하는지 확인하세요.")
    else:
        print("\n  (저장하려면 --save 를 붙여 다시 실행하세요)")


if __name__ == "__main__":
    main()
