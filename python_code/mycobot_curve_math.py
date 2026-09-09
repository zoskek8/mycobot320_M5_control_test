"""
곡선 수학 + 경로검증 로직 (6차 분할)
================================================================================

mycobot_path_editor.py에서 GUI(self)에 의존하지 않는 부분만 뽑아냈다:
  - 베지어 체인 평가 (bezier_chain_eval)
  - 점 목록 <-> 구간 인덱스 변환 (n_anchors / n_segments / segment_range_for_index)
  - IK 시드, 자세 보간, 현(chord) 균일 재샘플링
  - "실제 검증과 동일한 알고리즘"으로 곡선을 검사하는 run_curve_check와,
    그 위에서 후보 하나를 대입해보는 curve_ok_with_substitution

왜 분리했는가
-------------
파일 하나(mycobot_path_editor.py)가 2800줄을 넘어가면서, 메서드 하나의 `def` 선언
줄이 통째로 빠져도 눈에 안 띄는 사고가 실제로 있었다(6차, `_curve_ok_with_substitution`
버그 - §PROJECT_HANDOFF.md 참고). 이 모듈의 함수들은 전부 **인자로 받은 값만 쓰고
Qt(GUI)나 로봇 연결(self.mc)을 전혀 건드리지 않는다** - 그래서:
  1. GUI 없이(즉 PyQt5 없이도) import해서 단위 테스트를 돌릴 수 있다.
  2. 결과가 입력에만 의존하므로("같은 입력 -> 항상 같은 출력") 실수를 훨씬
     빨리 잡아낼 수 있다 - GUI 상태가 우연히 맞아떨어져야만 드러나는 버그가 없다.
  3. path_editor.py 쪽은 이 함수들을 호출하는 **얇은 위임 메서드**만 남기고,
     실제 로직은 여기 한 곳에만 존재한다(중복 정의로 인한 불일치 위험 제거).

path_editor.py의 PathEditor 클래스는 이 모듈의 함수들을 그대로 감싸는 메서드
(`self._run_curve_check(...)` 등)를 유지한다 - 기존 호출부를 하나도 안 건드려도
되도록 하기 위한 의도적 설계다(안전한 리팩터링의 핵심).

실행속도에 미치는 영향
-----------------------
**없다.** 파일을 나누는 것은 순수하게 코드 조직 문제이고, import된 함수를 호출하는
것과 같은 파일 안의 메서드를 호출하는 것은 파이썬 바이트코드 수준에서 사실상 같은
비용이다(둘 다 이름 조회 후 함수 호출). 체감할 만한 실행속도 차이는 없다.
얻는 건 유지보수성(가독성, 버그 발견 용이성, 부분적 재사용성)이지 성능이 아니다.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
try:
    from scipy.spatial.transform import RotationSpline
    _HAS_ROT_SPLINE = True
except ImportError:      # 구버전 scipy면 기존 Slerp로 자동 폴백
    RotationSpline = None
    _HAS_ROT_SPLINE = False

from mycobot_kinematics import (
    chain, active_indices, ALIGN_ANGLES,
    natural_pose_at, solve_pose_ik, within_joint_limits,
    jacobian_condition_number, analytic_jacobian_6d, check_self_collision,
    DESK_SAFETY_MARGIN_MM, CONDITION_NUMBER_MAX,
)


class PathPoint:
    """경로를 이루는 점.

    role:
      "anchor" — 사용자가 찍은 점. 곡선이 실제로 이 점을 지나고 로봇이 방문한다.
      "ctrl"   — 구간마다 자동으로 하나씩 생기는 2차 베지어 제어점.
                 곡선을 당기기만 하고 지나지는 않는다. 드래그해서 곡선을 휘게 만든다.

    [설계 메모] 예전에는 역할을 '인덱스가 짝수냐 홀수냐'로 추측했는데, 중간에 점을
    끼워 넣으면 뒤쪽 점들의 역할이 통째로 뒤바뀌는 문제가 있었다. 이제 점 자신이
    역할을 들고 다니므로 삽입/삭제와 무관하게 안정적이다.
    """

    def __init__(self, x, y, z, role="anchor"):
        self.x, self.y, self.z = x, y, z
        self.role = role
        self.valid = None  # None=미검증, True=안전, False=위험

    @property
    def is_anchor(self):
        return self.role == "anchor"

    def coord(self):
        return [self.x, self.y, self.z]

    def snapshot(self):
        """[6차] 되돌리기(undo) 스냅샷용 - 이 점의 상태를 순수 값으로 뽑는다."""
        return (self.x, self.y, self.z, self.role, self.valid)

    @staticmethod
    def from_snapshot(t):
        x, y, z, role, valid = t
        p = PathPoint(x, y, z, role=role)
        p.valid = valid
        return p


def bezier_chain_eval(xs, ys, zs, tq):
    """점 배열(인덱스 0..n-1)을 2차 베지어 체인으로 평가한다.

    짝수 인덱스(0,2,4,...) = 앵커: 곡선이 실제로 지나는 점 (로봇이 실제로 방문).
    홀수 인덱스(1,3,5,...) = 제어점: 그 앞뒤 앵커 사이를 당겨 곡선을 휘게 함
        (곡선이 지나지 않음 - De Casteljau 구성에서 P1 역할, 사용자가 준 그림의
        P0-P1-P2 패턴을 그대로 체인으로 확장한 것).

    tq: 0~n-1 범위의 실수(배열). 정수 위치는 항상 그 점 자신과 일치한다.
      n==1: 점 하나뿐 -> 그 점 고정.
      n==2: 앵커 둘뿐, 제어점이 아직 없음 -> 직선.
      n이 짝수(>=4): 마지막 점은 원래 제어점 자리인데 다음 앵커가 없어 짝이
        안 맞는다. 그 마지막 구간만 직선으로 잇는다 (점을 하나 더 찍으면
        정상적인 베지어 구간으로 자동 전환된다).

    반환: (x, y, z) 각각 tq와 같은 길이의 배열.
    """
    tq = np.atleast_1d(np.asarray(tq, dtype=float))
    n = len(xs)
    P = np.stack([np.asarray(xs, dtype=float),
                  np.asarray(ys, dtype=float),
                  np.asarray(zs, dtype=float)], axis=1)   # (n,3)
    out = np.empty((len(tq), 3))

    if n == 1:
        out[:] = P[0]
        return out[:, 0], out[:, 1], out[:, 2]

    if n == 2:
        u = np.clip(tq, 0.0, 1.0)
        out = P[0][None, :] + (P[1] - P[0])[None, :] * u[:, None]
        return out[:, 0], out[:, 1], out[:, 2]

    n_full_segments = (n - 1) // 2   # 완결된 앵커-제어-앵커 구간 개수
    tqc = np.clip(tq, 0.0, float(n - 1))
    full_end = 2.0 * n_full_segments   # 완결 구간이 끝나는 파라미터 위치

    for i, t in enumerate(tqc):
        if n_full_segments > 0 and t <= full_end:
            seg = min(int(t // 2), n_full_segments - 1)
            u = (t - 2 * seg) / 2.0
            u = min(max(u, 0.0), 1.0)
            p0, pc, p2 = P[2 * seg], P[2 * seg + 1], P[2 * seg + 2]
            out[i] = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * pc + u ** 2 * p2
        else:
            # n이 짝수일 때의 마지막 미완결 구간 - 직선으로 처리
            u = t - (n - 2)
            u = min(max(u, 0.0), 1.0)
            out[i] = P[n - 2] + (P[n - 1] - P[n - 2]) * u

    return out[:, 0], out[:, 1], out[:, 2]


def ik_seed_q():
    """검증에 쓸 IK 시드 (항상 고정된 정렬자세).

    [왜 로봇 현재자세를 읽지 않는가]
    예전에는 get_current_q_full(mc)로 '로봇이 지금 서 있는 자세'를 시드로 썼다.
    그런데 solve_pose_ik / natural_pose_at 는 시드에서 출발하는 국소 반복 해법이라
    시드가 바뀌면 수렴하는 해가 달라지고, 심하면 성공/실패 자체가 뒤바뀐다.
    게다가 natural_pose_at 에는 '시드가 이미 목표 근처(POS_TOL_MM=15mm)면 그 자세를
    그대로 채택'하는 분기가 있어서, 실행 직후 로봇이 경로 끝점에 서 있는 상태로
    재검증하면 그 끝점의 목표자세가 통째로 바뀌어 버린다.

    검증은 경로만의 함수여야 하므로 시드를 고정한다. 시작 시 로봇이 실제로
    정렬자세에 있으므로(ALIGN_ANGLES) 물리적으로도 이 시드가 타당하다.
    (기존 폴백이던 전부 0도 자세는 오히려 특이점이라 시드로 부적합했다)
    """
    q = [0.0] * len(chain.links)
    for i, idx in enumerate(active_indices):
        q[idx] = math.radians(ALIGN_ANGLES[i])
    return q


def n_anchors(points):
    return (len(points) + 1) // 2


def n_segments(points):
    return max(0, n_anchors(points) - 1)


def segment_range_for_index(points, idx):
    """점 idx를 움직였을 때 모양이 바뀌는 구간 범위 (seg_lo, seg_hi) 반환 (닫힌 구간).
    앵커는 좌우 두 구간에, 제어점은 자기 구간 하나에만 영향을 준다."""
    n_seg = n_segments(points)
    if n_seg < 1:
        return None
    if points[idx].is_anchor:
        a = idx // 2                      # 앵커 순번
        return max(0, a - 1), min(n_seg - 1, a)
    seg = (idx - 1) // 2                  # 제어점이 속한 구간
    return seg, seg


def build_orientations(pts_xyz, roles, q_seed):
    """검증 1단계와 **동일한 방식**으로 각 점의 목표 자세를 만든다.
    앵커는 natural_pose_at으로 자세를 구하고, 제어점은 직전 자세를 그대로 이어붙인다.
    반환: (회전행렬 리스트, 실패한 앵커 인덱스 또는 None)"""
    orientations = []
    q = list(q_seed)
    for i, (xyz, is_anchor) in enumerate(zip(pts_xyz, roles)):
        qn, Rm = natural_pose_at(list(xyz), q)
        if qn is None:
            if is_anchor:
                return None, i
            orientations.append(orientations[-1] if orientations else np.eye(3))
            continue
        orientations.append(Rm)
        q = qn
    return orientations, None


def resample_by_chord(t_fine, P_fine, base_step):
    """곡선(P_fine, t_fine으로 표현)을 따라 걸으면서, 직전에 찍은 웨이포인트로
    부터의 '직선거리(현)'가 base_step에 도달하는 지점마다 새 웨이포인트를 찍는다.

    [왜 호길이가 아니라 현인가] 로봇은 웨이포인트 사이를 직선으로 움직인다.
    균일해야 하는 건 곡선을 따라간 거리가 아니라 이 직선거리다. 급하게 꺾이는
    구간은 곡선이 안쪽으로 휘어 들어가므로, 호길이가 같아도 현은 짧아진다 -
    그 현이 곧 그 사이클의 실제 이동거리이므로 짧아지면 속도가 그대로 떨어진다.
    이 함수는 결과(현 길이)를 직접 제어하므로 원리적으로 그 노치가 생기지 않는다.

    반환: 선택된 지점들의 t값 배열 (첫 점과 끝점 포함)
    """
    n = len(P_fine)
    idxs = [0.0]
    last_pt = P_fine[0]
    i = 0
    while i < n - 1:
        p_a, p_b = P_fine[i], P_fine[i + 1]
        seg_vec = p_b - p_a
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < 1e-12:
            i += 1
            continue
        if np.linalg.norm(p_b - last_pt) >= base_step:
            # 이 미세 구간 안에 현 길이가 정확히 base_step이 되는 지점이 있다 -
            # 짧은 구간이므로 이분탐색으로 충분히 빠르고 정확하다.
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = (lo + hi) / 2.0
                pm = p_a + seg_vec * mid
                if np.linalg.norm(pm - last_pt) < base_step:
                    lo = mid
                else:
                    hi = mid
            idxs.append(i + hi)
            last_pt = p_a + seg_vec * hi
            # i는 그대로 둔다 - 같은 미세구간 안에서 base_step이 더 들어갈 수 있다
        else:
            i += 1
    # 끝점 포함 - 단, 마지막으로 찍은 점이 이미 끝점과 거의 겹치면 중복 추가하지 않는다
    # (그러면 마지막 현 길이가 0에 가까운 웨이포인트가 하나 남는다)
    if np.linalg.norm(P_fine[-1] - last_pt) > base_step * 0.3:
        idxs.append(float(n - 1))
    return np.interp(np.array(idxs), np.arange(n, dtype=float), t_fine)


def run_curve_check(pts_xyz, roles, q_seed, pos_tol_mm, orient_tol_deg,
                     t_lo=None, t_hi=None, step_mm=None):
    """**실제 validate_path와 동일한 알고리즘**으로 곡선을 검사한다.

    ┌─ 왜 이 함수가 따로 있어야 하는가 (추천이 계속 빗나갔던 진짜 이유) ─────┐
    │ 이전 시뮬레이션은 곡선 위 샘플마다 natural_pose_at을 독립 호출해 자세를 │
    │ 정했다. 그런데 실제 검증은 **앵커의 자세만** 구한 뒤 RotationSpline으로 │
    │ 전역 보간해서 중간 자세를 만든다. 두 방식은 같은 곡선에서도 자세가      │
    │ 최대 18°까지 어긋난다(실측). IK 성공/실패는 자세에 극도로 민감하므로   │
    │ (orient_tol=0.5°), 시뮬레이션이 "통과"라고 해도 실제 검증은 전혀 다른  │
    │ 자세를 요구하며 그대로 실패한다. 추천대로 옮겼는데 또 실패하던 원인.   │
    │                                                                        │
    │ 이제 이 함수 하나가 두 곳(추천 시뮬레이션 / 실제 검증)에서 **똑같이**  │
    │ 쓰이므로 구조적으로 어긋날 수가 없다.                                  │
    └────────────────────────────────────────────────────────────────────────┘

    [6차 분할] pos_tol_mm/orient_tol_deg/step_mm을 전부 **인자로 받는다** -
    예전엔 self.CURVE_POS_TOL_MM이나 self._stream_step_mm() 같은 GUI 상태를
    이 함수 안에서 직접 조회했는데, 이 모듈은 GUI(self)를 모르므로 호출자
    (path_editor.py의 얇은 위임 메서드)가 미리 계산해서 넘겨야 한다.
    step_mm을 안 넘기면(None) 곡선 간격을 정할 방법이 없으므로 예외를 낸다 -
    예전처럼 조용히 기본값으로 넘어가면 "검증과 추천이 서로 다른 간격을
    쓰는" 문제(§6.3)가 다른 형태로 재발할 수 있어서, 차라리 시끄럽게 죽는
    쪽을 택했다.

    반환: (통과여부, 최악 조건수, 실패사유 또는 None)
    """
    if step_mm is None:
        raise ValueError(
            "run_curve_check(step_mm=None) - 곡선 간격이 필요합니다. "
            "호출자가 명시적으로 넘겨야 합니다(예: path_editor의 self._stream_step_mm()).")

    n = len(pts_xyz)
    arr = np.asarray(pts_xyz, dtype=float)
    xs, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]
    ts = np.arange(n, dtype=float)

    orientations, bad_i = build_orientations(pts_xyz, roles, q_seed)
    if orientations is None:
        return False, None, f"{bad_i+1}번 앵커에 도달 자세 없음"

    # --- 실제 검증과 동일: RotationSpline(없으면 Slerp)로 전역 자세 보간 ---
    R_ctrl = Rotation.from_matrix(np.stack(orientations))
    if _HAS_ROT_SPLINE and n >= 3:
        _ri = RotationSpline(ts, R_ctrl)
        def _R_at(t_):
            return _ri(t_).as_matrix()
    else:
        _sl = Slerp(ts, R_ctrl)
        def _R_at(t_):
            return _sl(np.clip(t_, ts[0], ts[-1])).as_matrix()

    lo = 0.0 if t_lo is None else float(t_lo)
    hi = float(n - 1) if t_hi is None else float(t_hi)

    # [validate_path와 통일] 파라미터 균일이 아니라 현(chord) 균일 재샘플링.
    # 여기서 다시 arc-length나 파라미터 균일로 되돌아가면, validate_path의
    # 실제 웨이포인트 생성 방식과 또 어긋나서 지난번 고친 "추천대로 옮겨도
    # 재실패" 문제가 형태를 바꿔 재발한다 - 반드시 같은 함수를 써야 한다.
    t_probe = np.linspace(lo, hi, max(400, int(4000 * (hi - lo) / max(n - 1, 1))))
    px, py, pz = bezier_chain_eval(xs, ys, zs, t_probe)
    P_probe = np.stack([px, py, pz], axis=1)
    tq = resample_by_chord(t_probe, P_probe, step_mm)
    n_s = len(tq)
    sx, sy, sz = bezier_chain_eval(xs, ys, zs, tq)
    R_all = _R_at(tq)

    q = list(q_seed)
    worst_cond = 0.0
    for k in range(n_s):
        tgt = np.array([sx[k], sy[k], sz[k]])
        if tgt[2] < DESK_SAFETY_MARGIN_MM:
            return False, None, "책상 충돌"
        sol, pe, oe = solve_pose_ik(
            q, tgt, R_all[k],
            pos_tol_mm=pos_tol_mm, orient_tol_deg=orient_tol_deg)
        if sol is None:
            return False, None, f"IK 미도달(위치{pe:.1f}mm/자세{oe:.1f}°)"
        ok, bad = within_joint_limits(sol)
        if not ok:
            return False, None, f"J{bad+1} 관절한계"
        # [9차 세션, §24] numerical_jacobian(유한차분, FK 7회) 대신
        # analytic_jacobian_6d의 위치부분(위 3행)을 잘라 쓴다 - FK 1회로 끝난다.
        # [:3, :] 슬라이싱 필수: jacobian_condition_number는 SVD 기반이라
        # 6xN을 그대로 넘기면 조건수가 달라져 특이점 판정 기준이 바뀐다.
        cond = jacobian_condition_number(analytic_jacobian_6d(sol)[:3, :])
        if cond > CONDITION_NUMBER_MAX:
            return False, None, "특이점"
        worst_cond = max(worst_cond, cond)
        if check_self_collision(sol):
            return False, None, "자가충돌"
        q = sol
    return True, worst_cond, None


def curve_ok_with_substitution(points, idx, cand, q_seed, pos_tol_mm, orient_tol_deg, step_mm):
    """점 idx를 cand로 바꿨다고 가정하고, 영향받는 구간의 곡선이 통과하는지 확인.

    [중요] 검사 범위를 '영향받는 구간'으로 좁히되, **자세 보간은 전체 점 목록으로**
    해야 한다. RotationSpline은 전역 보간이라 앞뒤 점이 중간 자세에 영향을 주기
    때문이다. 그래서 좌표 배열은 통째로 넘기고 t 범위만 제한한다.

    [6차 버그수정 - 이 함수를 여기로 옮기기 전, path_editor.py에 있을 때 있었던 일]
    이 함수가 `def` 선언 없이 `_resample_by_chord()`의 `return` 뒤에 몸통만
    붙어있던 버그를 먼저 고쳤다(죽은 코드라 조용히 있다가 실제 호출 시
    AttributeError). 원본 업로드 파일에도 이미 있던 버그였다.

    [6차 시드 통일 - "3~4개 통과했는데 대체경로 없음"의 원인] 예전엔 호출자가
    넘겨준 q_seed(=실패 시점의 웜스타트 관절각)를 그대로 run_curve_check에
    넘겼다. 그런데 build_orientations()는 그 시드를 점 목록 인덱스 0부터
    순차 전파하는 시작점으로 쓴다 - 경로 중간의 웜스타트 값을 0번 점의
    시드로 쓰면 점 전체의 자세 사슬이 실제 검증(항상 고정 시드 ik_seed_q()
    에서 시작)과 어긋난다. 이제 국소검사도 고정 시드(ik_seed_q())를 쓴다 -
    호출자가 넘긴 q_seed 인자는 더 이상 자세 계산에 쓰이지 않는다(무시됨,
    호출 호환성 때문에 시그니처에만 남아있음).
    """
    rng = segment_range_for_index(points, idx)
    if rng is None:
        return True, 1.0
    seg_lo, seg_hi = rng

    pts_xyz = [[p.x, p.y, p.z] for p in points]
    roles = [p.is_anchor for p in points]
    pts_xyz[idx] = [float(cand[0]), float(cand[1]), float(cand[2])]

    t_lo, t_hi = 2.0 * seg_lo, 2.0 * (seg_hi + 1)
    t_hi = min(t_hi, float(len(pts_xyz) - 1))
    passed, wc, _why = run_curve_check(
        pts_xyz, roles, ik_seed_q(), pos_tol_mm, orient_tol_deg,
        t_lo=t_lo, t_hi=t_hi, step_mm=step_mm)
    return passed, wc
