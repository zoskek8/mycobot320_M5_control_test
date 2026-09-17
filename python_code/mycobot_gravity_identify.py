# -*- coding: utf-8 -*-
"""
mycobot_gravity_identify.py
============================
[신규 프로젝트, §3 개정] 중력 모델을 **표가 아니라 파라미터**로 식별한다.

왜 방식을 바꿨나
-----------------
처음엔 관절 하나씩 스윕해서 "각도 -> 부하" 표를 만들려 했다. 그런데
J2에 걸리는 중력 토크는 **J2 바깥 링크들의 무게중심이 어디 있느냐**로
정해지고, 그 위치는 q3, q4, q5가 바꾼다. 앞서 잰
`부하 = -123.9*sin(q2)` 는 **q3=q4=q5=0 일 때의 단면 하나**일 뿐이다.
6축 조합을 표로 채우려면 경우의 수가 수십만 개라 불가능하다.

**표로 만들 이유가 없다.** 중력 토크는 링크 질량과 무게중심 위치에 대해
**선형**이기 때문이다:

    U(q) = Σ_i  m_i * g · (p_i(q) + R_i(q) c_i)
    τ_g  = ∂U/∂q

여기서 미지수를 `φ_i = [m_i, m_i*c_ix, m_i*c_iy, m_i*c_iz]` 로 묶으면
(표준 barycentric 파라미터) τ_g 가 φ 에 대해 선형이 된다:

    τ_g = Y(q) · φ

즉 **자세 몇십 개만 재서 φ 를 풀면, 그 뒤로는 어떤 자세든 계산으로 나온다.**
표가 아니라 파라미터를 구하는 것이다.

이미 잰 J2 스윕은 버리지 않는다 - `A+B+C = 123.9` 같은 **식 하나**를 얻은
것이고, 식별이 끝난 뒤 그 단면을 예측해 대조하는 **검증용**으로 쓴다(--verify).

왜 무작위 자세인가
-------------------
- 자세 하나를 잡으면 **J2·J3·J4·J5의 부하를 동시에** 읽을 수 있다. 한 번
  이동에 네 관절 데이터가 나오므로 한 관절씩 스윕하는 것보다 효율적이다.
- 자세를 격자로 고르면 회귀 행렬이 특정 방향으로 쏠려 조건수가 나빠진다.
  무작위로 뽑으면 그 문제가 줄어든다.

Y(q) 를 손으로 유도하지 않는 이유
-----------------------------------
`τ_g2 = A sin(q2) + B sin(q2+q3) + ...` 형태는 J2·J3·J4가 나란한 피치
축일 때만 맞다. 그 가정을 코드에 박아넣는 대신 **URDF 체인에서 수치적으로**
Y(q) 를 만든다 - 축 배치가 무엇이든 맞고, 가정이 틀렸을 때 조용히 틀린
값을 내놓는 일이 없다. `--axes` 로 실제 축 배치를 먼저 확인할 수 있다.

사용법
-------
    python3 mycobot_gravity_identify.py --axes        # 축 배치만 출력(로봇 불필요)
    python3 mycobot_gravity_identify.py --dry 40      # 자세 40개의 조건수만 점검(로봇 불필요)
    python3 mycobot_gravity_identify.py --n 40        # 실측 40자세 -> 식별
    python3 mycobot_gravity_identify.py --fit gravity_poses.csv --verify gravity_map_J2.csv

**--n 을 쓰면 로봇이 실제로 움직입니다.** 주변을 비우고 비상정지에 손을 두세요.
"""

import argparse
import csv
import math
import time

import numpy as np

from mycobot_kinematics import (
    chain, active_indices, within_joint_limits, check_self_collision,
    find_robot_port, DESK_SAFETY_MARGIN_MM,
)

G_VEC = np.array([0.0, 0.0, -9.81])     # 중력 방향 (base 좌표)

# 부하를 읽을 관절. --axes 실측 결과(2026-09-10):
#   J1 축은 항상 수직 -> 중력 토크 0 -> 제외.
#   J2/J3/J4/J6 는 피치 축(-y) -> 포함.
#   J5 는 q=0 에서만 수직이고 앞쪽 피치 관절이 움직이면 기운다 -> 포함.
MEASURE_JOINTS = (2, 3, 4, 5, 6)

REG_LOAD_L, REG_LOAD_H = 60, 61
REG_TEMP = 63
SETTLE_SEC = 2.0
N_SAMPLE = 6
TEMP_LIMIT_C = 55


# ---------------------------------------------------------------------------
# 운동학 - Y(q) 를 수치적으로 만든다
# ---------------------------------------------------------------------------
def full_q(q6_deg):
    """활성관절 6개(도) -> 체인 전체 길이 배열(라디안)."""
    q = [0.0] * len(chain.links)
    for k, idx in enumerate(active_indices):
        q[idx] = math.radians(q6_deg[k])
    return q


def link_frames(q6_deg):
    """각 링크의 4x4 변환. ikpy 의 full_kinematics 를 쓴다."""
    return chain.forward_kinematics(full_q(q6_deg), full_kinematics=True)


def potential_regressor(q6_deg):
    """위치에너지 U 를 φ 에 대한 선형식으로 본 계수벡터.

    U = Σ_i [ g·p_i , (gᵀR_i)_x , (gᵀR_i)_y , (gᵀR_i)_z ] · φ_i
    반환 길이 = 4 * (활성관절 수) - 각 관절이 '그 뒤에 달린 링크'를 대표한다.
    """
    frames = link_frames(q6_deg)
    row = []
    for idx in active_indices:
        T = frames[idx]
        p = T[:3, 3]
        R = T[:3, :3]
        gR = G_VEC @ R
        row.extend([float(G_VEC @ p), float(gR[0]), float(gR[1]), float(gR[2])])
    return np.array(row)


def gravity_regressor(q6_deg, h_deg=0.05):
    """τ_g = Y(q)·φ 의 Y. 행 = 관절, 열 = φ 성분.

    τ_j = ∂U/∂q_j 이므로 위치에너지 계수벡터를 q_j 로 수치미분한다.
    **측정값이 아니라 계수벡터를 미분**하는 것이라 노이즈가 없다 - 해석적
    미분을 손으로 유도하는 것과 결과가 같고 실수할 여지가 적다.
    """
    n = len(active_indices)
    Y = np.zeros((n, 4 * n))
    for j in range(n):
        qp = list(q6_deg); qp[j] += h_deg
        qm = list(q6_deg); qm[j] -= h_deg
        dU = (potential_regressor(qp) - potential_regressor(qm)) / (2 * math.radians(h_deg))
        Y[j, :] = dU
    return Y


# ---------------------------------------------------------------------------
# 자세 생성
# ---------------------------------------------------------------------------
def random_poses(n, rng, lo=None, hi=None):
    """관절한계·자가충돌·책상 아래를 피한 무작위 자세.

    J6도 범위를 준다 - J6는 피치 축이라(--axes 실측) 그 뒤에 달린 것의
    중력 토크를 받는다. 0으로 고정하면 그 자세 성분이 회귀에 전혀 안
    들어가 해당 파라미터를 못 푼다.
    J1은 축이 항상 수직이라 중력 토크에 기여하지 않지만, 자세를 바꿔두면
    다른 관절의 측정이 특정 방향에 쏠리는 것을 줄일 수 있어 범위를 준다.
    """
    lo = lo or [-90, -100, -100, -100, -90, -90]
    hi = hi or [90, 100, 100, 100, 90, 90]
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        q = [float(rng.uniform(lo[k], hi[k])) for k in range(6)]
        fq = full_q(q)
        ok, _ = within_joint_limits(fq)
        if not ok:
            continue
        if check_self_collision(fq):
            continue
        T = chain.forward_kinematics(fq)
        if T[2, 3] * 1000.0 < DESK_SAFETY_MARGIN_MM:
            continue
        out.append(q)
    return out


def condition_report(poses):
    """식별 가능성 점검 - 로봇을 움직이기 전에 확인한다.

    조건수가 크면 그 자세 집합으로는 φ 를 못 푼다. 수십 분 측정한 뒤에
    알면 늦으므로 먼저 본다.
    """
    rows = []
    for q in poses:
        Y = gravity_regressor(q)
        for j in MEASURE_JOINTS:
            rows.append(Y[j - 1, :])
    A = np.array(rows)
    s = np.linalg.svd(A, compute_uv=False)
    nz = s[s > s[0] * 1e-10]
    print(f"  회귀행렬 {A.shape[0]}행 x {A.shape[1]}열")
    print(f"  특이값 최대 {s[0]:.3e}  최소(유효) {nz[-1]:.3e}")
    print(f"  유효 랭크 {len(nz)} / {A.shape[1]}   조건수 {s[0]/nz[-1]:.1e}")
    if len(nz) < A.shape[1]:
        print(f"  ※ 랭크 부족 - φ 중 {A.shape[1]-len(nz)}개 방향은 이 자세들로")
        print(f"    구분되지 않습니다(중력만으로는 원래 식별 안 되는 성분이 있습니다).")
        print(f"    최소제곱 해는 그 방향을 0으로 두는 최소노름 해가 됩니다 -")
        print(f"    예측에는 문제 없지만 개별 질량값에 물리적 의미를 두면 안 됩니다.")
    return A


# ---------------------------------------------------------------------------
# 측정
# ---------------------------------------------------------------------------
def decode_load(lo, hi):
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return None
    raw = int(lo) + int(hi) * 256
    mag = raw & 0x3FF
    return -mag if (raw >> 10) & 1 else mag


def read_loads(mc, joints):
    out = {}
    for j in joints:
        try:
            out[j] = decode_load(mc.get_servo_data(j, REG_LOAD_L),
                                 mc.get_servo_data(j, REG_LOAD_H))
        except Exception:
            out[j] = None
    return out


def wait_arrival(mc, target, tol_deg=1.5, timeout=8.0):
    """명령한 자세에 실제로 도달할 때까지 기다린다.

    **이게 없어서 1차 측정이 망가진 것으로 의심된다.** 정해진 시간만
    기다리고 읽으면, 큰 이동(무작위 자세는 100도 넘게 움직인다)이 끝나기
    전에 값을 읽게 된다. 이동 중의 Present Load는 중력이 아니라 가속
    토크를 반영한다.

    반환: (도달여부, 실측자세, 최대오차)
    """
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        a = mc.get_angles()
        if isinstance(a, list) and len(a) >= 6:
            last = a[:6]
            err = max(abs(last[k] - target[k]) for k in range(6))
            if err <= tol_deg:
                # 도달 후에도 잠깐 더 안정화를 본다
                time.sleep(0.3)
                a2 = mc.get_angles()
                if isinstance(a2, list) and len(a2) >= 6:
                    last = a2[:6]
                    err = max(abs(last[k] - target[k]) for k in range(6))
                return True, last, err
        time.sleep(0.05)
    err = (max(abs(last[k] - target[k]) for k in range(6))
           if last else float("nan"))
    return False, last, err


def measure(mc, poses, warmup=True):
    """각 자세를 **양방향으로 접근**해 평균낸다.

    §3.3에서 확인된 쿨롱 마찰 이력(평균 +15, 부호가 한쪽으로 치우침) 때문에
    한 방향으로만 가면 값이 그만큼 편향된다. 목표자세 직전에 +쪽/-쪽에서
    각각 접근해 두 번 재고 평균한다.

    [1차 실패 후 추가] 고정 시간 대기를 **도달 확인**으로 바꾸고, 명령값이
    아니라 **실측 자세**를 기록한다. 두 접근 방향의 값도 따로 남겨
    재현성을 볼 수 있게 했다 - 1차에서는 평균만 남겨서 "값이 왜 이상한가"를
    사후에 가릴 수 없었다.
    """
    rows = []
    if warmup:
        # §3.1: 스윕 첫 측정이 냉간/정착 미완으로 튀었다(raw 293, 온도 23도).
        print("  워밍업 이동...")
        mc.send_angles([0, -20, 0, 0, 0, 0], 30)
        wait_arrival(mc, [0, -20, 0, 0, 0, 0])
        mc.send_angles([0, 20, 0, 0, 0, 0], 30)
        wait_arrival(mc, [0, 20, 0, 0, 0, 0])

    n_bad = 0
    for i, q in enumerate(poses):
        per_dir = {}
        actual = None
        arrived_all = True
        max_err = 0.0
        for sign in (+1, -1):
            approach = [q[k] + sign * 8.0 for k in range(6)]
            mc.send_angles(approach, 25)
            wait_arrival(mc, approach)
            mc.send_angles(list(q), 25)
            ok, act, err = wait_arrival(mc, list(q))
            if not ok:
                arrived_all = False
            max_err = max(max_err, err if err == err else 999.0)
            if act is not None:
                actual = act
            vals = {j: [] for j in MEASURE_JOINTS}
            for _ in range(N_SAMPLE):
                r = read_loads(mc, MEASURE_JOINTS)
                for j in MEASURE_JOINTS:
                    if isinstance(r[j], (int, float)):
                        vals[j].append(r[j])
                time.sleep(0.03)
            per_dir[sign] = {j: (sum(vals[j]) / len(vals[j]) if vals[j] else None)
                             for j in MEASURE_JOINTS}

        row = {"idx": i, "arrived": int(arrived_all), "pos_err_deg": round(max_err, 2)}
        for k in range(6):
            row[f"q{k+1}_cmd"] = round(q[k], 3)
            # **회귀에는 실측 자세를 쓴다** - 명령값과 다를 수 있다.
            row[f"q{k+1}"] = round(actual[k], 3) if actual else round(q[k], 3)
        spread = []
        for j in MEASURE_JOINTS:
            a, b = per_dir[+1][j], per_dir[-1][j]
            row[f"load{j}_up"] = a
            row[f"load{j}_dn"] = b
            if a is not None and b is not None:
                row[f"load{j}"] = (a + b) / 2
                spread.append(abs(a - b))
            else:
                row[f"load{j}"] = a if a is not None else b
        row["dir_spread"] = round(max(spread), 1) if spread else None
        rows.append(row)

        flag = "" if arrived_all else "  ★미도달"
        if not arrived_all:
            n_bad += 1
        shown = " ".join(f"J{j}:{row[f'load{j}']:+7.1f}" if row[f"load{j}"] is not None
                         else f"J{j}:   --" for j in MEASURE_JOINTS)
        print(f"  [{i+1:>3}/{len(poses)}] 오차{max_err:5.1f}도 편차{row['dir_spread']:6.1f}"
              f"  {shown}{flag}")

        try:
            t = mc.get_servo_data(2, REG_TEMP)
            if isinstance(t, (int, float)) and t > TEMP_LIMIT_C:
                print(f"\n  ★ 서보 온도 {t}도 - 중단합니다. 식힌 뒤 이어서 하세요.")
                break
        except Exception:
            pass

    if n_bad:
        print(f"\n  ★ {n_bad}/{len(rows)} 자세가 목표에 도달하지 못했습니다.")
        print("    fit 단계에서 자동 제외됩니다(arrived=0).")
    return rows


# ---------------------------------------------------------------------------
# 식별과 검증
# ---------------------------------------------------------------------------
def repeatability(mc, rng, n_pose=4, n_rep=3):
    """같은 자세를 여러 번 재서 **측정이 재현되는지** 본다.

    1차 무작위 측정이 R²=0.08로 실패했는데, 단일관절 스윕은 R²=0.996이었다.
    읽기 경로는 멀쩡하므로 절차 문제다. 원인을 좁히는 가장 빠른 길은
    "같은 자세를 다시 재면 같은 값이 나오는가"를 직접 확인하는 것이다.

    - 재현된다 -> 측정은 신뢰할 수 있고, 중력 모델이 부하를 설명하지
      못한다는 뜻이다(모델/해석 문제).
    - 재현 안 된다 -> 측정 절차 문제다(도달, 데드밴드, 이력 등).
    """
    poses = random_poses(n_pose, rng)
    print(f"\n[반복성 진단] 자세 {n_pose}개 x {n_rep}회, 매번 다른 자세를 거쳐 돌아온다")
    print("  ⚠️ 로봇이 움직입니다.")
    if input("  진행하려면 yes 입력: ").strip().lower() != "yes":
        return
    results = {i: {j: [] for j in MEASURE_JOINTS} for i in range(n_pose)}
    for rep in range(n_rep):
        for i, q in enumerate(poses):
            # 매번 다른 중간 자세를 거쳐 접근한다 - 직전 자세가 값에 영향을
            # 주는지(이력) 보기 위해서다.
            via = [float(rng.uniform(-40, 40)) for _ in range(6)]
            mc.send_angles(via, 30); wait_arrival(mc, via)
            mc.send_angles(list(q), 25)
            ok, act, err = wait_arrival(mc, list(q))
            vals = {j: [] for j in MEASURE_JOINTS}
            for _ in range(N_SAMPLE):
                r = read_loads(mc, MEASURE_JOINTS)
                for j in MEASURE_JOINTS:
                    if isinstance(r[j], (int, float)):
                        vals[j].append(r[j])
                time.sleep(0.03)
            for j in MEASURE_JOINTS:
                if vals[j]:
                    results[i][j].append(sum(vals[j]) / len(vals[j]))
            print(f"  rep{rep+1} 자세{i+1}: 오차{err:5.1f}도  "
                  + " ".join(f"J{j}:{results[i][j][-1]:+7.1f}" if results[i][j] else f"J{j}: --"
                             for j in MEASURE_JOINTS)
                  + ("" if ok else "  ★미도달"))

    print(f"\n  {'자세':>5}{'관절':>6}{'값들':>32}{'표준편차':>10}{'범위':>9}")
    worst = 0.0
    for i in range(n_pose):
        for j in MEASURE_JOINTS:
            v = results[i][j]
            if len(v) < 2:
                continue
            sd = float(np.std(v)); rng_ = max(v) - min(v)
            worst = max(worst, rng_)
            print(f"  {i+1:>5}{('J%d' % j):>6}{' '.join(f'{x:+7.1f}' for x in v):>32}"
                  f"{sd:>10.1f}{rng_:>9.1f}")
    print(f"\n  최대 재현 범위 = {worst:.1f} (부하 단위)")
    print(f"  참고: 1차 측정에서 J2 부하는 ±150 범위로 흩어졌다.")
    if worst > 40:
        print("  -> ★ 같은 자세인데 값이 크게 다릅니다. **측정 절차 문제**입니다.")
        print("     중력 모델을 의심할 단계가 아닙니다.")
    else:
        print("  -> 측정은 재현됩니다. 그렇다면 부하가 중력만으로 설명되지")
        print("     않는다는 뜻이므로, 모델 쪽(마찰/데드밴드 항)을 봐야 합니다.")


def _r2(y, pred):
    """R² 문자열. 분산이 0이면 정의되지 않으므로 '--' 로 표시한다."""
    var = float(((y - y.mean()) ** 2).sum())
    if var < 1e-9:
        return "--"
    return f"{1 - ((y - pred) ** 2).sum() / var:.3f}"


def fit_per_joint(rows, use_joints=None):
    """**관절마다 따로** 중력 모델을 세운다. 계획 1~3단계가 원래 이 방식이다.

    [왜 공동 식별을 버렸나 - §3.7]
    처음엔 6축을 하나의 φ 로 묶어(barycentric 파라미터) 한꺼번에 풀었다.
    그런데 실측에서 이렇게 갈렸다:

        관절별 독립 적합 : J2 R²=0.998 / J3 0.993 / J4 0.971 (잔차 RMS 2~3)
        파라미터 공유 강제: 전체 R²=0.664, J3는 0.292 로 붕괴

    평면 사슬이면 바깥 링크의 무게가 안쪽 관절에도 그대로 실리므로 항이
    **중첩**되어야 한다. 실측이 그 구조를 따르지 않는다는 것은 **Present
    Load 가 물리적 토크가 아니라는 뜻**이다 - 서보 내부 PWM 듀티 계열
    지표라 관절마다 감속비·마찰·제어이득이 섞인 미지의 배율이 붙는다.

    관절별로 보면 그 배율이 계수에 흡수되므로 아무 문제가 없다(R² 0.99+).
    그래서 관절별로 돌아간다 - 교수님 계획의 원래 방식이기도 하다.

    자세 의존성은 그대로 담긴다. 회귀행렬 Y(q) 의 j 행은 그 관절 바깥
    링크들의 자세를 전부 포함하므로, "q3 를 접으면 q2 부하가 준다" 같은
    커플링(계획 3단계)이 자동으로 들어간다.

    반환: {관절번호: φ_j}
    """
    use = tuple(use_joints) if use_joints else MEASURE_JOINTS
    good = [r for r in rows if str(r.get("arrived", "1")) != "0"]
    n_skip = len(rows) - len(good)

    out = {}
    print(f"\n[관절별 식별]  자세 {len(good)}개"
          + (f"  (미도달 {n_skip}개 제외)" if n_skip else ""))
    print(f"  {'관절':>5}{'표본':>6}{'실측 표준편차':>14}{'잔차 RMS':>11}{'R²':>9}")
    for j in use:
        A, b = [], []
        for r in good:
            v = r.get(f"load{j}")
            if v is None or v == "":
                continue
            q = [float(r[f"q{k+1}"]) for k in range(6)]
            A.append(gravity_regressor(q)[j - 1, :])
            b.append(float(v))
        if len(b) < 8:
            print(f"  {('J%d' % j):>5}{len(b):>6}   표본 부족 - 건너뜀")
            continue
        A = np.array(A); b = np.array(b)
        phi, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = A @ phi
        out[j] = phi
        print(f"  {('J%d' % j):>5}{len(b):>6}{b.std():>14.1f}"
              f"{np.sqrt(((b - pred) ** 2).mean()):>11.1f}{_r2(b, pred):>9}")
    print("\n  잔차 RMS 가 2~7 수준이면 측정 불확실도와 같은 크기다(접근 방향")
    print("  두 값의 차이로 잰 값). 그보다 크면 모델이 못 잡는 성분이 남은 것.")
    return out


def save_params(params, path="gravity_params.json"):
    """이후 단계(T1~T12, residual 계산)에서 쓰도록 계수를 남긴다."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(j): list(map(float, v)) for j, v in params.items()}, f, indent=1)
    print(f"\n계수 저장: {path}")


def verify_against_sweep(params, csv_path):
    """이미 잰 J2 단면을 예측해 대조한다 - 식별에 안 쓴 데이터다."""
    if 2 not in params:
        print("\n[검증] J2 계수가 없어 대조를 건너뜁니다.")
        return
    phi = params[2]
    try:
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"\n[검증] {csv_path} 를 읽지 못했습니다: {e}")
        return
    byang = {}
    for r in rows:
        try:
            a = float(r["angle_deg"]); v = float(r["load_signed"])
        except (KeyError, ValueError, TypeError):
            continue
        byang.setdefault(a, []).append(v)
    if not byang:
        print(f"\n[검증] {csv_path} 에서 쓸 값을 찾지 못했습니다.")
        return
    print(f"\n[검증] J2 스윕 단면 예측 대조 ({csv_path})")
    print(f"  {'q2':>7}{'실측(양방향평균)':>18}{'예측':>10}{'차이':>9}")
    meas_all, pred_all = [], []
    for a in sorted(byang):
        meas = sum(byang[a]) / len(byang[a])
        pred = float(gravity_regressor([0, a, 0, 0, 0, 0])[1, :] @ phi)
        meas_all.append(meas); pred_all.append(pred)
        print(f"  {a:>7.1f}{meas:>18.1f}{pred:>10.1f}{meas - pred:>+9.1f}")
    mm = np.array(meas_all); pp = np.array(pred_all)
    e = mm - pp
    print(f"\n  RMS 차이 {np.sqrt((e ** 2).mean()):.1f}")
    ok = np.abs(pp) > 5.0
    if ok.sum() >= 3:
        ratios = mm[ok] / pp[ok]
        k = float((mm[ok] @ pp[ok]) / (pp[ok] @ pp[ok]))
        print(f"  배율 실측/예측: 중앙 {np.median(ratios):.2f}  흩어짐 {ratios.std():.2f}")
        print(f"  최적 배율 {k:.2f} 적용 시 RMS {np.sqrt(((mm - k * pp) ** 2).mean()):.1f}")
    print("  ※ 이 스윕은 다른 관절이 모두 0인 완전히 뻗은 자세라, 무작위")
    print("    자세 데이터에서 보면 바깥쪽 조건이다. 차이가 남으면 그 영역에")
    print("    자세를 더 넣어 재적합할 것.")


def report_axes():
    print("[축 배치] q=0 에서 각 관절의 회전축 (base 좌표)")
    frames = link_frames([0] * 6)
    axes = []
    for k, idx in enumerate(active_indices):
        R = frames[idx][:3, :3]
        a = R @ np.asarray(chain.links[idx].rotation, dtype=float)
        a = a / max(np.linalg.norm(a), 1e-12)
        axes.append(a)
        print(f"  J{k+1}: [{a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}]")
    print("\n  나란한 축끼리 묶으면:")
    used = set()
    for i in range(6):
        if i in used:
            continue
        grp = [i]
        for j in range(i + 1, 6):
            if j not in used and abs(abs(float(axes[i] @ axes[j])) - 1.0) < 1e-3:
                grp.append(j); used.add(j)
        used.add(i)
        print(f"    {' , '.join('J%d' % (g+1) for g in grp)}")
    vert = [i for i in range(6) if abs(abs(float(axes[i] @ np.array([0, 0, 1.0]))) - 1.0) < 1e-3]
    if vert:
        print(f"\n  q=0 에서 회전축이 수직인 관절: {', '.join('J%d' % (i+1) for i in vert)}")
        print(f"    ※ **q=0 에서만** 그렇다는 데 주의. 자세가 바뀌면 축도 따라")
        print(f"      돌아가므로 중력 토크가 생긴다.")
        if 0 in vert:
            print(f"    J1 은 베이스 회전축이라 **어떤 자세에서도** 수직이다")
            print(f"      -> 중력 토크가 항상 0. 측정 대상에서 뺀다.")
        others = [i for i in vert if i != 0]
        if others:
            names = ', '.join('J%d' % (i+1) for i in others)
            print(f"    {names} 는 앞쪽 피치 관절이 움직이면 축이 기운다")
            print(f"      -> 중력 토크가 생기므로 **측정 대상에 남긴다.**")

    pitch = [i for i in range(6) if i not in vert]
    if pitch:
        print(f"\n  q=0 에서 중력 토크를 받는 축: {', '.join('J%d' % (i+1) for i in pitch)}")
        print(f"    식별 가능한 방향의 개수는 '축 개수 x 2' 같은 단순한 규칙으로")
        print(f"    예측되지 않는다(모사 실험에서 어긋났다). --dry 로 직접 잴 것.")
        print(f"    **어느 관절을 재느냐에 따라 달라진다** - J6를 측정에 넣으면")
        print(f"    식별 가능한 방향이 늘어난다(모사에서 8 -> 10).")


def main():
    ap = argparse.ArgumentParser(description="중력 파라미터 식별")
    ap.add_argument("--axes", action="store_true", help="축 배치만 출력(로봇 불필요)")
    ap.add_argument("--dry", type=int, default=0, help="N개 자세의 조건수만 점검(로봇 불필요)")
    ap.add_argument("--n", type=int, default=0, help="실측할 자세 개수")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--out", default="gravity_poses.csv")
    ap.add_argument("--fit", default=None, help="이미 모은 CSV로 식별만 수행")
    ap.add_argument("--verify", default=None, help="대조할 J2 스윕 CSV")
    ap.add_argument("--joints", default=None,
                    help="식별할 관절, 쉼표 구분 (예: 2,3,4). 기본은 전부")
    ap.add_argument("--repeat", action="store_true",
                    help="같은 자세를 반복 측정해 재현성을 본다(원인 좁히기)")
    args = ap.parse_args()
    use_joints = ([int(x) for x in args.joints.split(",")] if args.joints else None)

    if args.axes:
        report_axes()
        return

    rng = np.random.default_rng(args.seed)

    if args.repeat:
        mc, _ = find_robot_port()
        if mc is None:
            print("로봇을 찾지 못했습니다.")
            return
        repeatability(mc, rng)
        return

    if args.dry:
        poses = random_poses(args.dry, rng)
        print(f"[조건수 점검] 자세 {len(poses)}개")
        condition_report(poses)
        return

    if args.fit:
        with open(args.fit, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        params = fit_per_joint(rows, use_joints)
        if params:
            save_params(params)
        if args.verify:
            verify_against_sweep(params, args.verify)
        return

    if not args.n:
        ap.print_help()
        return

    poses = random_poses(args.n, rng)
    print(f"[자세 생성] {len(poses)}개 (관절한계·자가충돌·책상 통과)")
    condition_report(poses)

    print(f"\n  자세당 양방향 접근 2회 + 정착 -> 약 {len(poses)*7/60:.0f}분 예상")
    print("  ⚠️ 로봇이 실제로 움직입니다. 주변을 비우고 비상정지에 손을 두세요.")
    if input("  진행하려면 yes 입력: ").strip().lower() != "yes":
        print("  중단합니다.")
        return

    mc, _ = find_robot_port()
    if mc is None:
        print("로봇을 찾지 못했습니다.")
        return

    rows = measure(mc, poses)
    fields = (["idx", "arrived", "pos_err_deg", "dir_spread"]
              + [f"q{k+1}" for k in range(6)]
              + [f"q{k+1}_cmd" for k in range(6)]
              + [f"load{j}" for j in MEASURE_JOINTS]
              + [f"load{j}_up" for j in MEASURE_JOINTS]
              + [f"load{j}_dn" for j in MEASURE_JOINTS])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\n저장: {args.out} ({len(rows)}행)")

    params = fit_per_joint(rows, use_joints)
    if params:
        save_params(params)
    if args.verify:
        verify_against_sweep(params, args.verify)


if __name__ == "__main__":
    main()
