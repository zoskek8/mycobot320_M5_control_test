# -*- coding: utf-8 -*-
"""
mycobot_tau_table.py
======================
[9차 세션, §27.7-3] §27.6에서 확정한 자세별 추종지연(tau) 실측값을 이용해
"현재 자세에서 이 관절의 tau는 몇 ms인가"를 조회하는 순수 함수 모듈이다.
Qt도 로봇 연결도 몰라도 된다(pose_log.py/curve_log.py와 같은 원칙) -
`mycobot_stream_exec.py`가 실시간 루프에서 값만 조회해 쓴다.

[데이터 출처와 한계 - 반드시 읽을 것]
  - **J2**: §27.6.2, 경유점(via-pose)으로 접근방향을 통제한 6자세 전체
    스윕. 신뢰할 수 있는 표다.
  - **J3**: §27.7-3 후속으로 6자세 전체 스윕(메뉴 [2], 고정방향 +) 실행
    완료 - `J3_TABLE`에 반영됨. 다만 자세=0도는 §27.6.1에서 확인된
    이봉분포(이번 스윕과는 다른 `-`방향에서 발견) 때문에 표에 값이
    있어도 `J3_UNSTABLE_MARGIN_DEG` 안이면 무조건 미보정 처리한다 -
    이번 스윕(+ 방향)의 0도 자체는 안정적이었지만(표준편차 0.13),
    호출 시점에 관절이 어느 방향에서 왔는지 이 모듈은 모르므로 보수적으로
    막는다.
  - **J4**: [10차 세션, §37] 6자세 전체 스윕(고정방향 +) 완료 -
    `J4_TABLE`에 반영됨. J2/J3보다도 가파른 계단형(0→30도 사이 +10.4ms)
    이라 선형보간이 그 구간을 실제보다 완만하게 근사한다. 표준편차는
    전부 작아 J3 같은 이봉불안정 징후는 없었다 - unstable margin 없음.
  - **J1/J5/J6**: 이번 조사 범위 밖이라 다루지 않는다. 항상 `None`.

[None의 의미] "이 관절/자세는 보정하지 않는다"는 뜻이고, 호출자
(`mycobot_stream_exec.py`)는 None을 받으면 look-ahead 이동 없이 원래
계획값을 그대로 보낸다 - 즉 **미보정 = 예전과 100% 동일 동작**이라
데이터가 없는 쪽으로 인해 새 버그가 생기지 않는다.
"""

# (자세 deg, tau ms) 쌍 - 자세 오름차순 정렬 필수(선형보간이 이 순서를 가정).
# §27.6.2 표 그대로.
J2_TABLE = [
    (-90.0, 85.8),
    (-60.0, 86.5),
    (-30.0, 83.3),
    (0.0, 89.9),
    (30.0, 91.3),
    (60.0, 92.9),
]

# [9차 세션 §27.7-3 후속] 메뉴 [2](고정방향 +)로 6자세 스윕 완료, 표 채움.
# 실측(자세, tau평균ms, 표준편차ms, 유효반복):
#   -60: 103.1 (±0.37, 4)   -30: 103.1 (±0.65, 4)    0: 102.6 (±0.13, 4)
#    30: 110.0 (±0.49, 4)    60: 110.5 (±0.60, 4)   90: 106.6 (±6.81, 4)
# -60~0은 ~103ms 평평, 30~60은 ~110ms 평평 - §17 J4(|J2|>=55 분기)와 같은
# 2단계 형태에 가깝다(선형회귀 R²=0.486로 약함, 전환점은 0~30 사이 어딘가 -
# 이 6점만으로는 정확한 위치 확정 못 함). 표는 J2와 동일하게 선형보간으로
# 처리한다 - 0->30 구간만 실제보다 완만한 램프로 근사되는 보수적 단순화다.
#
# [주의 - 90도 표본] 표준편차 6.81로 유독 크다. 반복4회 중 1회(94.8ms,
# t_d=245.9ms)만 나머지 3회(110.2~111.1ms)와 동떨어졌다 - §27.6.1의
# 진짜 이봉분포(4/6 갈림)와 달리 4회 중 1회뿐이라 증거가 약하지만, 완전히
# 무시하지도 않는다. 표에는 이상치를 뺀 3회 평균(110.5, 60도와 동일)을
# 넣는다 - 원시 평균(106.6)을 그대로 넣으면 60도(110.5)와 90도(106.6)
# 사이에 실제로 없을 수도 있는 내리막을 만들어낸다. 재현되는지는 다음에
# 90도 근방을 다시 재서 볼 것(§27.7 다음 세션 항목 c).
J3_TABLE = [
    (-60.0, 103.1),
    (-30.0, 103.1),
    (0.0, 102.6),
    (30.0, 110.0),
    (60.0, 110.5),
    (90.0, 110.5),   # 이상치 1건 제외한 3회 평균 - 위 주석 참고
]

# J3=0도 부근은 표가 채워지더라도(§27.6.1) 준확률적 이봉분포라 단일 tau가
# 없다 - 이 margin 안이면 표에 값이 있어도 무조건 미보정 처리한다.
J3_UNSTABLE_POSE_DEG = 0.0
J3_UNSTABLE_MARGIN_DEG = 10.0

# [10차 세션, §37] joint_lag.txt에서 6축 중 계속 평균/최대오차가 가장 큰
# 관절이었는데도 §27 조사 범위 밖이었던 J4를 메뉴 [7]로 스윕(고정방향 +,
# fixed={1:0, 2:-30, 3:30, 5:30, 6:0}, 자세당 4반복) 완료. 실측
# (자세, tau평균ms, 표준편차ms, 유효반복):
#   -90: 105.0 (±0.19, 4)   -60: 104.0 (±0.14, 4)   -30: 103.5 (±0.58, 4)
#     0: 101.8 (±0.15, 4)    30: 112.2 (±0.38, 4)    60: 114.1 (±0.12, 4)
# 변동폭 12.3ms(노이즈 0.26ms 대비 47.2배 - 확실한 자세의존). 선형회귀
# R²=0.522로 약함 - J2/J3처럼 2단계 계단형에 가깝고, 오히려 전환이 훨씬
# 가파르다(0도->30도 사이 30도 만에 +10.4ms, J2/J3의 전환 구간보다 급함).
# 표는 동일 원칙(선형보간)으로 처리한다 - 전환점이 0~30 사이 어디인지
# 이 6점만으로는 확정 못 하므로, 이 구간을 지나는 자세는 실제보다 완만한
# 램프로 근사되는 보수적 단순화다(J2/J3 주석과 동일 논리). 표준편차가
# 전부 작아(0.12~0.58ms) J3의 이봉분포 같은 불안정 징후는 없었다 - 별도
# unstable margin 불필요.
J4_TABLE = [
    (-90.0, 105.0),
    (-60.0, 104.0),
    (-30.0, 103.5),
    (0.0, 101.8),
    (30.0, 112.2),
    (60.0, 114.1),
]

MAX_LOOKAHEAD_STEPS = 6   # look-ahead가 몇 스텝을 넘지 못하게 하는 안전 상한
                          # (tau 추정이 크게 틀렸을 때 웨이포인트를 너무 멀리
                          #  건너뛰는 걸 막는다 - 48ms 주기 기준 6스텝=288ms,
                          #  실측 tau 범위(80~110ms)의 3배 가까운 여유).


def _interp(table, pose_deg):
    """정렬된 (자세, tau) 표에서 선형보간. 범위 밖이면 가장자리 값으로 클램프."""
    xs = [p for p, _ in table]
    ys = [t for _, t in table]
    if pose_deg <= xs[0]:
        return ys[0]
    if pose_deg >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= pose_deg <= xs[i + 1]:
            frac = (pose_deg - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + frac * (ys[i + 1] - ys[i])
    return ys[-1]  # 이론상 도달 안 함


def lookup_tau_ms(joint_idx, pose_deg):
    """joint_idx(1~6, 물리 관절번호)와 그 관절의 현재 자세(도)에서 tau(ms)를
    조회한다. 데이터가 없거나(J1/J5/J6, J3 미채움) 불안정 구간(J3=0도
    부근)이면 None."""
    if joint_idx == 2:
        return _interp(J2_TABLE, pose_deg)
    if joint_idx == 3:
        if J3_TABLE is None:
            return None
        if abs(pose_deg - J3_UNSTABLE_POSE_DEG) <= J3_UNSTABLE_MARGIN_DEG:
            return None
        return _interp(J3_TABLE, pose_deg)
    if joint_idx == 4:
        return _interp(J4_TABLE, pose_deg)
    return None


def lookahead_steps(joint_idx, pose_deg, dispatch_period_sec, max_steps=MAX_LOOKAHEAD_STEPS):
    """tau를 dispatch_period 기준 정수 스텝 수로 환산. 보정 없음이면 0."""
    tau_ms = lookup_tau_ms(joint_idx, pose_deg)
    if tau_ms is None or dispatch_period_sec <= 0:
        return 0
    steps = round((tau_ms / 1000.0) / dispatch_period_sec)
    return max(0, min(max_steps, steps))


if __name__ == "__main__":
    # 로봇 없이 돌아가는 자체 점검 - 표 보간이 측정점에서 정확히 원값을
    # 돌려주는지, 불안정 구간/미채움 관절이 항상 None인지 확인한다.
    print("자체 점검 시작...")

    for pose, tau in J2_TABLE:
        got = lookup_tau_ms(2, pose)
        assert abs(got - tau) < 1e-9, f"J2 {pose}도: 기대 {tau}, 실제 {got}"
    print(f"  ✅ J2 측정점 {len(J2_TABLE)}개 전부 정확히 재현")

    mid = lookup_tau_ms(2, -45.0)
    assert 83.3 <= mid <= 86.5, f"J2 -45도 보간값 범위 밖: {mid}"
    print(f"  ✅ J2 -45도 보간값 {mid:.2f}ms (83.3~86.5 범위 안)")

    edge_lo = lookup_tau_ms(2, -200.0)
    edge_hi = lookup_tau_ms(2, 200.0)
    assert edge_lo == 85.8 and edge_hi == 92.9, "J2 범위 밖 클램프 실패"
    print(f"  ✅ J2 범위 밖 클램프: {edge_lo} / {edge_hi}")

    assert lookup_tau_ms(3, 0.0) is None       # 불안정 마진 안 - 항상 미보정
    assert lookup_tau_ms(3, 5.0) is None       # 마진(±10도) 안쪽도 미보정
    j3_60 = lookup_tau_ms(3, 60.0)
    assert abs(j3_60 - 110.5) < 1e-9, f"J3 60도: 기대 110.5, 실제 {j3_60}"
    j3_90 = lookup_tau_ms(3, 90.0)
    assert abs(j3_90 - 110.5) < 1e-9, f"J3 90도: 기대 110.5(이상치 제외 평균), 실제 {j3_90}"
    print("  ✅ J3: 0도 마진 미보정, 60/90도 측정점 정확 재현 확인")

    for pose, tau in J4_TABLE:
        got = lookup_tau_ms(4, pose)
        assert abs(got - tau) < 1e-9, f"J4 {pose}도: 기대 {tau}, 실제 {got}"
    print(f"  ✅ J4 측정점 {len(J4_TABLE)}개 전부 정확히 재현")

    j4_mid = lookup_tau_ms(4, 15.0)   # 계단 구간 한복판 - 보수적 램프 근사 확인
    assert 101.8 <= j4_mid <= 112.2, f"J4 15도 보간값 범위 밖: {j4_mid}"
    print(f"  ✅ J4 15도(계단 구간 중앙) 보간값 {j4_mid:.2f}ms (101.8~112.2 범위 안)")

    j4_edge_lo = lookup_tau_ms(4, -200.0)
    j4_edge_hi = lookup_tau_ms(4, 200.0)
    assert j4_edge_lo == 105.0 and j4_edge_hi == 114.1, "J4 범위 밖 클램프 실패"
    print(f"  ✅ J4 범위 밖 클램프: {j4_edge_lo} / {j4_edge_hi}")

    for j in (1, 5, 6):
        assert lookup_tau_ms(j, 0.0) is None
    print("  ✅ J1/J5/J6: 데이터 없음 - 항상 None 확인")

    assert lookahead_steps(2, -45.0, 0.048) == round((mid / 1000.0) / 0.048)
    j3_60_shift = lookahead_steps(3, 60.0, 0.048)
    assert j3_60_shift == round((110.5 / 1000.0) / 0.048), f"J3 60도 lookahead 불일치: {j3_60_shift}"
    assert lookahead_steps(3, 0.0, 0.048) == 0   # 불안정 마진 - 항상 0
    j4_shift = lookahead_steps(4, 60.0, 0.048)
    assert j4_shift == round((114.1 / 1000.0) / 0.048), f"J4 60도 lookahead 불일치: {j4_shift}"
    print(f"  ✅ lookahead_steps 환산 정상 (예: J2 -45도, 48ms 주기 -> "
          f"{lookahead_steps(2, -45.0, 0.048)}스텝 / J3 60도 -> {j3_60_shift}스텝 / "
          f"J4 60도 -> {j4_shift}스텝)")

    # 안전 상한 확인 - 말도 안 되게 짧은 주기를 줘도 MAX_LOOKAHEAD_STEPS를 못 넘음.
    huge = lookahead_steps(2, 60.0, 0.001)
    assert huge == MAX_LOOKAHEAD_STEPS, f"안전 상한 실패: {huge}"
    print(f"  ✅ 안전 상한 확인: 극단적 주기에서도 {huge}스텝(<= {MAX_LOOKAHEAD_STEPS})")

    print("\n모든 자체 점검 통과.")
