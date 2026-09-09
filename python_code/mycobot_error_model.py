# -*- coding: utf-8 -*-
"""
mycobot_error_model.py
=======================
[9차 세션, §30] 추종오차를 **관절의 물리적 상태의 함수**로 학습해서, 곡선이
바뀌어도 재사용되는 피드포워드 보정을 만든다.

**왜 ILC가 아니라 이 방식인가**
ILC(반복학습제어)는 "이 곡선의 137번째 웨이포인트에서 J1을 0.4도 더" 같은
**웨이포인트 인덱스별 보정표**를 학습한다 - 효과는 빠르지만 곡선이 바뀌면
표가 통째로 무의미해지고, 같은 곡선이라도 속도가 다르면 웨이포인트 수가
달라져(35mm/s 302개 vs 25mm/s 423개) 인덱스가 어긋난다. 새 곡선을 그릴
때마다 처음부터 다시 배워야 한다.

여기서는 대신 오차를 **(자세, 속도, 가속도, 이동방향)의 함수**로 본다.
이러면 곡선이 달라도 "J2가 이 각도에서 이 속도로 이만큼 가속 중"이라는
상황은 반복되므로, **모든 실행이 곡선과 무관하게 학습 데이터가 된다.**

**특징을 물리에서 가져온 것이 핵심이다.** 강체 팔의 필요 토크는
    M(θ)·α + C(θ,ω) + G(θ) + 마찰·sign(ω)
형태이고, 추종오차는 결국 이걸 못 따라간 양이다. 그래서 관절별 특징을
이렇게 잡았고, **모든 항이 이 프로젝트에서 이미 실측으로 확인된 현상과
1:1 대응한다**(임의로 고른 특징이 아니라는 게 중요하다):

    α        관성        §29.20 - 오차가 속도 반전점(=α 최대)에 집중
    ω        점성마찰    §7.1  - 속도비례 추종지연
    sign(ω)  쿨롱마찰/백래시  §9 이슈7 - 방향이력이 실재함을 실측
    sinθ,cosθ 중력부하    §4.8  - J2 중력의존 정적오차 R²=0.815
    상수     고정 오프셋  §4.8  - 관절 오프셋

관절당 6개 파라미터 선형회귀(릿지)라 학습이 빠르고, 계수를 보면 "이 관절은
관성이 지배적/마찰이 지배적"처럼 해석이 된다.

**Qt도 로봇 연결도 모른다** - 순수 함수 모듈(pose_log/curve_log/tau_table과
같은 원칙). GUI가 실행 후 데이터를 넘겨 저장하고, 학습/검증은 별도 스크립트
(`mycobot_error_model_fit.py`)가 이 모듈을 불러 수행한다.

**[중요] 위상 정렬**: i번째 시각에 관측된 오차는 τ(85~390ms)만큼 **전에**
보낸 명령의 결과다. 그대로 같은 시각의 상태와 짝지으면 어긋나므로, 명령
궤적을 (t - τ_j)에서 다시 샘플링해서 짝을 맞춘다. §27에서 관절별 τ를
측정해둔 것이 여기서 그대로 쓰인다.
"""

import json
import os
from datetime import datetime

import numpy as np

TRAIN_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "error_model_train.jsonl")

# 특징 이름(순서 고정 - 계수 해석과 저장 포맷이 이 순서에 의존한다)
FEATURE_NAMES = ["alpha", "omega", "sign_omega", "sin_theta", "cos_theta", "bias"]
N_FEATURES = len(FEATURE_NAMES)

# [10차 세션, §34 신규] 관절별 다른 특징셋 - §30.6에서 제안했던 확장 특징
# (ω²/α²/자세-속도 결합항)을 실험해보니 J4/J5는 크게 개선(24%→32%,
# 16%→39%)되는데 J1은 오히려 나빠졌다(35%→24%, 홀드아웃 1건 음수전환) -
# 6관절이 같은 특징벡터를 공유하던 기존 구조로는 이 트레이드오프를 관절별로
# 가를 수 없었다. 그래서 특징셋 자체를 관절별로 고르게 한다. 아래 배정은
# 5곡선 교차검증(§32~34) 결과를 그대로 반영한 것 - J2/J3는 어느 쪽이든
# 문턱 미달이라 base로 둔다(단순함 유지, 손해 없음).
FEATURE_SET_BASE = FEATURE_NAMES
FEATURE_SET_EXTENDED = FEATURE_NAMES[:-1] + [
    "omega_sq_signed", "alpha_sq_signed", "sin_theta_x_omega", "bias"]
JOINT_FEATURE_SET = {
    0: "base",       # J1 - 확장 특징이 오히려 악화(35%→24%), base 유지
    1: "base",       # J2 - 어느 쪽이든 불안정(그라비티/백래시, §4.8·§27.6.1)
    2: "base",       # J3 - 어느 쪽이든 문턱 미달 + 이봉불안정성(§27.6.1)
    3: "extended",   # J4 - 확장 특징으로 개선(24%→32%)
    4: "extended",   # J5 - 확장 특징으로 대폭 개선(16%→39%, 5홀드아웃 전부 양수)
    5: "base",       # J6 - base가 근소 우위(59% vs 57%), 이미 실기검증된 계수 유지
}

# 릿지 정규화 - 특징 간 상관(예: 좁은 자세범위에서 sinθ와 cosθ)이 있을 때
# 계수가 폭주하는 것을 막는다. 0이면 순수 최소제곱.
RIDGE_LAMBDA = 1e-3

# 학습에서 제외할 구간 - 시작 과도응답(소프트스타트 구간)은 정상 동작이
# 아니므로 모델을 오염시킨다.
SKIP_HEAD_SAMPLES = 40


def build_features(theta_deg, omega_dps, alpha_dps2):
    """관절 하나의 상태에서 base(6개) 특징 벡터를 만든다. 입력은 스칼라 또는 배열.

    각도는 도 단위로 받아 sin/cos에서만 라디안으로 바꾼다(ω/α는 도 단위
    그대로 - 오차도 도 단위라 계수가 바로 해석 가능하다).
    """
    theta = np.radians(np.asarray(theta_deg, dtype=float))
    omega = np.asarray(omega_dps, dtype=float)
    alpha = np.asarray(alpha_dps2, dtype=float)
    return np.stack([
        alpha,
        omega,
        np.sign(omega),
        np.sin(theta),
        np.cos(theta),
        np.ones_like(omega),
    ], axis=-1)


def build_features_extended(theta_deg, omega_dps, alpha_dps2):
    """[10차 세션, §34] base 6개 + ω²/α²(부호유지)/sinθ·ω 결합항 3개 = 9개.

    ω²/α²는 스케일이 커서(도/s, 도/s²) 그대로 곱하면 계수가 극소해지므로
    100/1000으로 나눠 다른 특징과 비슷한 크기로 맞췄다 - 부호는
    x*|x| 형태로 유지한다(방향이 있는 비선형 항이라야 물리적으로 말이 됨,
    순수 x²는 항상 양수라 반대방향 운동을 구분 못 한다).
    """
    theta = np.radians(np.asarray(theta_deg, dtype=float))
    omega = np.asarray(omega_dps, dtype=float)
    alpha = np.asarray(alpha_dps2, dtype=float)
    sin_theta = np.sin(theta)
    return np.stack([
        alpha,
        omega,
        np.sign(omega),
        sin_theta,
        np.cos(theta),
        omega * np.abs(omega) / 100.0,
        alpha * np.abs(alpha) / 1000.0,
        sin_theta * omega / 10.0,
        np.ones_like(omega),
    ], axis=-1)


def build_features_for_joint(joint_idx, theta_deg, omega_dps, alpha_dps2):
    """joint_idx(0-based)에 배정된 특징셋(JOINT_FEATURE_SET)으로 분기한다.
    학습(build_training_rows)과 실시간 추론(mycobot_stream_exec.py) 양쪽에서
    반드시 이 함수를 거쳐야 저장된 가중치 길이와 특징벡터 길이가 맞는다."""
    which = JOINT_FEATURE_SET.get(joint_idx, "base")
    if which == "extended":
        return build_features_extended(theta_deg, omega_dps, alpha_dps2)
    return build_features(theta_deg, omega_dps, alpha_dps2)


def derive_state(times, angles, joint_idx):
    """명령 궤적에서 한 관절의 (θ, ω, α) 시계열을 만든다.

    **명령값에서 뽑는다는 게 중요하다** - 실측값을 미분하면 읽기 노이즈
    (§23, ~0.1도)가 ω에서 2도/s, α에서 40도/s²로 증폭돼 특징이 노이즈
    덩어리가 된다. 명령 궤적은 우리가 계산해서 보낸 값이라 깨끗하다.
    """
    t = np.asarray(times, dtype=float)
    th = np.asarray([a[joint_idx] for a in angles], dtype=float)
    if len(t) < 3:
        return th, np.zeros_like(th), np.zeros_like(th)
    om = np.gradient(th, t)
    al = np.gradient(om, t)
    return th, om, al


def build_training_rows(cmd_times, cmd_angles, meas_times, meas_angles,
                        tau_ms_per_joint=None, skip_head=SKIP_HEAD_SAMPLES):
    """한 번의 실행에서 관절별 (특징, 오차) 학습행을 만든다.

    반환: {joint_idx: (X[n,6], y[n])} - y는 추종오차(도, 명령-실측).

    tau_ms_per_joint: [J1..J6] 관절별 유효지연(ms). None이면 0으로 둔다
    (위상 정렬 없음 - 정확도가 떨어지므로 가급적 넘길 것).
    """
    cmd_t = np.asarray(cmd_times, dtype=float)
    meas_t = np.asarray(meas_times, dtype=float)
    if len(cmd_t) < 5 or len(meas_t) < 5:
        return {}

    out = {}
    for j in range(6):
        tau_s = 0.0
        if tau_ms_per_joint is not None and tau_ms_per_joint[j] is not None:
            tau_s = float(tau_ms_per_joint[j]) / 1000.0

        th_c, om_c, al_c = derive_state(cmd_t, cmd_angles, j)

        # 측정 시각 t에서 관측된 오차는 (t - tau) 시점 명령의 결과다.
        # 그 시점의 명령 각도/상태를 보간해서 짝을 맞춘다.
        src_t = meas_t - tau_s
        cmd_at = np.interp(src_t, cmd_t, th_c)
        om_at = np.interp(src_t, cmd_t, om_c)
        al_at = np.interp(src_t, cmd_t, al_c)

        meas_j = np.asarray([a[j] for a in meas_angles], dtype=float)
        err = cmd_at - meas_j

        # 궤적 범위 밖으로 나간 샘플(보간이 가장자리 값으로 클램프된 구간)과
        # 시작 과도응답 구간은 버린다.
        valid = (src_t >= cmd_t[0]) & (src_t <= cmd_t[-1])
        if skip_head > 0:
            valid[:skip_head] = False
        # [10차, §34] 필요 최소 샘플수는 이 관절의 실제 특징 개수 기준이어야
        # 한다(base 6개 vs extended 9개) - N_FEATURES 고정값을 쓰면 extended
        # 관절에서 너무 적은 표본도 통과시켜버릴 수 있다.
        n_feat = len(FEATURE_SET_EXTENDED) if JOINT_FEATURE_SET.get(j) == "extended" else len(FEATURE_SET_BASE)
        if valid.sum() < n_feat + 5:
            continue

        X = build_features_for_joint(j, cmd_at[valid], om_at[valid], al_at[valid])
        out[j] = (X, err[valid])
    return out


def fit_ridge(X, y, lam=RIDGE_LAMBDA):
    """릿지 회귀. 반환: (계수[6], R²).

    bias 항은 X에 이미 열로 들어있으므로 정규화에서 빼준다(상수항까지
    0으로 당기면 전체가 편향된다).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    P = np.eye(d) * lam
    P[-1, -1] = 0.0          # bias는 정규화 제외
    try:
        w = np.linalg.solve(X.T @ X + P, X.T @ y)
    except np.linalg.LinAlgError:
        return None, 0.0
    pred = X @ w
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return w, r2


def predict(w, X):
    """학습된 계수로 오차를 예측한다(= 피드포워드로 미리 더해줄 보정량)."""
    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float)


# ---------------------------------------------------------------------------
# 신뢰도 가중 - "모델이 자기가 모르는 영역인지 아는가"
# ---------------------------------------------------------------------------
# [9차 세션, §30.6] 순수 피드포워드의 가장 큰 위험은 **개루프**라는 점이다 -
# 모델이 틀리면 틀린 방향으로 적극적으로 밀어버린다(P 피드백은 최소한 실측을
# 보고 스스로 교정한다). 특히 학습에 없던 영역(예: 학습보다 빠른 속도)으로
# 나가면 선형모델은 아무 근거 없이 선형 외삽을 해버린다.
#
# 문헌에서는 GP 회귀의 예측 분산을 써서 "모델이 자신 없는 곳에서는 피드백
# 게인을 높이고 피드포워드를 줄이는" 방식이 쓰인다(GP 평균=피드포워드,
# GP 분산=신뢰도). 다만 GP는 3차 시간복잡도라 이 프로젝트에는 과하다.
#
# 같은 성질을 훨씬 싸게 얻는다: **학습 데이터 분포로부터의 마할라노비스
# 거리**를 신뢰도로 쓴다. 학습에서 본 상태 근처면 거리가 작고(신뢰↑),
# 처음 보는 상태면 거리가 크다(신뢰↓). 계산은 6x6 역행렬 한 번뿐이라
# 실시간 비용이 사실상 0이다.
#
# **속도를 올릴 때의 안전장치가 정확히 이것이다** - 학습에 없던 빠른 속도로
# 가면 ω/α가 학습 분포 밖으로 나가 신뢰도가 떨어지고, 피드포워드가 자동으로
# 약해져 위험한 외삽을 스스로 자제한다.

def fit_confidence(X):
    """학습 데이터의 분포(평균·역공분산)를 기억한다. 신뢰도 계산에 쓴다.

    bias 열(항상 1)은 분산이 0이라 공분산을 특이하게 만들므로 제외한다.
    """
    X = np.asarray(X, dtype=float)[:, :-1]     # bias 제외
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    # 정칙화 - 특징 하나가 거의 상수인 구간(예: 한 방향으로만 움직인 데이터의
    # sign_omega)에서 공분산이 특이해지는 것을 막는다.
    cov = np.atleast_2d(cov) + np.eye(len(mu)) * 1e-6
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.eye(len(mu))
    return {"mu": mu.tolist(), "inv_cov": inv.tolist()}


def confidence(conf, X, d_full=3.0):
    """0~1 신뢰도. 학습 분포 중심이면 1, 멀어질수록 0.

    d_full: 신뢰도가 0이 되는 거리를 "차원당 표준편차" 단위로 준다.
    [주의] 마할라노비스 거리는 차원 수에 따라 커진다 - 분포 안의 평범한
    점도 d차원에서는 평균적으로 sqrt(d)만큼 떨어져 있다(5차원이면 2.24).
    그래서 sqrt(차원수)로 정규화하지 않으면 학습 데이터 자신조차 신뢰도가
    낮게 나온다(처음 구현했을 때 실제로 0.32가 나왔다). d_full=3은
    "차원당 3σ 밖은 안 믿는다"는 뜻이다.

    [NumPy 2.x 주의] 입력이 단일 샘플(1차원, 특징 6개)이면 결과도 스칼라로
    돌려준다 - 2차원(여러 샘플)이면 배열을 그대로 돌려준다. 예전엔 항상
    shape(1,) 배열을 돌려줬는데, NumPy 2.x부터 `float(배열(1,))`가
    `only 0-dimensional arrays can be converted to...`로 막혀서, 단일
    샘플을 실시간 루프에서 쓰는 mycobot_stream_exec.py의 `float(...)`
    호출이 실제로 이 에러로 죽는 걸 오프라인 회귀 테스트에서 잡았다.
    """
    X = np.asarray(X, dtype=float)
    single = (X.ndim == 1)
    X2 = np.atleast_2d(X)[:, :-1]
    mu = np.asarray(conf["mu"], dtype=float)
    inv = np.asarray(conf["inv_cov"], dtype=float)
    diff = X2 - mu
    d2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
    d = np.sqrt(np.maximum(d2, 0.0))
    scale = d_full * np.sqrt(len(mu))
    out = np.clip(1.0 - d / scale, 0.0, 1.0)
    return float(out[0]) if single else out


def predict_weighted(w, conf, X, d_full=3.0):
    """신뢰도로 가중된 피드포워드 보정량. 모르는 영역에서는 스스로 물러선다."""
    return predict(w, X) * confidence(conf, X, d_full)


# ---------------------------------------------------------------------------
# 모델 저장/적재 - mycobot_stream_exec.py가 실시간 루프에서 쓰기 위함
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "error_model_weights.json")


def save_model(models, path=MODEL_PATH, joints=None):
    """학습된 모델을 JSON으로 저장한다. joints를 주면 그 관절만 저장한다
    (§30.7 결정: J1/J4/J6만 피드포워드 채택 - J2/J3/J5는 신뢰도가 낮아
    제외). 저장 포맷의 키는 문자열 관절번호(1~6, 사람이 읽기 쉽게) - 파일을
    직접 열어봐도 뭐가 들어있는지 바로 보이게 하려는 것.
    """
    if joints is not None:
        models = {j: m for j, m in models.items() if j in joints}
    payload = {str(j + 1): m for j, m in models.items()}   # 0-based -> J1..J6 표기
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_model(path=MODEL_PATH):
    """저장된 모델을 읽는다. 파일이 없으면 빈 dict(=모든 관절 피드포워드 없음).

    [10차 세션, §35] JSON에서 읽으면 conf["mu"]/["inv_cov"]가 파이썬 리스트인데,
    predict_weighted()가 실시간 루프에서 매 사이클·매 관절마다 confidence()를
    부르고, 그때마다 np.asarray(리스트)로 새로 배열을 만들었다 - 사이클당
    수백 마이크로초 수준이라 평균 소요시간엔 안 잡히지만, 반복 할당이
    가비지컬렉션 압박을 키워 아주 가끔(수백 사이클에 한 번) 한 사이클이
    확 튀는 꼬리지연을 만들 수 있다는 게 §35.1 실기 A/B 3회에서 나온
    간접 증거였다(대조군 J2/J3까지 FF ON 실행에서만 일관되게 지연이
    커짐, 순서를 바꿔도 유지됨). 여기서 미리 numpy로 바꿔두면 이후
    confidence()의 np.asarray()가 항상 이미 배열인 걸 받아 사실상
    공짜(복사 없이 그대로 반환)가 된다 - 결과값은 완전히 동일하고
    속도만 바뀌는 순수 최적화라 회귀 위험이 없다.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    models = {}
    for j_str, m in payload.items():
        m = dict(m)
        m["w"] = np.asarray(m["w"], dtype=float)
        conf = dict(m["conf"])
        conf["mu"] = np.asarray(conf["mu"], dtype=float)
        conf["inv_cov"] = np.asarray(conf["inv_cov"], dtype=float)
        m["conf"] = conf
        models[int(j_str) - 1] = m   # J1..J6 표기 -> 0-based
    return models


# ---------------------------------------------------------------------------
# 실행 기록 저장/적재
# ---------------------------------------------------------------------------

def save_run(curve_label, tcp_speed_mms, cmd_times, cmd_angles,
             meas_times, meas_angles, tau_ms_per_joint=None, extra=None):
    """한 번의 실행을 학습 데이터로 남긴다.

    **원본 시계열을 그대로 저장한다** - 특징을 미리 계산해서 저장하면
    나중에 특징 설계를 바꿀 때(예: α² 항 추가) 과거 데이터를 못 쓴다.
    원본을 두면 언제든 다시 뽑을 수 있다.

    실패해도 예외를 던지지 않는다(로그 실패가 본래 작업을 막으면 안 된다는
    기존 원칙 - append_log 계열과 동일).
    """
    record = {
        "timestamp": datetime.now().isoformat(timespec="microseconds"),
        "curve_label": curve_label,
        "tcp_speed_mms": tcp_speed_mms,
        "cmd_times": [round(float(t), 5) for t in cmd_times],
        "cmd_angles": [[round(float(v), 4) for v in a] for a in cmd_angles],
        "meas_times": [round(float(t), 5) for t in meas_times],
        "meas_angles": [[round(float(v), 4) for v in a] for a in meas_angles],
        "tau_ms": tau_ms_per_joint,
    }
    if extra:
        record["extra"] = extra
    try:
        with open(TRAIN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"⚠️ 오차모델 학습데이터 저장 실패: {e}")
        return False


def load_runs():
    """저장된 실행 기록을 전부 읽는다. 파일이 없으면 빈 리스트."""
    if not os.path.exists(TRAIN_LOG_PATH):
        return []
    runs = []
    with open(TRAIN_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


def rows_from_run(run, skip_head=SKIP_HEAD_SAMPLES):
    """저장된 레코드 하나에서 학습행을 뽑는다."""
    return build_training_rows(
        run["cmd_times"], run["cmd_angles"],
        run["meas_times"], run["meas_angles"],
        tau_ms_per_joint=run.get("tau_ms"), skip_head=skip_head)


def fit_from_runs(runs, lam=RIDGE_LAMBDA):
    """여러 실행을 합쳐 관절별 모델을 학습한다.
    반환: {joint_idx: {"w": [...], "r2": float, "n": int}}"""
    pooled = {}
    for run in runs:
        for j, (X, y) in rows_from_run(run).items():
            if j not in pooled:
                pooled[j] = [[], []]
            pooled[j][0].append(X)
            pooled[j][1].append(y)

    models = {}
    for j, (Xs, ys) in pooled.items():
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        w, r2 = fit_ridge(X, y, lam)
        if w is not None:
            models[j] = {"w": w.tolist(), "r2": r2, "n": int(len(y)),
                         # [§30.6] 신뢰도 가중용 - 이 관절이 "본 적 있는 상태"의
                         # 분포. 3단계(실제 적용)에서 외삽을 자제하는 데 쓴다.
                         "conf": fit_confidence(X)}
    return models


def evaluate(models, runs):
    """학습된 모델을 다른 실행들에 적용해 평가한다(일반화 검증용).

    반환: {joint_idx: {"r2": 홀드아웃 R², "rmse_before": .., "rmse_after": ..}}
    rmse_after는 "모델이 예측한 만큼 미리 보정했다면 남았을 오차"의 RMSE다.
    """
    pooled = {}
    for run in runs:
        for j, (X, y) in rows_from_run(run).items():
            if j not in pooled:
                pooled[j] = [[], []]
            pooled[j][0].append(X)
            pooled[j][1].append(y)

    out = {}
    for j, (Xs, ys) in pooled.items():
        if j not in models:
            continue
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        pred = predict(models[j]["w"], X)
        resid = y - pred
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 1e-12 else 0.0
        out[j] = {
            "r2": r2,
            "rmse_before": float(np.sqrt(np.mean(y ** 2))),
            "rmse_after": float(np.sqrt(np.mean(resid ** 2))),
            "n": int(len(y)),
        }
    return out


if __name__ == "__main__":
    # 로봇 없이 도는 자체 점검 - 합성 데이터로 "물리가 있으면 찾아내는가"를 본다.
    print("자체 점검 시작...")
    rng = np.random.default_rng(0)

    # 알려진 계수로 인공 오차를 만들고, 학습이 그걸 되찾는지 확인한다.
    TRUE_W = np.array([0.004, 0.02, 0.15, 0.30, -0.10, 0.05])

    def synth_run(n=600, seed=0, freq=0.7):
        r = np.random.default_rng(seed)
        t = np.linspace(0, 20, n)
        # 관절마다 다른 궤적(곡선이 다른 상황을 흉내)
        angles = []
        for i in range(n):
            angles.append([40 * np.sin(freq * t[i] + j) + 10 * j for j in range(6)])
        cmd_t, cmd_a = t.tolist(), angles
        # 측정 = 명령 - (참 모델 오차) - 노이즈
        meas_a = []
        for j_dummy in range(1):
            pass
        th_all, om_all, al_all = {}, {}, {}
        for j in range(6):
            th, om, al = derive_state(t, angles, j)
            th_all[j], om_all[j], al_all[j] = th, om, al
        for i in range(n):
            row = []
            for j in range(6):
                X = build_features(th_all[j][i], om_all[j][i], al_all[j][i])
                e = float(X @ TRUE_W) + r.normal(0, 0.05)
                row.append(angles[i][j] - e)
            meas_a.append(row)
        return cmd_t, cmd_a, t.tolist(), meas_a

    # 1) 같은 곡선으로 학습 -> 참 계수를 되찾는가
    c1 = synth_run(seed=1, freq=0.7)
    run1 = {"cmd_times": c1[0], "cmd_angles": c1[1],
            "meas_times": c1[2], "meas_angles": c1[3], "tau_ms": [0]*6}
    models = fit_from_runs([run1])
    assert len(models) == 6, models.keys()
    w0 = np.array(models[0]["w"])
    err = np.abs(w0 - TRUE_W)
    assert models[0]["r2"] > 0.95, models[0]["r2"]
    print(f"  ✅ 참 계수 복원: R²={models[0]['r2']:.4f}, 계수 최대오차={err.max():.4f}")

    # 2) **핵심 검증** - 다른 곡선(다른 주파수=다른 속도/가속도 분포)에 일반화되는가
    c2 = synth_run(seed=2, freq=1.6)   # 학습에 없던 곡선
    run2 = {"cmd_times": c2[0], "cmd_angles": c2[1],
            "meas_times": c2[2], "meas_angles": c2[3], "tau_ms": [0]*6}
    ev = evaluate(models, [run2])
    r2_holdout = ev[0]["r2"]
    reduction = 1 - ev[0]["rmse_after"] / ev[0]["rmse_before"]
    assert r2_holdout > 0.9, ev[0]
    print(f"  ✅ 홀드아웃 곡선 일반화: R²={r2_holdout:.4f}, "
          f"RMSE {ev[0]['rmse_before']:.3f}→{ev[0]['rmse_after']:.3f}도 "
          f"({reduction*100:.0f}% 감소)")

    # 3) 위상 정렬(τ)이 실제로 필요한지 - τ를 무시하면 성능이 나빠져야 한다
    n = 600
    t = np.linspace(0, 20, n)
    angles = [[40*np.sin(0.7*tt + j) for j in range(6)] for tt in t]
    TAU = 0.15
    meas = []
    th_all = {j: derive_state(t, angles, j) for j in range(6)}
    for i in range(n):
        row = []
        for j in range(6):
            # 측정은 tau만큼 늦게 반영 - 명령을 tau 전 시점에서 가져온다
            th_s = np.interp(t[i]-TAU, t, th_all[j][0])
            om_s = np.interp(t[i]-TAU, t, th_all[j][1])
            al_s = np.interp(t[i]-TAU, t, th_all[j][2])
            e = float(build_features(th_s, om_s, al_s) @ TRUE_W)
            row.append(th_s - e)
        meas.append(row)
    r_ok = {"cmd_times": t.tolist(), "cmd_angles": angles,
            "meas_times": t.tolist(), "meas_angles": meas, "tau_ms": [TAU*1000]*6}
    r_no = dict(r_ok, tau_ms=[0]*6)
    m_ok = fit_from_runs([r_ok]); m_no = fit_from_runs([r_no])
    print(f"  ✅ 위상정렬 효과: τ 반영 R²={m_ok[0]['r2']:.4f}  vs  "
          f"무시 R²={m_no[0]['r2']:.4f}")
    assert m_ok[0]["r2"] > m_no[0]["r2"], "τ 반영이 더 나빠졌다 - 정렬 로직 확인 필요"

    # 4) [§30.6] 신뢰도 가중 - 모르는 영역에서 스스로 물러서는가
    conf = models[0]["conf"]
    rows_in = rows_from_run(run1)[0][0]           # 학습에서 본 상태
    c_in = float(np.mean(confidence(conf, rows_in)))
    # 학습보다 훨씬 빠른 상황(ω·α를 5배) - 속도를 크게 올린 경우를 흉내
    rows_out = rows_in.copy()
    rows_out[:, 0] *= 5.0     # alpha
    rows_out[:, 1] *= 5.0     # omega
    c_out = float(np.mean(confidence(conf, rows_out)))
    print(f"  ✅ 신뢰도: 학습 분포 안 {c_in:.2f}  vs  학습보다 5배 빠름 {c_out:.2f}")
    assert c_in > 0.5, c_in
    assert c_out < c_in * 0.5, (c_in, c_out)

    # 가중 예측이 실제로 작아지는지(= 외삽 자제)
    ff_in = np.abs(predict_weighted(models[0]["w"], conf, rows_in)).mean()
    ff_raw_out = np.abs(predict(models[0]["w"], rows_out)).mean()
    ff_out = np.abs(predict_weighted(models[0]["w"], conf, rows_out)).mean()
    print(f"  ✅ 외삽 자제: 학습 밖에서 보정량 {ff_raw_out:.3f} → {ff_out:.3f}도 "
          f"({(1-ff_out/max(ff_raw_out,1e-9))*100:.0f}% 축소, 학습 안에서는 {ff_in:.3f}도 유지)")
    assert ff_out < ff_raw_out * 0.5

    # 5) [§30.7 채택 방식] 선택 관절만 저장/적재 - 저장 포맷 왕복 확인
    import tempfile
    tmp_path = os.path.join(tempfile.mkdtemp(), "model_test.json")
    save_model(models, path=tmp_path, joints=[0, 3, 5])   # J1/J4/J6만(0-based)
    loaded = load_model(tmp_path)
    assert set(loaded.keys()) == {0, 3, 5}, loaded.keys()
    assert np.allclose(loaded[0]["w"], models[0]["w"])
    assert 1 not in loaded and 2 not in loaded and 4 not in loaded
    print(f"  ✅ 모델 저장/적재 왕복 확인 (J1/J4/J6만 저장, J2/J3/J5 제외됨)")

    # 6) [10차 세션, §34] 관절별 다른 특징셋(base/extended) - 폭이 섞인
    # 모델이 저장/적재/예측 전 구간에서 안 깨지는지 확인. 가장 걱정되는
    # 실패 지점은 실시간 추론(mycobot_stream_exec.py)이 base 관절에 9폭
    # 특징을 넣거나 extended 관절에 6폭을 넣어 차원이 안 맞는 경우다.
    base_x = build_features_for_joint(0, 10.0, 5.0, 1.0)     # J1 -> base
    ext_x = build_features_for_joint(3, 10.0, 5.0, 1.0)      # J4 -> extended
    assert base_x.shape == (len(FEATURE_SET_BASE),), base_x.shape
    assert ext_x.shape == (len(FEATURE_SET_EXTENDED),), ext_x.shape
    print(f"  ✅ 관절별 특징 폭 분기: J1(base)={base_x.shape[0]}개, "
          f"J4(extended)={ext_x.shape[0]}개")

    # 합성 데이터로 실제 참계수 복원이 extended 관절(9폭)에서도 되는지 확인
    rng2 = np.random.default_rng(1)
    TRUE_W_EXT = rng2.normal(0, 0.1, len(FEATURE_SET_EXTENDED))
    n2 = 500
    t2 = np.linspace(0, 15, n2)
    ang2 = [[35*np.sin(0.5*tt + j) for j in range(6)] for tt in t2]
    th4, om4, al4 = derive_state(t2, ang2, 3)   # J4
    X4 = build_features_extended(th4, om4, al4)
    meas2 = [list(a) for a in ang2]
    err4 = X4 @ TRUE_W_EXT
    for i in range(n2):
        meas2[i][3] = ang2[i][3] - err4[i]
    run_ext = {"cmd_times": t2.tolist(), "cmd_angles": ang2,
               "meas_times": t2.tolist(), "meas_angles": meas2, "tau_ms": [0]*6}
    models_ext = fit_from_runs([run_ext])
    assert 3 in models_ext and models_ext[3]["r2"] > 0.9, models_ext.get(3)
    print(f"  ✅ extended 특징(J4) 참계수 복원: R²={models_ext[3]['r2']:.4f}")

    # base(J1)+extended(J4) 섞어서 한 파일에 저장/적재 - 실사용(SELECTED_JOINTS
    # =[0,3,4,5]) 패턴 그대로. predict()가 각 관절 폭에 맞는 X를 받으면 정상
    # 동작하는지까지 확인한다(실제 실시간 루프에서 build_features_for_joint를
    # 안 거치고 옛날처럼 build_features만 쓰면 여기서 차원 에러로 잡힌다).
    mixed_models = dict(models_ext)
    mixed_models[0] = models[0]   # 위에서 이미 학습한 J1(base) 모델 재사용
    tmp_path2 = os.path.join(tempfile.mkdtemp(), "model_mixed_test.json")
    save_model(mixed_models, path=tmp_path2, joints=[0, 3])
    loaded_mixed = load_model(tmp_path2)
    pred_j1 = predict(loaded_mixed[0]["w"], base_x)      # 스칼라 안 터지면 통과
    pred_j4 = predict(loaded_mixed[3]["w"], ext_x)
    assert np.isfinite(pred_j1) and np.isfinite(ext_x).all() and np.isfinite(pred_j4)
    print(f"  ✅ base+extended 혼합 저장/적재 후 예측 정상 (J1 pred={pred_j1:.4f}, "
          f"J4 pred={pred_j4:.4f})")

    # 7) [10차 세션, §35] load_model()이 conf/w를 numpy로 캐싱해서 돌려주는지 -
    # 이게 안 되면 confidence()가 실시간 루프에서 매 호출 리스트->배열
    # 변환을 계속 하게 되어 §35.1의 꼬리지연 문제가 되돌아온다.
    assert isinstance(loaded_mixed[0]["w"], np.ndarray), type(loaded_mixed[0]["w"])
    assert isinstance(loaded_mixed[0]["conf"]["mu"], np.ndarray)
    assert isinstance(loaded_mixed[0]["conf"]["inv_cov"], np.ndarray)
    print(f"  ✅ load_model() 캐싱: w/conf.mu/conf.inv_cov 전부 numpy 배열로 반환")

    print("\n모든 자체 점검 통과.")
