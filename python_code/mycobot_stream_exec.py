"""
스트리밍 실행 루프 (6차 분할)
================================================================================

mycobot_path_editor.py의 execute_path()에서 **실시간 데드라인이 있는 부분**만
뽑아냈다: 통신 주기 자동보정(calibrate_dispatch_period)과 브레이크 없는 스트리밍
전송 루프(run_streaming_dispatch, 적응형 재보정 포함).

이 모듈도 mycobot_curve_math.py와 같은 이유로 분리했다 - GUI(self)에 의존하지
않는 순수 로직만 모아서, 실수(예: `def` 누락 같은)를 눈에 띄게 만들고 재사용/
검증을 쉽게 한다. 자세한 배경은 mycobot_curve_math.py 모듈 docstring 참고.

**실행속도 차이는 없다** - 파일 분할은 조직 문제고, import한 함수를 부르는 것과
같은 파일의 메서드를 부르는 것은 파이썬 수준에서 비용이 사실상 같다.
"""

import time
import math
import gc
from collections import deque
import numpy as np

from mycobot_kinematics import active_indices, fast_send_angles, joint_offset_correction
from mycobot_tau_table import lookahead_steps
import mycobot_error_model as err_model

# ┌─ 스트리밍(브레이크 없음) 모드 설정 ────────────────────────────────────────┐
# │ fresh_mode(1)은 이전 명령을 버리고 최신 명령만 즉시 실행 -> 웨이포인트마다  │
# │ 감속하지 않음. 대신 목표에 도달하기 전에 다음 목표를 받으므로 코너를 조금  │
# │ 잘라먹는데, 그 양은 웨이포인트 간격에 비례하므로 간격을 촘촘히 하면 오차가 │
# │ 작게 유지된다.                                                             │
# │ [실측으로 밝혀진 사실] 한 사이클의 하한은 시리얼 왕복이 아니라 ATOM        │
# │ 펌웨어의 프레임 소화속도이며 약 28ms(MIN_DISPATCH_PERIOD_SEC)다.           │
# │ 읽기(get_coords/get_angles)는 ATOM 펌웨어 차원에서 1회당 약 20ms가 걸린다  │
# │ (Elephant Robotics 공식 이슈 #53: "읽기는 50Hz가 최대"). 매 사이클 2회     │
# │ 읽으면 40ms 이상이 측정에만 소모되어 움직임 자체가 끊긴다 -> 기본은        │
# │ '단방향'(쓰기 전용).                                                       │
# └─────────────────────────────────────────────────────────────────────────────┘
MIN_DISPATCH_PERIOD_SEC = 0.028   # 전송 주기 하한 (펌웨어 소화 한계 보호).
                                  #   실측: 22ms에서 밀림 1.7~1.8초 발생 -> 로봇이 못 따라감.

# ┌─ [패치 3] 속도 파라미터 모순 제거 ────────────────────────────────────────┐
# │ 이전에는 세 값이 서로 어긋나 있었다.                                      │
# │   · dispatch_period = max(0.028, STREAM_STEP_MM/STREAM_TCP_SPEED_MMS)     │
# │     -> 2/90=0.0222 < 0.028 이라 STREAM_TCP_SPEED_MMS(90)는 항상 무시됨.   │
# │        즉 죽은 파라미터였고, 실행 후 경고문 "STREAM_TCP_SPEED_MMS를       │
# │        낮추세요"를 따라해도 아무 일도 일어나지 않았다.                    │
# │   · GUI 표시는 STREAM_STEP_MM/0.044 = 45mm/s 라고 알려주는데              │
# │        실제 속도는 2/0.028 = 71mm/s 였다 (Figure 3 실측 68mm/s).          │
# │                                                                          │
# │ 이제 손잡이는 STREAM_TCP_SPEED_MMS 하나다. 간격과 주기는 여기서 유도한다.│
# └──────────────────────────────────────────────────────────────────────────┘
STREAM_TCP_SPEED_MMS = 35   # ★ 목표 TCP 속도 - 스트리밍 속도는 오직 이 값으로만 조절한다.
                            #   68mm/s에서 추종지연 0.25~0.4s가 관측되어 낮춰 잡음.
                            #   높일수록 지연/코너컷이 커지고, 낮출수록 웨이포인트가 늘어
                            #   검증 시간이 길어진다. 40~70 사이에서 실험할 것.
                            #
                            #   [9차 세션, §26 후속] 45에서도 급커브 곡선의 특정 구간
                            #   (방향전환이 급한 곳)에서 속도 프로파일 오차가 후반부에
                            #   크게 튀는 게 관측됐다 - 요구 각속도가 서보 추종한계에
                            #   부딪히는 것으로 §26에서 이미 확인된 것과 같은 메커니즘.
                            #   전체 속도를 낮추면 여유 있던 구간보다 한계에 가까웠던
                            #   구간이 비례해서 더 크게 개선될 것으로 기대. 40~70 권장
                            #   범위 밖(35)이라 실기로 개선 여부를 반드시 확인할 것 -
                            #   효과 없으면 40으로, 그래도면 되돌릴 것(45).

# 웨이포인트 간격: 주기가 하한(28ms)에 붙어 있다는 전제로 목표속도에서 역산.
#   간격이 너무 촘촘하면(<0.5mm) 검증이 무거워지고, 너무 성기면(>6mm) 코너를 잘라먹는다.
STREAM_STEP_MM = float(min(6.0, max(0.5, STREAM_TCP_SPEED_MMS * MIN_DISPATCH_PERIOD_SEC)))

# ┌─ [6차] "정확도를 유지하면서 더 빠르게" - 간격을 '실제' 사이클에 맞춘다 ────┐
# │ 위 STREAM_STEP_MM은 사이클이 하한 28ms에 붙어 있다고 **가정**하고 만든    │
# │ 값이다. 그런데 '매번 실측' 모드는 매 사이클 get_angles()를 한 번 더 하고, │
# │ 그 읽기가 펌웨어 차원에서 ~20ms라 실제 사이클이 33ms 언저리로 나온다.     │
# │                                                                          │
# │ TCP 속도 = 간격 / 실제사이클 이므로:                                     │
# │     1.26mm / 33ms = 38mm/s   <- 목표 45가 아니라 38이 나오던 이유         │
# │ (실측 status.txt: "실측 33±4ms -> TCP 약 38mm/s" 와 정확히 일치)          │
# │                                                                          │
# │ 즉 느린 원인은 로봇이 못 따라가서가 아니라, **간격을 실제보다 짧은 주기   │
# │ 기준으로 잡아서** 애초에 목표속도가 나올 수 없는 조합이었던 것이다.       │
# │ 간격을 실제 사이클 기준으로 키우면 사이클 수를 늘리지 않고 속도만 오른다  │
# │ - 통신을 더 몰아치는 게 아니므로 밀림도 늘지 않는다.                      │
# │                                                                          │
# │ [정확도에 미치는 영향] 간격 1.26 -> 1.49mm (1.18배). 코너컷은 간격에      │
# │ 비례하므로 형상오차(cross-track)가 최대 그만큼 커질 수 있다. 반대로       │
# │ 속도는 38->45mm/s로 18% 오른다. 원치 않으면 아래 스위치를 False로 두면   │
# │ 예전 동작 그대로다.                                                       │
# └──────────────────────────────────────────────────────────────────────────┘
STEP_MATCHES_REAL_CYCLE = True     # False면 6차 이전 동작(28ms 가정)으로 되돌아감
# 측정모드별 '실제로 관측된' 사이클 시간(초). 캘리브레이션 로그(사이클바닥/실측)에서
# 온 값이며, 환경이 바뀌면 실행 후 상태줄의 '실측 NNms'를 보고 갱신할 것.
MEASURE_CYCLE_SEC = {
    "none": MIN_DISPATCH_PERIOD_SEC,   # 읽기 없음 - 펌웨어 하한이 그대로 사이클
    "full": 0.033,                     # 매번 실측 - get_angles() 왕복이 더해짐
}

# ┌─ [9차 세션, §25] 스파이크 흡수 전략 - 실기 검증 완료, 채택 ────────────────┐
# │ 배경: 사이클 리듬(§25)의 정확한 원인(GC 아님, Logitech 리시버 아님 -    │
# │ 둘 다 확인됨)은 못 찾았지만, "원인을 몰라도 주기 자체를 스파이크보다    │
# │ 넉넉히 잡으면 흡수된다"는 전략은 원인과 무관하게 유효했다.              │
# │                                                                          │
# │ [실기 결과, 2026-08-24] 목표주기 48ms 적용 후:                          │
# │   - 주기미달 15% -> 0%, 사이클 47.2~48.9ms 좁은 띠로 수렴                │
# │   - cross-track 평균 1.54->1.50mm, 최대 5.90->5.22mm (거의 무손실,      │
# │     오히려 근소 개선 - 간격이 1.45배 커졌는데도 §17.5 비례관계로        │
# │     추정했던 정확도 악화는 일어나지 않았다)                             │
# │   - 진짜 대가: 관절 추종지연(τ)이 전 관절 고르게 +20ms 안팎 증가        │
# │     (J1 200->223ms, J4 196->224ms 등) - 초당 명령 개수가 30->21개로     │
# │     줄어 보정 빈도 자체가 낮아진 게 원인. 이 정도는 감수하기로 결정.    │
# │                                                                          │
# │ 이후 False로 되돌리고 싶으면(예: τ 증가가 실사용에서 체감되면) 아래     │
# │ 한 줄만 바꾸면 된다 - 다른 코드는 안 건드려도 됨.                       │
# └──────────────────────────────────────────────────────────────────────────┘
ABSORB_SPIKES_TEST = True         # [채택됨] False로 바꾸면 예전(주기미달 있는) 동작으로 복귀
ABSORB_TARGET_PERIOD_SEC = 0.048  # 실측 스파이크(최대 51.8~55.4ms)를 덮는 목표주기

# ┌─ [9차 세션, §27.7-3] 자세의존 지연(tau) 보정 - 실기 검증 완료, 채택 ────────┐
# │ §27.6에서 확정한 J2의 자세별 tau표(`mycobot_tau_table.py`)를 이용해,      │
# │ 매 웨이포인트를 보낼 때 그 관절만 자기 tau만큼 미래 웨이포인트를 미리     │
# │ 당겨(look-ahead) 보낸다 - "관절이 도착하는 데 tau만큼 걸린다면, 지금      │
# │ 시점에 tau 앞선 목표를 줘서 실제 도착이 원래 계획된 시각에 맞도록"        │
# │ 하는 발상이다.                                                            │
# │                                                                          │
# │ J2/J3/J4 모두 §27.6/§27.7-3/§37에서 확정한 표를 쓴다 - J3의 자세=0도  │
# │ 부근은 이봉분포(§27.6.1)라 표에 값이 있어도 자동으로 미보정 처리된다     │
# │ (`mycobot_tau_table.py`의 `J3_UNSTABLE_MARGIN_DEG` 참고). J1/J5/J6은     │
# │ 데이터가 없어 여전히 미보정이다.                                        │
# │                                                                          │
# │ **cmd_angles(그래프용 "원래 계획값")는 절대 건드리지 않는다** -          │
# │ joint_offset_correction과 똑같은 원칙(§ 정지오차 보정 주석 참고): 보정은  │
# │ 실제 전송값에만 적용하고, 그래프는 항상 원래 계획 대비 실측을 보여줘야    │
# │ "보정 덕분에 얼마나 좋아졌는지"가 그대로 드러난다.                       │
# │                                                                          │
# │ **[10차 세션, §44] 실기 A/B 완료 - 극적으로 개선.** curve_e433b126       │
# │ (35mm/s) 기준 무작위순서 6회(§43 프로토콜, 대조군조정 분석):             │
# │     J2 유효지연  146.8ms -> 50.8ms  (raw -96.0ms, 조정 -95.1ms)          │
# │     J3 유효지연  133.2ms -> 31.7ms  (raw -101.5ms, 조정 -100.6ms)        │
# │     J4 유효지연  117.3ms -> 22.5ms  (raw -94.8ms, 조정 -93.9ms)          │
# │ 대조군(J1/J5/J6)은 전부 1ms 이내 무변화 - 반복간 흩어짐 대비 ratio       │
# │ 24.8~42.8배로, 이번 세션 A/B 중 가장 깨끗하고 큰 신호였다. raw/조정이    │
# │ 거의 같아 §35~36의 공통교란도 거의 없었다. 기본 ON으로 전환한다.         │
# └──────────────────────────────────────────────────────────────────────────┘
APPLY_POSE_DEPENDENT_LAG_COMPENSATION = True

POSE_DEPENDENT_LAG_JOINTS = (2, 3, 4)   # [10차, §43] mycobot_tau_table.py에 표가 있는 관절 - 상태줄 표시에도 재사용

# ┌─ [10차 세션, §45] 파이프라인 읽기 - 실기 미검증, 기본 OFF ────────────────┐
# │ §39~40: get_angles() 1회 ~30ms(펌웨어 하한, fast_get_angles로도         │
# │ 못 줄임 - §40). '매번 실측' 모드는 이 30ms를 메인 디스패치 루프 안에서   │
# │ 동기(블로킹)로 문다. mycobot_kinematics.AsyncAngleReader(§6-d)가 이     │
# │ 블로킹을 백그라운드 스레드로 빼낸다 - 메인 루프는 get_latest()로        │
# │ 즉시(비차단) '가장 최근 측정값'만 가져다 쓴다.                          │
# │                                                                          │
# │ **P 피드백은 이미 stream_meas_angles[-1](직전 사이클 측정값)을 쓰고     │
# │ 있었다** - 즉 피드백이 원래부터 1사이클가량 낡은 값을 쓰는 구조였다.     │
# │ 그래서 이 기능은 "새 지연을 만드는" 게 아니라 "그 지연을 만드는 30ms    │
# │ 블로킹만 배경으로 옮기는" 것에 가깝다 - 디스패치 주기는 빨라지되        │
# │ 피드백 신선도는 비슷하게 유지될 가능성이 있지만, 정확한 순효과는        │
# │ 실기 A/B 없이는 확정할 수 없다(오프라인 검증은 스레드 메커니즘          │
# │ 자체의 정확성만 확인했다 - AsyncAngleReader 클래스 주석 참고).          │
# │                                                                          │
# │ **아직 로봇으로 A/B 검증 전이라 기본값 False.**                         │
# └──────────────────────────────────────────────────────────────────────────┘
APPLY_ASYNC_MEASUREMENT_PIPELINE = False

# ┌─ [9차 세션, §29] 실시간 P 피드백 제어 - 실기 미검증, 기본 OFF ────────────┐
# │ 지도교수 제안: 명령이 액추에이터로 가기 전에 비례기(P, y = k·error)를    │
# │ 거친다 - 에러가 0이면 보정 없음, 생기면 k배로 밀어준다.                  │
# │                                                                          │
# │ [먼저 시도, 만족스럽지 않으면 §27.7-3 lag lookahead + 잔차피드백으로     │
# │ 전환하기로 함(Smith predictor류) - 순서는 사용자 결정]                   │
# │                                                                          │
# │ [이 시스템에서 걱정되는 점] §27~28에서 계속 확인했듯 τ(85~390ms)가 전    │
# │ 관절에 걸쳐 크다. "측정값"은 최소 20ms(읽기 자체) + τ 전 명령에 대한     │
# │ 반응이라 이미 낡은 정보다 - 데드타임이 클수록 발산 없이 줄 수 있는 게인  │
# │ 상한이 낮아지는 게 제어이론의 기본 결과다. 그래서 기본 게인을 보수적으로 │
# │ 낮게(0.3) 잡고 안전 클램프(한 사이클당 최대 보정폭)도 같이 뒀다.         │
# │                                                                          │
# │ '매번 실측' 모드에서만 동작한다 - 다른 모드는 실시간 측정 자체가 없다.   │
# │ 첫 사이클(과거 측정이 아직 없음)은 보정 없이 그냥 넘어간다.              │
# └──────────────────────────────────────────────────────────────────────────┘
APPLY_P_FEEDBACK_CONTROL = True
# [9차 세션, §29.8] 관절별 독립 게인으로 전환 - §29.7 실측(자동조정이 k=0.6까지
# 밀어붙인 결과)에서 J1/J4/J5/J6은 계속 개선되는데 J2/J3는 k=0.5에서 이미
# 개선이 거의 멈췄다(최대오차조차 그대로거나 미세 반등). J2(§4.8, 중력의존
# 정적오차)·J3(§27.6.1, 백래시 이봉불안정성)는 애초에 순간오차에 비례해
# 미는 P제어로는 못 잡는 종류의 오차를 갖고 있다는 게 이전 조사와도
# 일치한다 - 전 관절에 같은 k를 쓰면 J2/J3에서는 효과 없이 클램프 위험만
# 키우고, J1/J4/J5/J6에서는 아직 남아있는 개선 여지를 못 쓰게 된다.
# 리스트 순서는 [J1, J2, J3, J4, J5, J6].
# [9차 세션, §29.10 후속] 자동조정으로 도달한 "k=0.6"이 램프업 구간(0.3에서
# 서서히 올라오는 과정)까지 섞인 회귀값이라 순수 고정값 성능과 공정하게
# 비교가 안 된다는 지적에 따라, 처음부터 끝까지 이 값 그대로 고정해서
# 도는 클린 A/B용으로 §29.9 관절별 자동조정이 도달했던 최종값을 기본
# 시작값에 그대로 반영한다(자동조정은 기본 OFF이므로 이제부터 항상 이
# 고정값으로 실행됨). §29.7/§29.9와 직접 비교할 기준점이 된다.
# [9차 세션, §29.14] k=0/0.3/0.5/0.7/0.8/0.9 순수 고정값 스윕(§29.13) 결과
# **k=0.8을 잠정 권장값으로 채택**한다 - 0.9는 평균 지표는 아직 나쁘지 않지만
# 위험 신호 3가지(클램프 8→23→42회로 매 단계 배증, J4만 유일하게 반전
# 143.4→122.0→128.4ms, 시작 과도응답이 160mm/s까지 격화)가 처음으로 동시에
# 나타난 지점이라 그 아래에서 멈춘다.
P_FEEDBACK_GAIN = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]   # y = k·error, 관절별 고정 k (§29.14 권장값)
P_FEEDBACK_MAX_CORRECTION_DEG = 5.0    # 안전 클램프(전 관절 공통) - 측정치 글리치로 한 사이클에 확 튀는 것 방지

# ┌─ [9차 세션, §29.5] 실기 A/B(k=0/0.3/0.5) 이후 자동 게인 조정 - 기본 OFF ────┐
# │ 실측 결과(§29.5): k 0→0.3→0.5로 갈수록 모든 지표가 개선됐지만 한계효용이   │
# │ 뚜렷이 줄었다(예: 최종오차 개선폭 -17.4mm→-5.9mm, -66% 체감) - 그리고      │
# │ k=0.5에서 처음으로 안전클램프가 2회 발동했고 J3 최대오차가 유일하게       │
# │ 반등했다(6.37→6.52도). "아직 더 올려도 되지만 여유가 빠르게 준다"는       │
# │ 신호라, 사람이 매번 값 하나씩 손으로 찔러보는 대신 위 ADAPT_*(통신주기     │
# │ 자동보정)와 똑같은 패턴으로 실행 중 자동 조정한다:                        │
# │   - 클램프가 창 안에서 단 한 번이라도 보이면 즉시 크게 내림(안전 방향이라  │
# │     주저 없이) + 쿨다운(바로 다시 안 올림)                                │
# │   - 클램프가 전혀 없으면 조금씩만 올림(신중하게)                          │
# │ 상한(P_FEEDBACK_GAIN_MAX)은 k=0.5까지만 실측했다는 사실 자체를 반영한     │
# │ 보수적 값이다 - 데이터 없는 구간까지 자동으로 밀고 들어가지 않는다.       │
# └──────────────────────────────────────────────────────────────────────────┘
P_FEEDBACK_AUTO_GAIN = False
P_FEEDBACK_GAIN_MIN = 0.1
# [9차 세션, §29.11 후속, 되돌림] §29.8에서 J2/J3만 낮게(0.5) 묶었던 근거는
# §29.7의 자동조정(램프업 포함 회귀) 결과였는데, §29.10에서 순수 고정
# k=0.7(램프업 없음)로 다시 재보니 J2/J3도 k=0.3→0.5 때와 거의 같은
# 속도로 계속 개선됐다 - "J2/J3가 k=0.5에서 멈춘다"는 이전 결론 자체가
# 램프업 구간이 섞인 평균이 만든 착시였을 가능성이 높다는 뜻이다. 근거가
# 흔들렸으므로 상한을 다시 균일하게 되돌린다 - §29.5(최초 자동조정
# 도입) 때와 같은 값(0.7). 재현 확인(k=0.7 재실행) 및 더 높은 지점
# (k=0.8~0.9) 탐색은 아직 안 됐다 - §29.10-후속에서 확정되면 다시 조정.
# 순서는 [J1, J2, J3, J4, J5, J6] - 리스트 형태는 유지(관절별 자동조정
# 루프가 인덱싱하므로) 하되 값만 균일.
P_FEEDBACK_GAIN_MAX = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
P_FEEDBACK_GAIN_STEP_UP = 0.05         # 안전할 때만, 천천히
P_FEEDBACK_GAIN_STEP_DOWN = 0.10       # 위험 신호 보이면 곧바로 크게
P_GAIN_ADAPT_WINDOW = 60               # 판단에 쓸 최근 사이클 수 (~3초, 48ms 주기 기준)
P_GAIN_ADAPT_CHECK_INTERVAL = 20       # 몇 사이클마다 한 번씩 판단할지
P_GAIN_ADAPT_COOLDOWN_CHECKS = 5       # 한 번 내린 뒤 이만큼 판단주기 동안 재상승 보류

# ┌─ [9차 세션, §29.15] 시작 과도응답(overshoot) 억제 - 게인 소프트스타트 ─────┐
# │ §29.13 스윕에서 게인을 올릴수록 Figure 3(속도 프로파일)의 **처음 1~2초**  │
# │ 구간이 점점 격해졌다(k=0.7 ~90-107mm/s → k=0.9 최대 160mm/s, 목표는       │
# │ 35mm/s). 정상 구간은 멀쩡한데 시작만 튄다는 게 핵심 단서다:               │
# │   - 곡선 시작 시점엔 로봇이 아직 시작점에 완전히 정착하지 않아 초기       │
# │     오차가 평소보다 크다.                                                │
# │   - P 피드백은 그 큰 초기 오차에 k를 곱해 그대로 증폭한다 - k가 클수록    │
# │     첫 몇 사이클의 보정량이 과해지고, τ(85~390ms)만큼 늦게 반영되므로     │
# │     그때는 이미 지나쳐서 반대로 튀는 진동이 된다.                        │
# │                                                                          │
# │ 대응: 게인을 처음부터 풀로 주지 않고 0에서 시작해 P_GAIN_SOFTSTART_CYCLES │
# │ 동안 선형으로 올린다. 정상 구간(그 이후)의 동작은 100% 그대로이므로       │
# │ §29.13에서 확인한 개선 효과는 유지하면서 시작 구간만 순해진다.           │
# │                                                                          │
# │ [§20과 혼동 금지] 6차 세션에 제거한 시작가속램프(STREAM_RAMP_SEC)는       │
# │ **명령 속도 자체**를 서서히 올리는 것이라 곡선을 늦게 출발시켰고, A/B     │
# │ 두 번 다 τ를 오히려 키워서 폐기했다. 이건 다르다 - 명령 궤적은 처음부터   │
# │ 정상 속도 그대로 나가고, **피드백 보정량의 세기**만 서서히 붙인다.        │
# │ 즉 §20이 반증한 가설("천천히 출발하면 좋아진다")과는 무관하다.            │
# │                                                                          │
# │ **실기 미검증 - 기본 OFF.** 켜서 Figure 3의 시작 구간 피크가 줄어드는지,  │
# │ 그러면서 관절별 유효지연(§29.13 k=0.8 수치)이 유지되는지 확인할 것.       │
# └──────────────────────────────────────────────────────────────────────────┘
P_GAIN_SOFTSTART = True
P_GAIN_SOFTSTART_CYCLES = 40   # 약 2초(48ms 주기) - Figure 3에서 과도응답이 관측된 구간 길이

# ┌─ [9차 세션, §29.15] 진동 억제 - 데드밴드 + 보정량 평활화 ──────────────────┐
# │ "그래프 오차는 줄었는데 눈으로 보면 진동이 세졌다"는 관찰. Figure 3의     │
# │ 속도 프로파일에도 증거가 있다(k=0.9에서 5샘플 평균선이 20~50mm/s를 계속   │
# │ 규칙적으로 출렁임 - k=0 기준선은 훨씬 잔잔). 유효지연 회귀와 cross-track  │
# │ 평균은 고주파 진동을 평균으로 뭉개서 "좋아졌다"고만 나온다.               │
# │                                                                          │
# │ 원인: §23에서 확인했듯 get_angles()는 정지 중에도 0.1도대로 흔들린다      │
# │ (읽기 노이즈+양자화). P 피드백은 이걸 진짜 오차와 구분하지 못하고         │
# │ k*노이즈를 매 사이클 명령에 더한다 - k=0.8이면 0.1도 노이즈가 0.08도      │
# │ 명령 지터가 되고, 48ms마다 부호가 바뀌며 서보에 그대로 전달된다.          │
# │ **진짜 오차가 없는 구간에서도 로봇을 떨게 만드는 경로**가 새로 생긴 것.   │
# │                                                                          │
# │ 대응 1) 데드밴드: |오차|가 이 값 이하면 보정을 아예 0으로 둔다. 지도교수  │
# │    가 말한 "에러값이 0이면 작동 안 한다"를 노이즈 바닥까지 넓힌 형태다.   │
# │    분해능(~0.1도) 언저리 흔들림은 어차피 진짜 오차인지 알 수 없으므로     │
# │    쫓아가봐야 노이즈만 증폭한다.                                         │
# │ 대응 2) 보정량 평활화(EMA): 보정값을 이전 사이클과 섞어 급변을 눌러       │
# │    고주파 성분을 줄인다.                                                 │
# │    [트레이드오프] 평활화는 위상지연을 약 1사이클(48ms) 추가한다 - 데드타임│
# │    이 이미 큰 시스템이라 과하게 주면(alpha를 너무 낮추면) 오히려 불안정해질│
# │    수 있다. 그래서 기본값을 0.5(절반만 섞음)로 보수적으로 잡았다.        │
# │                                                                          │
# │ 둘 다 **중립값이 곧 OFF**라 별도 불리언 스위치가 필요 없다:               │
# │   DEADBAND=0.0  -> 모든 오차를 보정(지금까지와 100% 동일)                 │
# │   SMOOTH_ALPHA=1.0 -> 평활화 없음(지금까지와 100% 동일)                   │
# │ **실기 미검증이므로 기본값은 둘 다 중립(=꺼짐)으로 둔다.**                │
# └──────────────────────────────────────────────────────────────────────────┘
P_FEEDBACK_DEADBAND_DEG = 0.2   # [9차, §29.20] 권장값 채택 - 두 곡선에서 확인됨
P_FEEDBACK_SMOOTH_ALPHA = 0.5   # [9차, §29.20] 권장값 채택 - 두 곡선에서 확인됨

# ┌─ [9차 세션, §29.20] D항 추가 - PD 제어로 확장 ────────────────────────────┐
# │ §29.19 실측에서 cross-track 오차가 곡선 전체에 고르지 않고 **J1의 속도    │
# │ 반전점**(155도까지 올라갔다 급히 꺾여 내려오는 지점)에 집중된다는 게      │
# │ 드러났다. 25mm/s로 낮춰 재실행해도 cross-track 평균은 2.93→2.94mm로       │
# │ 그대로여서 속도 가설은 기각됐다(§29.20).                                 │
# │                                                                          │
# │ 그 지점은 각속도 부호가 바뀌며 **가속도 요구량이 최대**가 되는 곳인데,    │
# │ 순수 P제어는 "지금 오차"만 볼 뿐 "오차가 얼마나 빨리 커지는가"는 전혀     │
# │ 못 본다. D항(오차 변화율에 비례)은 원리적으로 이 구간에 반응한다.        │
# │                                                                          │
# │ [왜 '오차'의 미분이 맞는가 - 흔한 함정 하나] 일반적인 PID에서는 목표값이  │
# │ 바뀔 때 D가 튀는 걸 막으려고 '측정값'만 미분하는 게 정석이다. 그런데      │
# │ 여기서는 목표가 매 사이클 궤적을 따라 계속 움직이므로, 측정값을 미분하면  │
# │ 그냥 '현재 관절속도'가 나와서 정상 주행 중에도 계속 제동을 걸어버린다.    │
# │ 반면 **추종오차 e = 목표 - 실측은 정상 주행 중에는 거의 일정**(고정 지연) │
# │ 이라 de/dt ≈ 0 이고, 반전점처럼 따라가지 못하기 시작할 때만 커진다 -      │
# │ 정확히 우리가 반응하고 싶은 신호다. 그래서 여기서는 오차 미분이 맞다.     │
# │                                                                          │
# │ [노이즈] 미분은 노이즈를 증폭한다(0.1도 노이즈 / 48ms ≈ 2도/s). 그래서    │
# │ D 전용 저역통과 필터를 P의 평활화보다 더 강하게(기본 β=0.3) 따로 건다.    │
# │                                                                          │
# │ 단위: 오차[도], 미분[도/초] -> KD는 '초' 단위. 반전 구간에서 오차가       │
# │ 0.2초에 0.5도 벌어진다면(=2.5도/s) KD=0.05는 0.13도를 더한다 - P항(오차   │
# │ 1.5도 x 0.8 = 1.2도)의 10% 수준이라 보수적인 출발점이다.                  │
# │                                                                          │
# │ **실기 미검증이라 기본값 0.0(=비활성, 순수 P제어와 100% 동일).**          │
# └──────────────────────────────────────────────────────────────────────────┘
P_FEEDBACK_KD = 0.0          # 권장 시작값 0.05 (단위: 초). 0.0이면 비활성 = 순수 P
P_FEEDBACK_D_FILTER_BETA = 0.3   # D 전용 저역통과. 낮을수록 강한 필터(1.0=필터 없음)

# ┌─ [9차 세션, §30.7] 곡선 무관 오차모델 - 선택 관절(J1/J4/J6) 피드포워드 ───┐
# │ §30.7 교차검증에서 관절별로 갈렸다 - J1/J4/J6은 두 홀드아웃 모두 60%     │
# │ 이상 감소(일관됨), J2/J3/J5는 약하거나 음수(J3는 §27.6.1 이봉불안정성 -  │
# │ 비반복 오차라 원리적으로 이 방식이 못 잡는 케이스로 추정). 절충안으로     │
# │ **J1/J4/J6만** 적용한다 - `mycobot_error_model_fit.py`의 [4]로 저장한    │
# │ 모델을 여기서 읽는다.                                                    │
# │                                                                          │
# │ **적용 위치가 P/D와 다르다** - 이건 계획(θ,ω,α)만으로 미리 계산되는      │
# │ 피드포워드라 실측을 기다릴 필요가 없다(§27.7-3 lag lookahead와 같은      │
# │ 층). 그래서 P/D보다 먼저, 정지오차 보정 직전에 얹는다.                   │
# │                                                                          │
# │ **신뢰도 가중이 항상 걸린다**(§30.6) - 학습 때 안 본 상태(예: 학습보다   │
# │ 훨씬 빠른 속도)로 나가면 피드포워드가 자동으로 약해진다. 이 스위치를     │
# │ 켠 채로 속도를 크게 올려도 안전한 이유다.                                │
# │                                                                          │
# │ **실기 검증 완료 - 기본 ON.** [10차 세션, §33] curve_e433b126(35mm/s)     │
# │ 기준 A/B에서 J1/J4/J6 유효지연 전부 감소(-9.7%/-5.6%/-15.1%), cross-track │
# │ /tracking 전 지표 개선, 클램프·데드밴드 증가 없음(오히려 둘 다 감소)을    │
# │ 확인해 기본값을 켰다. 모델 파일이 없으면 켜도 조용히 아무 효과가         │
# │ 없다(§30.7의 [4]를 아직 안 돌렸다는 뜻이므로).                          │
# └──────────────────────────────────────────────────────────────────────────┘
APPLY_ERROR_MODEL_FEEDFORWARD = True
_ERROR_MODEL = err_model.load_model()   # 파일 없으면 {} - 모든 관절 미적용

# [9차 세션, §29.16 정리] STREAM_DISPATCH_PERIOD_SEC(파생 상수)는 삭제했다 -
# 실제 전송 주기는 calibrate_dispatch_period()가 실행마다 새로 재고, 웨이포인트
# 간격도 _stream_step_mm()이 동적으로 계산하므로 이 import 시점 고정값을 읽는
# 곳이 코드베이스 어디에도 없었다(전수 확인).

STREAM_SEND_SPEED = 100     # send_angles에 넘길 속도값 (항상 앞서가도록 높게)
# [6차 최종] 시작가속램프(STREAM_RAMP_SEC/STREAM_RAMP_START_FACTOR)는 제거했다.
# "0→목표속도 순간점프가 추종지연(τ)의 원인이니 서서히 가속하면 개선될 것"이라는
# 가설로 도입했었는데, 서로 다른 곡선 2개에서 A/B 비교한 결과 둘 다 **정반대**로
# 나왔다(램프 ON일 때 τ가 오히려 컸다: 280ms vs 180ms, 280ms vs 210ms - 두 번
# 다 OFF가 이김). 램프 구간 자체의 가감속 과도응답이 새로운 지연을 만드는
# 쪽이 맞았던 것으로 결론내고, GUI 토글과 함께 기능 자체를 걷어냈다. 이제
# run_streaming_dispatch는 처음부터 끝까지 dispatch_period로 등속 전송한다.

# ┌─ [6차 신규] 실시간 적응형 재보정 ──────────────────────────────────────────┐
# │ 그동안 "정지캘리브와 실행중 평균 사이의 격차"를 고정 상수(퍼센타일, +2ms  │
# │ 등)로 잡으려 했는데 세션마다 그 값 자체가 달랐다(§9 이슈1 - 90pct도       │
# │ 95pct도 효과 없었고, +2ms 고정값은 오히려 어떤 세션에선 더 나쁘게 만들며   │
# │ 다른 세션에선 필요없었다). 상수를 더 정교하게 다듬는 대신, 스트리밍       │
# │ 도중 실제 사이클을 계속 관찰해서 그 세션의 진짜 상태를 실행 중에 따라간다. │
# │                                                                           │
# │ 방식: 최근 ADAPT_WINDOW개 사이클의 '주기미달 비율'을 ADAPT_CHECK_INTERVAL │
# │ 사이클마다 확인한다. 미달이 잦으면(환경이 나빠짐) 주기를 조금 늘리고,     │
# │ 미달이 거의 없으면(여유가 생김) 원래 계산했던 값 쪽으로 조금 되돌린다.    │
# │                                                                           │
# │ 안전장치:                                                                 │
# │  - calib_period 아래로는 절대 안 내려간다 - 그 값은 정지 상태에서 신중히  │
# │    잰 하한이다. 빨라지는 방향은 딱 거기까지만.                            │
# │  - ADAPT_MAX_FACTOR(calib의 1.6배)를 넘게까지 늘어나면 더 안 늘리고       │
# │    경고만 남긴다 - 무한정 느려지는 건 재보정이 아니라 사실상 멈추는       │
# │    것과 같다. 그 이상은 '이 세션은 환경이 심각하다'는 신호로 다뤄야 한다. │
# │  - 매 사이클이 아니라 ADAPT_CHECK_INTERVAL마다만 판단한다 - deque         │
# │    append는 사실상 무료지만, 판단 자체를 너무 자주 하면 짧은 순간의       │
# │    우연한 튐에도 주기가 씰룩거려(진동) 오히려 더 불안정해진다.            │
# └───────────────────────────────────────────────────────────────────────────┘
ADAPT_WINDOW = 20            # 판단에 쓸 최근 사이클 수
ADAPT_CHECK_INTERVAL = 10    # 몇 사이클마다 한 번씩 판단할지
ADAPT_MISS_HIGH = 0.30       # 이 이상 미달이면 주기를 늘림
ADAPT_MISS_LOW = 0.05        # 이 이하면 원래값 쪽으로 회복 시도
ADAPT_STEP_UP = 1.08         # 늘릴 때 배율 (한 번에 8%)
ADAPT_STEP_DOWN = 0.97       # 되돌릴 때 배율 (더 조심스럽게 - 급히 빨라지면 다시 미달 위험)
ADAPT_MAX_FACTOR = 1.6       # calib_period 대비 최대 배율


def calibrate_dispatch_period(mc, measure_mode, still_angles_deg):
    """[자동 주기 보정] MIN_DISPATCH_PERIOD_SEC를 손으로 맞추는 대신,
    실제로 몇 사이클을 돌려 '이 컴퓨터+이 로봇+지금 선택된 측정모드'
    조합에서 한 사이클이 실제로 얼마나 걸리는지 재고 거기서 주기를 정한다.

    이미 도착해 정지해 있는 좌표(still_angles_deg)를 그대로 다시 보내므로
    로봇은 움직이지 않는다 - 순수하게 통신 왕복 비용만 잰다.

    [지연 FK 적용 후] 스트리밍 루프는 이제 get_angles()만 하고 순기구학은
    나중에 일괄 계산한다. 그래서 캘리브레이션도 get_angles()만 재는 게
    다시 맞다 - 실제 루프가 하는 일과 항상 같은 것을 재야 한다는 원칙은
    그대로다.

    반환: (전송 주기 sec, 실측 사이클 평균 sec, 표준편차 sec)

    [5차 개선, §9 이슈1] 표본 12개(측정모드 'full')는 지터(표준편차)를
    짧은 시간에 다 못 담아서, 최근 실행에서도 주기미달 31%가 관측됐다.
    두 가지를 함께 바꿨다: (1) 표본을 12→48('full')/8→32('none')로 증량,
    (2) 마진 공식을 '평균+1.5*표준편차'에서 90퍼센타일 실측값으로 교체.

    **[6차 - 시도했다 되돌림]** 95퍼센타일도 효과가 없어서(적용주기/실측/
    미달비율이 90퍼센타일 때와 소수점까지 동일) "정지캘리브와 실행중
    평균 사이에 고정 2ms 격차가 있다"고 보고 그 값을 더해봤는데, 세 번째
    실행에서 마진을 올린 만큼 실측 평균도 그대로 같이 올라버렸고 미달
    비율은 오히려 악화됐다(13%→21%). 표본 2개로 성급히 일반화한 판단
    오류였다 - 격차는 고정 상수가 아니라 세션마다(그때그때 USB/시리얼
    상태에 따라) 달라지는 값이었다. 그 수정은 되돌렸다.

    **[6차 재도입 - 이번엔 실시간 적응형]** 고정 상수 대신, 아래
    run_streaming_dispatch()가 실행 중에 직접 관찰하며 주기를 조정한다.
    이 함수(calibrate_dispatch_period)는 여전히 "시작할 때의 합리적인
    출발점"을 정하는 역할만 하고, 세션 중 변동 대응은 적응형 로직이 맡는다.
    """
    n_samples = 48 if measure_mode == "full" else 32   # 'none': 읽기가 없어 사이클이 매우 짧다

    # 워밍업 1회 - 첫 호출의 일회성 오버헤드(캐시 등)가 통계에 섞이지 않게.
    if not fast_send_angles(mc, still_angles_deg, STREAM_SEND_SPEED):
        mc.send_angles(still_angles_deg, STREAM_SEND_SPEED)
    if measure_mode == "full":
        mc.get_angles()

    cycle_times = []
    for _i in range(n_samples):
        t0 = time.perf_counter()
        if not fast_send_angles(mc, still_angles_deg, STREAM_SEND_SPEED):
            mc.send_angles(still_angles_deg, STREAM_SEND_SPEED)
        if measure_mode == "full":
            mc.get_angles()
        cycle_times.append(time.perf_counter() - t0)

    arr = np.array(cycle_times)
    floor_mean = float(arr.mean())
    floor_std = float(arr.std())
    floor_p95 = float(np.percentile(arr, 95))
    period = max(MIN_DISPATCH_PERIOD_SEC, floor_p95 * 1.05)
    return period, floor_mean, floor_std


def run_streaming_dispatch(mc, curve_waypoints, calib_period, apply_correction,
                            measure_mode, t_start,
                            cmd_times, cmd_angles, cmd_targets,
                            stream_meas_times, stream_meas_angles,
                            sample_measurement_raw_fn, log_fn=print):
    """스트리밍(브레이크 없음) 모드의 실시간 전송 루프.

    execute_path()에서 그대로 옮겨온 부분이다 - 실시간 데드라인이 있는 루프라
    안에서 무거운 계산을 하면 안 되고(§ 지연FK 원칙), 그 성질이 GUI 코드와
    섞여있으면 알아보기 어려워서 이 함수 하나로 독립시켰다.

    [부수효과에 대해] cmd_times/cmd_angles/cmd_targets/stream_meas_times/
    stream_meas_angles는 호출자가 만든 리스트를 **그대로 채운다**(새 리스트를
    반환하지 않음) - execute_path()가 이 루프 전후로 같은 리스트를 계속
    이어 쓰기 때문에, 참조를 유지하는 쪽이 자연스럽고 예전 동작과도 100%
    동일하다.

    sample_measurement_raw_fn(stream_meas_times, stream_meas_angles, t_start):
        get_angles()만 읽어서 버퍼에 쌓는 콜백. GUI 클래스(PathEditor)의
        `self._sample_measurement_raw`를 그대로 넘겨받는다 - 이 함수는 로봇
        객체(mc)에 접근해야 하는데, 이 모듈은 mc는 알아도 "PathEditor의 그
        메서드"까지는 몰라야 하므로(그래야 GUI와 무관해진다) 콜백으로 주입받는다.

    log_fn: 적응형 재보정이 조정할 때마다 부를 로그 함수 (기본 print - 콘솔에
        [적응형 재보정] 메시지가 그대로 뜬다. 테스트에서는 리스트에 담는 함수로
        바꿔치기할 수 있다).

    반환: dict(cycle_mean, cycle_std, late_ratio, target_period, adapt_events, calib_period)
    """
    dispatch_period = calib_period
    # [6차 최종] 시작가속램프를 제거했다 - 서로 다른 곡선 2개에서 A/B 비교한
    # 결과 둘 다 램프 ON이 오히려 τ(추종지연)를 늘렸다(280ms vs 180ms, 280ms
    # vs 210ms, 두 번 다 OFF가 이김). "0→목표속도 순간점프가 지연 원인"이라는
    # 가설이 두 번 다 반증됐다 - 램프 구간 자체의 가감속 과도응답이 오히려
    # 새 지연을 만드는 쪽이 맞았다. 그래서 램프 관련 코드(STREAM_RAMP_SEC,
    # STREAM_RAMP_START_FACTOR, GUI 토글)를 전부 걷어내고 처음부터 목표
    # 주기로 등속 전송한다 - 더 단순하고, 더 빠르고(워밍업 구간 없음),
    # 실측으로도 더 낫다는 게 확인됐다.
    cycle_times = []
    cycle_end_abs = []   # [9차, GC 조사] 각 사이클이 끝난 절대시각 (time.time())
    late_count = 0
    p_fb_clamp_count = 0   # [9차, §29] P 피드백이 안전 클램프에 걸린 횟수(관절×사이클 합)
    # [9차, §29.8] 자동 게인 조정 상태 - 관절마다 독립적으로 추적한다(ADAPT_*와
    # 같은 구조를 6벌 두는 것과 같음). 판단 주기 타이머(gain_adapt_since_check)만
    # 공유한다 - 어차피 한 사이클에 6관절이 동시에 처리되므로 타이밍을 굳이
    # 따로 둘 이유가 없다.
    current_p_gain = list(P_FEEDBACK_GAIN)   # [J1..J6] - 실행 중 값이 바뀌므로 복사
    recent_pfb_clamped = [deque(maxlen=P_GAIN_ADAPT_WINDOW) for _ in range(6)]
    gain_cooldown_remaining = [0] * 6
    gain_adapt_since_check = 0
    gain_adapt_events = []   # [(idx, joint_num(1~6), '올림'/'낮춤', 새게인), ...] - 상태줄/로그용
    # [9차, §29.15] 보정량 평활화(EMA)용 직전 사이클 보정값 - 관절별.
    prev_p_correction = [0.0] * 6
    # [9차, §29.20] D항 상태 - 관절별 직전 오차와 필터링된 미분값.
    prev_p_err = [None] * 6      # None = 아직 첫 사이클(미분할 이전 값이 없음)
    prev_d_filt = [0.0] * 6
    p_fb_deadband_hits = 0   # 데드밴드로 보정을 건너뛴 횟수(관절×사이클) - 효과 확인용
    # [6차] 적응형 재보정용 상태 - 위 ADAPT_* 상수 설명 참고.
    recent_late = deque(maxlen=ADAPT_WINDOW)
    adapt_since_check = 0
    adapt_events = []        # [(idx, '상향'/'복원', 새주기), ...] - 상태줄/로그용
    adapt_cap_warned = False
    next_t = time.time()
    prev_t = next_t
    # [9차 세션, 전송 리듬 조사] 사이클 리듬의 원인으로 파이썬 GC(가비지컬렉터)를
    # 의심 - 매 사이클 만들어지는 자잘한 리스트(angles_deg 등)가 일정 개수 쌓이면
    # 순환 GC가 주기적으로 발동할 수 있다. gc.callbacks에 콜백을 걸어 "발동
    # 시각"만 관찰한다 - 발동 여부/시점을 바꾸는 게 아니라 기록만 하므로 루프의
    # 실시간 동작 자체에는 영향이 없다(콜백 자체 비용도 리스트 append 1회뿐,
    # 이미 매 사이클 하고 있는 cmd_times.append와 같은 급).
    gc_events = []

    def _on_gc(phase, info):
        if phase == "start":
            gc_events.append(time.time())

    gc.callbacks.append(_on_gc)

    # [9차, §30.7] 오차모델 피드포워드용 - 계획 궤적의 ω/α를 실시간으로
    # 추정하기 위한 이전 사이클 상태(P/D의 prev_p_err와 같은 패턴).
    prev_plan_angle = [None] * 6
    prev_plan_omega = [None] * 6
    err_model_ff_abs_sum = 0.0   # [§30.7] 진단용 - 실제로 얼마나 밀었는지
    err_model_ff_count = 0

    try:
        for idx, (sol, target) in enumerate(curve_waypoints):
            angles_deg = [math.degrees(sol[i]) for i in active_indices]

            # [9차 세션, §27.7-3] 자세의존 지연보정 - 전송용 사본에만 적용한다
            # (angles_deg 자체는 cmd_angles 그래프용 "원래 계획값"이라 건드리면
            # 안 된다 - 위 스위치 설명 주석 참고). 기본 OFF.
            angles_for_send = angles_deg
            if APPLY_POSE_DEPENDENT_LAG_COMPENSATION:
                angles_for_send = list(angles_deg)
                for joint_idx in POSE_DEPENDENT_LAG_JOINTS:
                    k = joint_idx - 1
                    shift = lookahead_steps(joint_idx, angles_deg[k], dispatch_period)
                    if shift > 0:
                        future_idx = min(idx + shift, len(curve_waypoints) - 1)
                        future_sol = curve_waypoints[future_idx][0]
                        angles_for_send[k] = math.degrees(future_sol[active_indices[k]])

            # [9차, §30.7] 오차모델 피드포워드 - 계획(θ,ω,α)만으로 미리 계산되므로
            # 실측을 기다릴 필요가 없다(위 lag lookahead와 같은 층, P/D보다 먼저).
            # ω/α는 직전 사이클의 계획값과 비교해 실시간으로 추정한다.
            for k in range(6):
                theta = angles_deg[k]
                omega = (0.0 if prev_plan_angle[k] is None
                        else (theta - prev_plan_angle[k]) / max(dispatch_period, 1e-6))
                if APPLY_ERROR_MODEL_FEEDFORWARD and k in _ERROR_MODEL:
                    alpha = (0.0 if prev_plan_omega[k] is None
                            else (omega - prev_plan_omega[k]) / max(dispatch_period, 1e-6))
                    m = _ERROR_MODEL[k]
                    X = err_model.build_features_for_joint(k, theta, omega, alpha)
                    ff = float(err_model.predict_weighted(m["w"], m["conf"], X))
                    if angles_for_send is angles_deg:   # aliasing 방지 - §27.7-3과 동일 원칙
                        angles_for_send = list(angles_deg)
                    angles_for_send[k] += ff
                    err_model_ff_abs_sum += abs(ff)
                    err_model_ff_count += 1
                prev_plan_angle[k] = theta
                prev_plan_omega[k] = omega

            # [정지오차 보정] 그래프의 'commanded'는 원래 계획값(angles_deg)을
            # 그대로 쓴다 - 그래야 "보정 덕분에 measured가 얼마나 더 가까워졌는지"
            # 가 그래프에 그대로 드러난다. 로봇에는 보정된 값만 보낸다.
            angles_to_send = (joint_offset_correction(angles_for_send)
                              if apply_correction else angles_for_send)

            # [9차 세션, §29] P 피드백 - 위 스위치 설명 참고. '매번 실측'
            # 모드에서만 가능하다(다른 모드는 실시간 측정이 없다). 방금 시작한
            # 첫 사이클(stream_meas_angles가 아직 비어있음)은 피드백할 과거
            # 측정이 없으므로 건너뛴다.
            #
            # [aliasing 주의] angles_to_send는 위 조건에 따라 angles_for_send나
            # angles_deg를 '그대로'(복사 없이) 가리킬 수 있다 - 여기서 바로
            # 원소를 수정하면 몇 줄 아래 cmd_angles.append(angles_deg)가 이미
            # 오염된 값을 기록하게 된다(§ 정지오차 보정과 같은 함정). 그래서
            # 반드시 list()로 새로 복사한 뒤에만 고친다.
            if (APPLY_P_FEEDBACK_CONTROL and measure_mode == "full"
                    and stream_meas_angles):
                last_measured = stream_meas_angles[-1]
                angles_to_send = list(angles_to_send)
                for k in range(6):
                    gain_now = current_p_gain[k] if P_FEEDBACK_AUTO_GAIN else P_FEEDBACK_GAIN[k]
                    # [9차, §29.15] 소프트스타트 - 처음 N사이클 동안 게인을 0에서
                    # 선형으로 올린다. idx는 웨이포인트 인덱스라 곧 경과 사이클 수다.
                    if P_GAIN_SOFTSTART and idx < P_GAIN_SOFTSTART_CYCLES:
                        gain_now *= (idx + 1) / P_GAIN_SOFTSTART_CYCLES

                    err = angles_to_send[k] - last_measured[k]

                    # [9차, §29.15-1] 데드밴드 - 노이즈 바닥 이하의 오차는 아예
                    # 쫓아가지 않는다(진짜 오차인지 읽기 노이즈인지 구분 불가).
                    # D항도 같이 막는다 - "이 구간은 건드리지 않는다"는 의미를
                    # 일관되게 유지하기 위함(P만 막고 D가 살아있으면 데드밴드
                    # 안에서도 노이즈 미분으로 계속 떨게 된다).
                    if P_FEEDBACK_DEADBAND_DEG > 0.0 and abs(err) <= P_FEEDBACK_DEADBAND_DEG:
                        correction = 0.0
                        p_fb_deadband_hits += 1
                    else:
                        correction = gain_now * err
                        # [9차, §29.20] D항 - 오차 변화율에 비례. 위 상수 주석 참고.
                        if P_FEEDBACK_KD > 0.0 and prev_p_err[k] is not None:
                            d_raw = (err - prev_p_err[k]) / max(dispatch_period, 1e-6)
                            d_filt = (P_FEEDBACK_D_FILTER_BETA * d_raw
                                      + (1.0 - P_FEEDBACK_D_FILTER_BETA) * prev_d_filt[k])
                            prev_d_filt[k] = d_filt
                            correction += P_FEEDBACK_KD * d_filt
                    # 다음 사이클의 미분을 위해 항상 갱신한다 - 데드밴드로 건너뛴
                    # 사이클도 오차 자체는 관측됐으므로, 여기서 빼먹으면 데드밴드를
                    # 빠져나오는 순간 dt가 실제보다 여러 배 긴 미분이 튄다.
                    prev_p_err[k] = err

                    # [9차, §29.15-2] 보정량 평활화(EMA) - 급변을 눌러 고주파 성분을
                    # 줄인다. alpha=1.0이면 이 줄은 아무것도 안 한 것과 같다.
                    if P_FEEDBACK_SMOOTH_ALPHA < 1.0:
                        correction = (P_FEEDBACK_SMOOTH_ALPHA * correction
                                      + (1.0 - P_FEEDBACK_SMOOTH_ALPHA) * prev_p_correction[k])

                    clamped = False
                    if correction > P_FEEDBACK_MAX_CORRECTION_DEG:
                        correction = P_FEEDBACK_MAX_CORRECTION_DEG
                        p_fb_clamp_count += 1
                        clamped = True
                    elif correction < -P_FEEDBACK_MAX_CORRECTION_DEG:
                        correction = -P_FEEDBACK_MAX_CORRECTION_DEG
                        p_fb_clamp_count += 1
                        clamped = True

                    # [중요, 안티와인드업] EMA가 기억하는 값은 반드시 **클램프 뒤**,
                    # 즉 실제로 적용된 보정량이어야 한다. 클램프 전 값을 저장하면
                    # (예: 20도로 계산돼 5도로 잘렸는데 20도를 기억하면) 적용된 적도
                    # 없는 값이 다음 사이클에 계속 되먹임돼 클램프 상태가 필요 이상으로
                    # 오래 유지된다 - 적분 와인드업과 같은 구조다.
                    prev_p_correction[k] = correction

                    angles_to_send[k] += correction
                    if P_FEEDBACK_AUTO_GAIN:
                        recent_pfb_clamped[k].append(clamped)

                # [9차, §29.8] 자동 게인 조정 - 관절마다 독립적으로 판단한다.
                # 판단 주기/창 구조는 ADAPT_*(통신주기)와 동일, 6벌을 따로 돈다.
                if P_FEEDBACK_AUTO_GAIN:
                    gain_adapt_since_check += 1
                    for k in range(6):
                        if gain_cooldown_remaining[k] > 0:
                            gain_cooldown_remaining[k] -= 1
                    if (gain_adapt_since_check >= P_GAIN_ADAPT_CHECK_INTERVAL
                            and len(recent_pfb_clamped[0]) >= min(P_GAIN_ADAPT_WINDOW, 10)):
                        gain_adapt_since_check = 0
                        for k in range(6):
                            n_clamped = sum(recent_pfb_clamped[k])
                            if n_clamped > 0:
                                new_gain = max(P_FEEDBACK_GAIN_MIN,
                                               current_p_gain[k] - P_FEEDBACK_GAIN_STEP_DOWN)
                                if new_gain < current_p_gain[k] - 1e-9:
                                    log_fn(f"[P게인 자동조정] J{k+1}: 최근 "
                                           f"{len(recent_pfb_clamped[k])}사이클 중 클램프 "
                                           f"{n_clamped}회 -> 게인 {current_p_gain[k]:.2f} → "
                                           f"{new_gain:.2f} (낮춤)")
                                    current_p_gain[k] = new_gain
                                    gain_adapt_events.append((idx, k + 1, "낮춤", current_p_gain[k]))
                                gain_cooldown_remaining[k] = P_GAIN_ADAPT_COOLDOWN_CHECKS
                                recent_pfb_clamped[k].clear()
                            elif (gain_cooldown_remaining[k] == 0
                                  and current_p_gain[k] < P_FEEDBACK_GAIN_MAX[k]):
                                new_gain = min(P_FEEDBACK_GAIN_MAX[k],
                                               current_p_gain[k] + P_FEEDBACK_GAIN_STEP_UP)
                                if new_gain > current_p_gain[k] + 1e-9:
                                    log_fn(f"[P게인 자동조정] J{k+1}: 최근 "
                                           f"{len(recent_pfb_clamped[k])}사이클 클램프 없음 -> "
                                           f"게인 {current_p_gain[k]:.2f} → {new_gain:.2f} (올림)")
                                    current_p_gain[k] = new_gain
                                    gain_adapt_events.append((idx, k + 1, "올림", current_p_gain[k]))
                                recent_pfb_clamped[k].clear()

            # pymycobot 우회 고속 전송 (30ms -> 0.4ms). 실패하면 안전하게 원래 API로 폴백
            if not fast_send_angles(mc, angles_to_send, STREAM_SEND_SPEED):
                mc.send_angles(angles_to_send, STREAM_SEND_SPEED)

            # 명령값 기록은 통신이 필요 없어 '공짜' - 항상 기록
            cmd_times.append(time.time() - t_start)
            cmd_angles.append(angles_deg)
            cmd_targets.append(list(target))

            # [지연 FK] 실시간 데드라인이 있는 이 루프 안에서는 get_angles()만 하고
            # 저장한다. 순기구학 변환은 나중에(로봇이 멈춘 뒤) 한꺼번에 한다.
            # 정확도는 그대로, 이 루프의 사이클 비용만 줄어든다.
            do_measure = (measure_mode == "full")
            if do_measure:
                sample_measurement_raw_fn(stream_meas_times, stream_meas_angles, t_start)

            next_t += dispatch_period
            sleep_left = next_t - time.time()
            if sleep_left > 0:
                time.sleep(sleep_left)
                was_late = False
            else:
                was_late = True
                late_count += 1
                next_t = time.time()
            now = time.time()
            cycle_times.append(now - prev_t)
            # gc_events와 같은 time.time() 기준의 절대시각 - 나중에 "이 GC 이벤트가
            # 몇 번째 사이클 중에 일어났는가"를 찾을 때 이 배열과 대조한다.
            cycle_end_abs.append(now)

            # [6차] 적응형 재보정 판단.
            recent_late.append(was_late)
            adapt_since_check += 1
            if (adapt_since_check >= ADAPT_CHECK_INTERVAL
                    and len(recent_late) >= min(ADAPT_WINDOW, 10)):
                adapt_since_check = 0
                miss_recent = sum(recent_late) / len(recent_late)
                if miss_recent > ADAPT_MISS_HIGH:
                    cap = calib_period * ADAPT_MAX_FACTOR
                    new_period = min(cap, dispatch_period * ADAPT_STEP_UP)
                    if new_period > dispatch_period + 1e-6:
                        log_fn(f"[적응형 재보정] 최근 미달 {miss_recent*100:.0f}% "
                               f"({len(recent_late)}사이클 중) -> 주기 "
                               f"{dispatch_period*1000:.1f}ms → {new_period*1000:.1f}ms")
                        dispatch_period = new_period
                        adapt_events.append((idx, "상향", dispatch_period))
                        recent_late.clear()   # 새 주기 기준으로 다시 관찰 시작
                    elif not adapt_cap_warned and dispatch_period >= cap - 1e-6:
                        log_fn(f"[적응형 재보정] 상한({cap*1000:.1f}ms, calib의 "
                               f"{ADAPT_MAX_FACTOR}배) 도달 - 더 안 늘립니다. "
                               "환경이 계속 나쁜 것으로 보입니다(USB/시리얼 확인 권장).")
                        adapt_cap_warned = True
                elif miss_recent < ADAPT_MISS_LOW and dispatch_period > calib_period * 1.001:
                    new_period = max(calib_period, dispatch_period * ADAPT_STEP_DOWN)
                    if new_period < dispatch_period - 1e-6:
                        log_fn(f"[적응형 재보정] 최근 미달 {miss_recent*100:.0f}% (여유있음) "
                               f"-> 주기 {dispatch_period*1000:.1f}ms → {new_period*1000:.1f}ms "
                               "(원래값 쪽으로 회복)")
                        dispatch_period = new_period
                        adapt_events.append((idx, "복원", dispatch_period))
                        recent_late.clear()
            prev_t = now

        # [9차, GC 조사] 관찰이 끝났으니 콜백을 반드시 해제한다 - 안 지우면
        # 다음 실행 때 또 append돼서 콜백이 중첩되고, gc_events에 이번 실행과
        # 무관한 이벤트까지 섞여 들어간다.
    finally:
        # [9차, GC 조사] 루프 중간에 예외(로봇 통신 오류 등)가 나도
        # 콜백이 남지 않도록 반드시 여기서 해제한다.
        gc.callbacks.remove(_on_gc)

    if cycle_times:
        arr = np.array(cycle_times)
        cycle_mean = float(arr.mean())
        cycle_std = float(arr.std())
    else:   # 웨이포인트가 0개였던 극단적인 경우
        cycle_mean = cycle_std = 0.0
    late_ratio = late_count / max(1, len(curve_waypoints))

    return {
        "cycle_mean": cycle_mean,
        "cycle_std": cycle_std,
        "late_ratio": late_ratio,
        "target_period": dispatch_period,    # 적응형 조정이 있었다면 최종값이 반영됨
        "adapt_events": adapt_events,
        "calib_period": calib_period,
        # [9차 세션, 전송 리듬 조사] 원본 시계열을 그대로 얹는다.
        # mycobot_cycle_log.py가 저장하고 mycobot_cycle_rhythm_diagnostic.py가 분석한다.
        "cycle_times": cycle_times,
        # [9차, GC 조사] GC 발동 절대시각과 사이클별 종료 절대시각 - 이 둘을
        # 대조하면 "몇 번째 사이클에 GC가 끼었는가"를 알 수 있다.
        "gc_events": gc_events,
        "cycle_end_abs": cycle_end_abs,
        # [9차, §29] P 피드백 클램프 발동 횟수 - 상태줄에서 "게인이 너무 세서
        # 계속 클램프에 걸리고 있다"를 바로 알아볼 수 있게.
        "p_fb_clamp_count": p_fb_clamp_count,
        # [9차, §29.15] 데드밴드로 건너뛴 횟수 - 효과/설정 적정성 확인용.
        "p_fb_deadband_hits": p_fb_deadband_hits,
        # [9차, §29.8] 자동 게인 조정 - 관절별 조정 이력과 최종 게인(리스트, [J1..J6]).
        "p_gain_adapt_events": gain_adapt_events,
        "final_p_gain": current_p_gain,
        # [9차, §30.7] 오차모델 피드포워드 진단 - 평균 적용량으로 신뢰도가
        # 대체로 살아있었는지(값이 크면 정상 적용) 아니면 학습 밖이라 거의
        # 0으로 물러섰는지(값이 작으면 §30.6의 외삽 자제가 자주 발동) 가늠.
        "error_model_ff_mean_deg": (err_model_ff_abs_sum / err_model_ff_count
                                     if err_model_ff_count else None),
        "error_model_joints": sorted(_ERROR_MODEL.keys()) if APPLY_ERROR_MODEL_FEEDFORWARD else [],
    }
