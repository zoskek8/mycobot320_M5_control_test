"""
myCobot 320 인터랙티브 경로 에디터
====================================

CAD 프로그램처럼: 서로 다른 두 평면(XY/YZ/ZX)에서 같은 점을 각각 한 번씩
클릭하면 3D 좌표 하나가 완성됩니다. 완성된 점들을 순서대로 이으면 경로가
되고, 경로(선분)를 클릭하면 그 위치에 새 조절점이 추가됩니다. 점을
드래그하면 경로가 그쪽으로 휘어지고, 손을 뗀 순간에만 안전성을 재검증합니다.

의존 파일:
    같은 폴더에 mycobot_smooth_planner.py 가 있어야 합니다.
    (IK, 충돌검사, 데드존 데이터, 로봇 통신 함수를 그대로 재사용)

사전 설치:
    pip3 install PyQt5

실행:
    python3 mycobot_path_editor.py
"""

import matplotlib
matplotlib.use("Qt5Agg")  # mycobot_smooth_planner가 pyplot을 임포트하기 전에 백엔드 고정

import sys
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D 프로젝션 등록용)
# [경로 방식 변경] 이제 곡선을 '모든 점을 지나는 보간 스플라인'이 아니라
# '앵커-제어점을 번갈아가며 잇는 2차 베지어 체인'으로 그린다. PchipInterpolator는
# 더 이상 안 쓴다 - 아래 _bezier_chain_eval이 그 자리를 대신한다.
from scipy.spatial.transform import Rotation, Slerp
try:
    # [패치 5] 자세를 C1으로 보간하기 위해 필요 (scipy >= 1.2).
    from scipy.spatial.transform import RotationSpline
    _HAS_ROT_SPLINE = True
except ImportError:      # 구버전 scipy면 기존 Slerp로 자동 폴백
    RotationSpline = None
    _HAS_ROT_SPLINE = False
import math
# [8차, §23.12 정리] distance_transform_edt / to_rgba / patheffects는 그리기
# 코드와 함께 mycobot_canvas_draw.py로 옮겨가서 여기서는 더 이상 안 쓴다.

from mycobot_kinematics import (
    chain, active_indices, JOINT_LIMITS_DEG,
    quick_prefilter, natural_pose_at,
    find_robot_port, matrix_to_rxryrz,
    solve_pose_ik, jacobian_condition_number, analytic_jacobian_6d,
    check_self_collision, within_joint_limits,
    # [8차, §23.12 정리] _reachable_grid / _collision_only_grid는 배경 슬라이스
    # 코드와 함께 mycobot_canvas_draw.py로 갔다 - 여기선 더 이상 안 쓴다.
    _x_min, _y_min, _z_min, _step,
    _nx, _ny, _nz, _grid_loaded, _safe_indices,
    DESK_SAFETY_MARGIN_MM, CONDITION_NUMBER_MAX,
    # get_current_q_full 은 일부러 import하지 않는다 - §6.1 참조.
    #   검증에 로봇의 현재 자세를 쓰면 재현성이 깨진다. import가 없으면 실수도 없다.
    SETTLE_DELAY_SEC, MOVE_TIMEOUT_SEC, TRAJ_SAMPLE_SEC,
    ALIGN_ANGLES, SPEED,
    joint_offset_correction,
    read_angles,   # [10차, §39] USE_FAST_GET_ANGLES 스위치를 보고 빠른 읽기/폴백을 고른다
    AsyncAngleReader,   # [10차, §45] 파이프라인 읽기 - APPLY_ASYNC_MEASUREMENT_PIPELINE용
)

# ---------------------------------------------------------------------------
# [6차 분할] 이 파일이 2800줄을 넘어가면서 `def` 한 줄이 통째로 빠져도 눈에 안
# 띄는 사고가 실제로 있었다(_curve_ok_with_substitution 버그). GUI에 의존하지
# 않는 두 덩어리를 별도 모듈로 뺐다:
#   mycobot_curve_math : 베지어/재샘플링/경로검증 (순수 함수, Qt·로봇 무관)
#   mycobot_stream_exec: 주기 자동보정 + 실시간 스트리밍 전송 루프 + 관련 상수
# 이 파일은 GUI(창·그리기·이벤트)와 그 둘을 엮는 역할만 남는다.
# [성능] 파일 분할 자체는 실행속도에 영향이 없다 - import한 함수 호출과 같은
# 파일의 메서드 호출은 파이썬 수준에서 비용이 사실상 같다. 얻는 것은 유지보수성
# (가독성·버그 발견 용이성·GUI 없이 단위테스트 가능)이지 속도가 아니다.
# ---------------------------------------------------------------------------
import mycobot_curve_math as cm
import mycobot_stream_exec as se
import mycobot_result_plots as rp
import mycobot_pose_log as pose_log
import mycobot_curve_log as curve_log
import mycobot_cycle_log as cycle_log
import mycobot_status_msg as status_msg
import mycobot_error_model as err_model
from mycobot_curve_math import PathPoint, bezier_chain_eval as _bezier_chain_eval
from mycobot_stream_exec import (
    MIN_DISPATCH_PERIOD_SEC, STREAM_TCP_SPEED_MMS, STREAM_STEP_MM,
    STEP_MATCHES_REAL_CYCLE, MEASURE_CYCLE_SEC, STREAM_SEND_SPEED,
    ABSORB_SPIKES_TEST, ABSORB_TARGET_PERIOD_SEC,
    APPLY_P_FEEDBACK_CONTROL, P_FEEDBACK_GAIN,
    P_FEEDBACK_AUTO_GAIN, P_FEEDBACK_GAIN_MAX,
    P_GAIN_SOFTSTART, P_GAIN_SOFTSTART_CYCLES,
    P_FEEDBACK_DEADBAND_DEG, P_FEEDBACK_SMOOTH_ALPHA, P_FEEDBACK_KD,
    APPLY_ERROR_MODEL_FEEDFORWARD,
    APPLY_POSE_DEPENDENT_LAG_COMPENSATION,   # [10차, §43] A/B용 체크박스 추가
)
# [7차 분할, §22] _show_result_plots/_next_trace_dir/_save_figure는
# mycobot_result_plots(순수 함수, Qt·로봇 무관)로 뺐다. PathEditor에는
# 아래 3개의 얇은 위임 메서드만 남는다 - §17.7과 같은 패턴.

# ---------------------------------------------------------
# 결과 버전 관리 - 아래 파일 중 하나라도 고치면 이 숫자를 1 올릴 것
#   mycobot_path_editor.py / mycobot_kinematics.py /
#   mycobot_curve_math.py / mycobot_stream_exec.py / mycobot_result_plots.py /
#   mycobot_canvas_draw.py
#   ([6차] 분할로 4개, [7차, §22] 분할로 5개, [8차, §23.13] 분할로 6개가 됐다)
# ---------------------------------------------------------
# 실행 결과 그래프 4장이 RESULTS_ROOT_DIR/{CODE_VERSION}_figure/ 에 자동 저장되므로,
# 이 번호만 보면 어떤 코드 상태에서 나온 결과인지 바로 구분된다.
CODE_VERSION = "ver_3.0.1"   # 수정할 때마다 소수점 뒤 숫자만 올릴 것 (ver_2.2, ver_2.3, ...)
                            # [9차 세션] 2.9에서 여러 세션 머무른 뒤 3.0으로 올림 - 그 사이
                            # 누적된 변경: 자세의존 지연보정(§27.7-3), TCP속도 GUI(§28.6),
                            # P 피드백 제어(§29, 이 버전에서 처음 ON으로 실측).
RESULTS_ROOT_DIR = "/home/kms/바탕화면/curve_line_test"

# [8차 세션 추가, §23 진동 조사 전용] True로 켜면 종료 시 정렬자세 복귀를
# 건너뛴다. 원래는 종료(버튼/X 둘 다)마다 _move_to_align()이 실행돼
# "멈춘 자세"가 항상 사라지므로, GUI를 닫고 mycobot_vibration_diagnostic.py로
# 넘어가기 전에 그 자세를 그대로 유지하려면 이 플래그를 True로 켜야 한다.
# **테스트가 끝나면 반드시 False로 되돌릴 것** - 평소 운용 시에는 정렬자세
# 복귀가 안전을 위해 필요하다 (§4 등, 다음 세션 시작 시 알려진 자세에서
# 출발하기 위함).
SKIP_EXIT_ALIGN_FOR_TESTING = False

DENSE_SAMPLE_STEP_MM = 3    # 곡선 형태를 파악하기 위한 초기 조밀 샘플링 간격 (실행용 아님)

# [8차 세션, §23.13] 안전여유(margin) 스칼라장/색상 계산 + 배경 슬라이스/
# 곡선 샘플링/RDP 단순화/2D·3D 그리기를 mycobot_canvas_draw.py로 뺐다.
# self를 전혀 안 쓰는 순수 함수들이었다 - §17.7/§22와 같은 패턴.
import mycobot_canvas_draw as cd

# --- 스트리밍(브레이크 없음) 모드 설정 ---
# fresh_mode(1)은 이전 명령을 버리고 최신 명령만 즉시 실행 -> 웨이포인트마다 감속하지 않음.
# 대신 목표에 도달하기 전에 다음 목표를 받으므로 코너를 조금 잘라먹는데,
# 그 양은 웨이포인트 간격에 비례하므로 간격을 촘촘히 하면 오차가 작게 유지된다.
# [실측으로 밝혀진 사실]  ※ [패치 3]에서 수치 갱신
#   한 사이클의 하한은 시리얼 왕복이 아니라 ATOM 펌웨어의 프레임 소화속도이며
#   약 28ms(MIN_DISPATCH_PERIOD_SEC)다. 예전 주석의 '44ms'는 단방향 전송
#   도입 이전 값으로, 지금은 어디에도 해당하지 않는다.
#   -> 속도 손잡이는 STREAM_TCP_SPEED_MMS 하나뿐이고,
#      STREAM_STEP_MM 은 거기서 자동으로 유도된다 (아래 참조).
#
#   또한 "N번에 1번만 측정"하면 측정하는 사이클만 느려져 전송 리듬이 불규칙해지고,
#   그 리듬이 그대로 속도 톱니로 나타난다. 매 사이클 동일한 작업을 해야 균일해진다.
# [핵심] 읽기(get_coords/get_angles)는 ATOM 펌웨어 차원에서 1회당 약 20ms가 걸린다.
#   (Elephant Robotics 공식 이슈 #53: "읽기는 50Hz가 최대")
#   매 사이클 2회 읽으면 40ms 이상이 측정에만 소모되어 움직임 자체가 끊긴다.
#   -> 기본은 '단방향'(쓰기 전용). 응답을 기다리지 않으므로 사이클이 수 ms로 줄어
#      전송 주기를 정확히 지킬 수 있고, 그 결과 속도가 균일해진다.
#   [측정 전략]
#     명령값(무엇을 언제 보냈는지)을 기록하는 건 통신이 필요 없어 '공짜'다.
#     -> 항상 기록하며, 이것만으로 위치/자세/관절각/속도 프로파일 그래프를 그릴 수 있다.
#     실측이 필요하면 get_angles()만 읽고 좌표는 순기구학으로 직접 계산한다.
#     (get_coords까지 읽으면 읽기가 2회가 되어 부하가 두 배)
SNAP_RADIUS_MM = 25         # 추천 위치 이 반경 안에서 손을 떼면 정확히 그 좌표로 스냅
# [6차 신규] 추천을 '점'이 아니라 '범위'로 보여주기 위한 상한 2개.
#   REGION_MAX    : 화면에 영역으로 그려줄 국소검사 통과 후보 최대 개수.
#                   국소검사는 6차에서 시드를 실제 검증과 통일해 신뢰도가
#                   올라갔으므로 이 정도 모아도 오해를 주지 않는다.
#   FULLCHECK_MAX : 그중 '전체 경로 재검증'까지 돌릴 최대 개수. 이 단계가
#                   validate_path 한 번과 맞먹는 비용이라 여기는 늘리면 안 된다
#                   (§6.5 "검증이 끝나지 않는다" 버그의 원인이 이 값이었다).
REGION_MAX = 20
FULLCHECK_MAX = 4
# [6차 분할] 스트리밍 관련 상수(MIN_DISPATCH_PERIOD_SEC / STREAM_TCP_SPEED_MMS /
# STREAM_STEP_MM / STEP_MATCHES_REAL_CYCLE / MEASURE_CYCLE_SEC / STREAM_SEND_SPEED /
# 램프·적응형 재보정 ADAPT_*)는 전부 mycobot_stream_exec.py로 옮겼다.
# 위 import에서 이름을 그대로 가져오므로 이 파일 안의 기존 사용처는 안 바뀐다.
# 속도를 조절하려면 이제 mycobot_stream_exec.py의 STREAM_TCP_SPEED_MMS를 고칠 것.

CURVE_FIDELITY_MM = 5       # 실제 곡선과 이만큼(mm) 이내로만 벗어나면 됨 - 이 오차 안에서
                             # 최소한의 점만 남기므로, 굽은 구간엔 점이 촘촘히, 곧은 구간엔 듬성듬성 배치됨
# [중요] 이 허용오차가 곧 '명령값이 계획 경로에서 벗어나는 양'의 상한이 된다.
#   15mm로 헐겁게 두면 IK가 8mm쯤에서 대충 멈춰도 통과되어 경로 정확도가 그만큼 나빠진다.
#   실측: 15mm 설정 시 planned vs commanded 평균 오차 8.19mm 발생 -> 대폭 조임.
# ┌─ [패치 2] 2mm -> 0.2mm ──────────────────────────────────────────────────┐
# │ 허용오차 2mm는 웨이포인트 간격 2mm와 '같은 크기'였다. 즉 스텝만한 잔차가  │
# │ 매 명령마다, 매번 다른 방향으로 실렸다. 이제 간격의 1/10 이하로 두어      │
# │ 잔차가 경로 오차에 기여하지 않게 한다. (패치 1의 수렴 개선과 한 세트)     │
# │ 검증 시간은 늘어나지만 검증은 1회, 실행은 반복이므로 남는 장사.           │
# └──────────────────────────────────────────────────────────────────────────┘
CURVE_POS_TOL_MM = max(0.05, STREAM_STEP_MM / 10.0)   # ≈0.13mm (간격 1.26mm 기준)
CURVE_ORIENT_TOL_DEG = 0.5
# [참고: linear_x.py 계열 외부 스크립트의 max_step_delta_deg 진단]
# 웨이포인트 간격이 1.26mm로 이렇게 촘촘하면 정상적인 경우 스텝당 관절변화가
# 1도를 넘기 어렵다. 이 값을 넘으면 IK가 다중해 중 다른 분기로 튀었을 가능성이
# 높다는 신호로 본다 (임계값은 경험적 - 오탐이 잦으면 올릴 것).
STEP_JUMP_WARN_DEG = 5.0
CURVE_SPEED = 55            # 곡선 실행 시 각 웨이포인트 전송 속도
DISPATCH_DELAY_SEC = 0.05   # 웨이포인트 사이 전송 간격
ROUTE_VIA_ALIGN_BEFORE_START = False   # [6차 최종] §9 이슈2 - "정렬자세를 먼저 거치면
                                        # 백래시 접근방향이 통일돼 재현성이 좋아질 것"이라는
                                        # 가설로 기본 True였다. 서로 다른 곡선 2개로 A/B 비교한
                                        # 결과 둘 다 **OFF가 동등하거나 더 나았다**(cross-track
                                        # 1.60→1.54mm, 2.62→2.50mm - 두 번 다 OFF 우세, τ는
                                        # 거의 동일). 정렬 경유는 실행마다 1~2초를 더 쓰는데
                                        # 대가로 얻는 게 없다고 결론내고 기본을 OFF로 바꿨다.
                                        # 시작램프(§9 이슈3)와 같은 패턴 - "직관적으로 도움될
                                        # 것 같던 전처리 단계가 실측으로는 도움이 안 됐다".

# [정지오차 보정] mycobot_static_error_test.py 10자세 실측 기반 (kinematics.py
# joint_offset_correction 참고). J2/J5/J6 보정은 근거가 탄탄하지만 J1/J3/J4는
# 약하다(특히 J1은 방향이력 의심 - run_hysteresis_test로 추가 확인 전까지는
# 효과가 제한적일 수 있음). 그래도 켜두는 게 순손실은 아니라고 판단해 기본 on -
# 이상하면 이 스위치로 바로 끌 수 있다.
# [6차 신규] 이제 이 상수는 '기본값'일 뿐이다 - GUI의 체크박스
# (self.correction_check)가 실행 중에 이 전역변수를 직접 덮어쓴다(§9 이슈9,
# ON/OFF A/B 비교를 매번 소스코드 고쳐서 재시작할 필요 없게). 보정은
# execute_path() 시점에만 참조되므로(검증 자체는 좌표 계산일 뿐 보정과 무관),
# 체크박스를 바꾸고 '경로 검증'을 다시 안 해도 바로 '실행'만 다시 누르면
# 같은 곡선을 다른 보정 상태로 재실행할 수 있다.
APPLY_JOINT_OFFSET_CORRECTION = True


class PathEditor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("myCobot 320 경로 에디터")
        self.points = []
        self.pending = {}       # CAD 2클릭 조합 중인 임시 좌표
        self.drag_index = None
        self.mc = None
        self.curve_waypoints = []   # validate_path()가 채우는, 실제 실행에 쓸 촘촘한 웨이포인트
        self.bad_point_idx = None   # 검증 실패한 점 인덱스
        self.bad_segment = None     # 검증 실패한 구간 (i, i+1) 중 i
        self.suggestion = None      # 실패한 점의 추천 대체좌표 (x,y,z) - 전체검증까지 통과한 '확정' 1개
        # [6차 신규] 추천을 점 하나가 아니라 '범위'로 보여주기 위한 후보 목록.
        # [(x,y,z,verified), ...] - verified=True는 전체경로 재검증까지 통과한
        # 확정 후보(=self.suggestion), False는 국소검사만 통과한 유력 후보다.
        # 점 하나만 보여주면 "왜 하필 거기냐"를 알 수 없고 사용자가 옮길 여지도
        # 안 보이므로, 통과 가능한 영역 자체를 눈으로 보여준다.
        self.suggestion_region = []
        # [6차 신규] 되돌리기 스택. 예전 '마지막 점 취소'는 무슨 행동을 했든
        # 무조건 **끝 앵커**를 지웠다 - 방금 한 게 점 이동이나 구간 분할이었어도
        # 엉뚱한 점이 사라져서 되돌리기라기보다 '끝에서 자르기'에 가까웠다.
        # 이제 상태를 바꾸는 행동마다 스냅샷을 쌓고, 되돌리기는 **직전 행동**을
        # 그대로 취소한다(점 추가/이동/구간분할/전체지움 모두 대상).
        # 스냅샷 방식을 택한 이유: 행동별 역연산을 따로 짜면 불변식(앵커-제어점
        # 교대 배열)이 깨지는 경우를 하나씩 다 막아야 하는데, 점 개수가 많지
        # 않아 통째로 저장하는 게 훨씬 안전하고 단순하다.
        self.undo_stack = []        # [(라벨, [PathPoint.snapshot(), ...]), ...]
        self.UNDO_MAX = 50

        self.z_value = 200.0
        self.x_value = 0.0
        self.y_value = 0.0

        self._build_ui()
        self._redraw_all()
        self._startup_align()

    # ---------------------------------------------------------
    # UI 구성
    # ---------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)

        self.fig_xy, self.ax_xy, self.canvas_xy = self._make_canvas()
        self.fig_yz, self.ax_yz, self.canvas_yz = self._make_canvas()
        self.fig_xz, self.ax_xz, self.canvas_xz = self._make_canvas()

        self.fig_3d = Figure(figsize=(5, 5))
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self.canvas_3d = FigureCanvas(self.fig_3d)

        self.z_slider = self._make_slider(int(_z_min), int(_z_min + _nz * _step), self._on_z_slider)
        self.x_slider = self._make_slider(int(_x_min), int(_x_min + _nx * _step), self._on_x_slider)
        self.y_slider = self._make_slider(int(_y_min), int(_y_min + _ny * _step), self._on_y_slider)
        # [순서 주의] setValue()는 valueChanged 시그널을 즉시 발생시키고,
        #   그 콜백(_on_z_slider 등)이 _update_slider_labels()로 아래 라벨을 참조한다.
        #   따라서 라벨을 먼저 만들어 두지 않으면 시작하자마자 AttributeError가 난다.
        self.z_slider_label = QtWidgets.QLabel()
        self.x_slider_label = QtWidgets.QLabel()
        self.y_slider_label = QtWidgets.QLabel()

        self.z_slider.setValue(int(self.z_value))
        self.x_slider.setValue(int(self.x_value))
        self.y_slider.setValue(int(self.y_value))
        self._update_slider_labels()

        grid.addWidget(QtWidgets.QLabel("XY 평면 (X가 세로) — 'Z 높이' 단면. 초록이 진할수록 데드존 경계에서 여유(mm)가 큼 (등고선=같은 여유)"), 0, 0)
        grid.addWidget(self.canvas_xy, 1, 0)
        z_row = QtWidgets.QHBoxLayout()
        z_row.addWidget(QtWidgets.QLabel("배경 단면 Z:"))
        z_row.addWidget(self.z_slider)
        z_row.addWidget(self.z_slider_label)
        grid.addLayout(z_row, 2, 0)

        grid.addWidget(QtWidgets.QLabel("YZ 평면 (Z가 세로) — 'X 위치' 단면. 초록이 진할수록 데드존 경계에서 여유(mm)가 큼 (등고선=같은 여유)"), 0, 1)
        grid.addWidget(self.canvas_yz, 1, 1)
        x_row = QtWidgets.QHBoxLayout()
        x_row.addWidget(QtWidgets.QLabel("배경 단면 X:"))
        x_row.addWidget(self.x_slider)
        x_row.addWidget(self.x_slider_label)
        grid.addLayout(x_row, 2, 1)

        grid.addWidget(QtWidgets.QLabel("XZ 평면 (Z가 세로) — 'Y 위치' 단면. 초록이 진할수록 데드존 경계에서 여유(mm)가 큼 (등고선=같은 여유)"), 0, 2)
        grid.addWidget(self.canvas_xz, 1, 2)
        y_row = QtWidgets.QHBoxLayout()
        y_row.addWidget(QtWidgets.QLabel("배경 단면 Y:"))
        y_row.addWidget(self.y_slider)
        y_row.addWidget(self.y_slider_label)
        grid.addLayout(y_row, 2, 2)

        grid.addWidget(QtWidgets.QLabel("3D 미리보기"), 0, 3)
        grid.addWidget(self.canvas_3d, 1, 3)

        self.status_label = QtWidgets.QLabel(
            "새 점: 서로 다른 두 평면에서 같은 점을 각각 클릭하세요 (예: XY에서 한 번, YZ에서 한 번)"
        )
        self.status_label.setWordWrap(True)
        self.validate_btn = QtWidgets.QPushButton("경로 검증")
        self.execute_btn = QtWidgets.QPushButton("실행 (로봇 이동)")
        self.execute_btn.setEnabled(False)
        self.clear_btn = QtWidgets.QPushButton("전체 지우기")
        self.undo_btn = QtWidgets.QPushButton("마지막 행동 취소 (Ctrl+Z)")
        self.exit_btn = QtWidgets.QPushButton("종료 (정렬자세로 복귀 후 닫기)")
        # [8차 세션, §23.10] 경로(곡선) 전체를 불러와 편집/재검증/재실행
        # 가능하게 하는 버튼. 최종 자세 하나만 불러오는 별도 버튼("로그에서
        # 자세 불러오기")도 처음엔 만들었으나, 경로를 불러와 실행하면 그
        # 안에서 로봇이 최종적으로 어느 자세든 거치게 되므로 상위 기능인
        # 이 버튼과 중복이라 뺐다(§23.11) - 자세 로그 자체(mycobot_pose_log.py)와
        # 곡선 실행 시 자동 기록은 그대로 남아있다 - mycobot_vibration_diagnostic.py가
        # "비슷한 과거 자세 찾기"에 계속 쓴다.
        self.load_curve_btn = QtWidgets.QPushButton("저장된 경로 불러오기")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("큐 모드 (웨이포인트마다 감속, 톱니 있음)", "queue")
        self.mode_combo.addItem("스트리밍 모드 (브레이크 없음, 코너 약간 잘림)", "stream")
        self.mode_combo.setCurrentIndex(1)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.measure_combo = QtWidgets.QComboBox()
        self.measure_combo.addItem("명령값만 (가장 부드러움)", "none")
        self.measure_combo.addItem("매번 실측 (거침)", "full")
        self.measure_combo.setCurrentIndex(1)   # 기본값: 매번 실측
        self.measure_combo.currentIndexChanged.connect(self._on_measure_changed)

        # [6차 신규, §9 이슈9] 관절오프셋보정 ON/OFF 토글 - 예전엔 소스코드의
        # APPLY_JOINT_OFFSET_CORRECTION 상수를 손으로 고치고 재시작해야 했다.
        # 이제 체크박스로 바로 켜고 끌 수 있다. 검증(validate_path)은 좌표
        # 계산이라 보정과 무관하므로, 체크박스만 바꾸고 검증 없이 바로 '실행'을
        # 다시 눌러 같은 곡선을 ON/OFF로 번갈아 실행해 비교할 수 있다.
        self.correction_check = QtWidgets.QCheckBox(
            f"관절오프셋보정 {'ON' if APPLY_JOINT_OFFSET_CORRECTION else 'OFF'}")
        self.correction_check.setChecked(APPLY_JOINT_OFFSET_CORRECTION)
        self.correction_check.stateChanged.connect(self._on_correction_toggled)

        # [9차 세션, §28.6] "속도 바꿀 때마다 소스코드 STREAM_TCP_SPEED_MMS를
        # 손으로 고치고 재시작해야 하냐"는 질문에 대한 답 - 이제 GUI에서 바로
        # 바꿀 수 있다. mode_combo/measure_combo와 같은 이유로 바뀌면 웨이포인트
        # 간격도 바뀌므로(_stream_step_mm) 재검증이 필요하다 - _on_speed_changed가
        # _on_measure_changed와 같은 패턴으로 처리한다.
        self.speed_spin = QtWidgets.QSpinBox()
        self.speed_spin.setRange(5, 90)
        self.speed_spin.setSuffix(" mm/s")
        self.speed_spin.setValue(int(STREAM_TCP_SPEED_MMS))
        self.speed_spin.valueChanged.connect(self._on_speed_changed)

        # [9차 세션, §29] P 피드백 제어 - 위 mycobot_stream_exec.py의 스위치
        # 설명 참고. 관절오프셋보정 체크박스와 같은 이유로 좌표 계산과
        # 무관해 재검증 불필요 - 켜고 바로 '실행'하면 그 상태로 반영된다.
        self.pfb_check = QtWidgets.QCheckBox(f"P 피드백 {'ON' if APPLY_P_FEEDBACK_CONTROL else 'OFF'}")
        self.pfb_check.setChecked(APPLY_P_FEEDBACK_CONTROL)
        self.pfb_check.stateChanged.connect(self._on_pfb_toggled)

        self.pfb_gain_spin = QtWidgets.QDoubleSpinBox()
        self.pfb_gain_spin.setRange(0.0, 1.0)
        self.pfb_gain_spin.setSingleStep(0.05)
        self.pfb_gain_spin.setDecimals(2)
        self.pfb_gain_spin.setPrefix("k=")
        # [9차 세션, §29.8] 관절별 독립 게인으로 바뀌면서 P_FEEDBACK_GAIN은
        # 이제 [J1..J6] 리스트다 - 이 스핀박스는 "6관절에 동일하게 broadcast할
        # 시작값"만 다룬다. 시작은 항상 균일하게(§29.5까지 안전 확인된 0.3
        # 기준) 두고, 실행 중 관절별로 갈라지는 건 자동조정(§29.8, 관절별
        # 상한이 다름)에 맡긴다 - 시작점까지 관절별로 손으로 나눠 넣게 하면
        # GUI만 복잡해지고 실익은 적다.
        self.pfb_gain_spin.setValue(P_FEEDBACK_GAIN[0])
        self.pfb_gain_spin.valueChanged.connect(self._on_pfb_gain_changed)

        # [9차 세션, §29.5/§29.8] 자동 게인 조정 - 위 스핀박스는 이게 켜지면
        # "전 관절 공통 시작 게인"으로 의미가 바뀐다. 실행 중에는 관절별로
        # 독립적으로 오르내린다 - §29.7 실측에서 J2/J3(중력의존 정적오차·
        # 백래시 이봉불안정성)만 개선이 먼저 멈추는 게 확인돼서, 관절별 상한을
        # 다르게 뒀다(mycobot_stream_exec.py의 P_FEEDBACK_GAIN_MAX 리스트 참고).
        self.pfb_auto_check = QtWidgets.QCheckBox(f"자동조정 {'ON' if P_FEEDBACK_AUTO_GAIN else 'OFF'}")
        self.pfb_auto_check.setChecked(P_FEEDBACK_AUTO_GAIN)
        self.pfb_auto_check.stateChanged.connect(self._on_pfb_auto_toggled)

        # [9차 세션, §29.15] 게인 소프트스타트 - 시작 과도응답(overshoot) 억제.
        # 게인을 처음 N사이클 동안 0에서 선형으로 올린다. §20에서 폐기한
        # 시작가속램프와는 다른 것이다(그건 명령 속도 자체를 늦췄고, 이건
        # 명령 궤적은 그대로 두고 피드백 세기만 서서히 붙인다) -
        # mycobot_stream_exec.py 상수 설명 참고.
        self.pfb_soft_check = QtWidgets.QCheckBox(f"소프트스타트 {'ON' if P_GAIN_SOFTSTART else 'OFF'}")
        self.pfb_soft_check.setChecked(P_GAIN_SOFTSTART)
        self.pfb_soft_check.stateChanged.connect(self._on_pfb_softstart_toggled)

        # [9차 세션, §29.15] 진동 억제 - 데드밴드/평활화. 둘 다 "중립값이 곧 OFF"라
        # 별도 체크박스 없이 스핀박스 하나씩으로 끝난다(0.0 / 1.0이면 예전 동작).
        # 상세 설명은 mycobot_stream_exec.py 상수 주석 참고.
        self.pfb_deadband_spin = QtWidgets.QDoubleSpinBox()
        self.pfb_deadband_spin.setRange(0.0, 1.0)
        self.pfb_deadband_spin.setSingleStep(0.05)
        self.pfb_deadband_spin.setDecimals(2)
        self.pfb_deadband_spin.setPrefix("데드밴드=")
        self.pfb_deadband_spin.setSuffix("도")
        self.pfb_deadband_spin.setValue(P_FEEDBACK_DEADBAND_DEG)
        self.pfb_deadband_spin.valueChanged.connect(self._on_pfb_deadband_changed)

        self.pfb_smooth_spin = QtWidgets.QDoubleSpinBox()
        self.pfb_smooth_spin.setRange(0.1, 1.0)
        self.pfb_smooth_spin.setSingleStep(0.1)
        self.pfb_smooth_spin.setDecimals(2)
        self.pfb_smooth_spin.setPrefix("평활화 α=")
        self.pfb_smooth_spin.setValue(P_FEEDBACK_SMOOTH_ALPHA)
        self.pfb_smooth_spin.valueChanged.connect(self._on_pfb_smooth_changed)

        # [9차 세션, §29.20] D항(PD 제어) - 오차 변화율에 비례한 보정. 0.0이면
        # 순수 P제어와 100% 동일. 속도 반전점처럼 오차가 급격히 벌어지는 구간에
        # 반응한다(mycobot_stream_exec.py 상수 주석 참고).
        self.pfb_kd_spin = QtWidgets.QDoubleSpinBox()
        self.pfb_kd_spin.setRange(0.0, 0.5)
        self.pfb_kd_spin.setSingleStep(0.01)
        self.pfb_kd_spin.setDecimals(3)
        self.pfb_kd_spin.setPrefix("Kd=")
        self.pfb_kd_spin.setValue(P_FEEDBACK_KD)
        self.pfb_kd_spin.setToolTip(
            "D항 게인(초). 0이면 순수 P제어.\n"
            "오차가 얼마나 빨리 벌어지는지에 반응 - 관절 속도 반전점처럼\n"
            "가속도 요구가 큰 구간에 효과가 있습니다. 권장 시작값 0.05.")
        self.pfb_kd_spin.valueChanged.connect(self._on_pfb_kd_changed)

        # [9차 세션, §30.7] 오차모델 피드포워드 - J1/J4/J6에만 적용(교차검증
        # 결과 그 세 관절만 곡선 간 일반화가 확인됨, J2/J3/J5는 PD에만 맡김).
        # mycobot_error_model_fit.py의 [4]로 저장한 모델이 있어야 효과가 있다 -
        # 없으면 켜도 조용히 아무 일도 안 일어난다.
        self.err_model_check = QtWidgets.QCheckBox(
            f"오차모델 FF {'ON' if APPLY_ERROR_MODEL_FEEDFORWARD else 'OFF'}")
        self.err_model_check.setChecked(APPLY_ERROR_MODEL_FEEDFORWARD)
        self.err_model_check.setToolTip(
            "곡선 무관 오차모델(§30) 피드포워드 - J1/J4/J6만.\n"
            "mycobot_error_model_fit.py [4]로 모델을 먼저 저장해야 합니다.\n"
            "실기 미검증 - A/B로 개선을 확인한 뒤 기본 ON으로 바꿀 것.")
        self.err_model_check.stateChanged.connect(self._on_err_model_toggled)

        # [10차 세션, §37/§43] 자세의존 지연(tau) 보정 - J2/J3/J4만 적용
        # (mycobot_tau_table.py에 표가 있는 관절만, §27.6/§37). 오차모델 FF와는
        # 완전히 별개 기능이라 서로 안 겹친다(오차모델은 J1/J5/J6). 실기 A/B가
        # 아직 없어(§37.4) 이번 세션에 체크박스를 추가한다 - err_model_check와
        # 똑같은 배선 원칙(전역변수+se모듈 동시 갱신, from-import 함정 회피).
        self.lag_comp_check = QtWidgets.QCheckBox(
            f"자세지연보정 {'ON' if APPLY_POSE_DEPENDENT_LAG_COMPENSATION else 'OFF'}")
        self.lag_comp_check.setChecked(APPLY_POSE_DEPENDENT_LAG_COMPENSATION)
        self.lag_comp_check.setToolTip(
            "자세별 추종지연(τ) look-ahead 보정(§27.7-3) - J2/J3/J4만.\n"
            "mycobot_tau_table.py의 실측 표를 써서 웨이포인트를 미리 당겨 보냅니다.\n"
            "오차모델 FF(J1/J5/J6)와는 별개 기능 - 서로 안 겹칩니다.\n"
            "실기 미검증 - A/B로 개선을 확인한 뒤 기본 ON으로 바꿀 것.")
        self.lag_comp_check.stateChanged.connect(self._on_lag_comp_toggled)


        self.validate_btn.clicked.connect(self.validate_path)
        self.execute_btn.clicked.connect(self.execute_path)
        self.clear_btn.clicked.connect(self.clear_path)
        self.undo_btn.clicked.connect(self.undo_last)
        # [6차] 익숙한 Ctrl+Z도 같은 동작에 연결 - 되돌리기가 '진짜 undo'가 됐으니
        # 표준 단축키를 붙이는 게 자연스럽다.
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, activated=self.undo_last)
        self.exit_btn.clicked.connect(self._on_exit)
        self.load_curve_btn.clicked.connect(self._on_load_curve_from_log)

        # ── 컨트롤 배치 ────────────────────────────────────────────────────
        # [9차 세션, §29.18] 예전엔 이 모든 위젯이 btn_row 하나(가로 한 줄)에
        # 들어가 있었다 - 세션을 거치며 위젯이 6개에서 16개로 늘어나면서 창이
        # 화면 밖으로 밀려나가 오른쪽 버튼들을 못 누르는 상태가 됐다.
        #
        # 세 줄로 나누되, **그룹 경계가 실제 의미를 갖도록** 했다:
        #   1줄: 상태 readout (실행 결과가 숫자로 길게 나오므로 전체 폭을 준다)
        #   2줄: "경로 설정"(바꾸면 웨이포인트가 달라져 재검증 필요) vs
        #        "실시간 보정"(좌표 계산과 무관 - 검증 없이 바로 실행 가능)
        #        - 이건 장식이 아니라 이 코드베이스의 실제 동작 구분이다.
        #        _on_speed_changed는 curve_waypoints를 비우지만
        #        _on_correction_toggled/_on_pfb_*는 안 비운다.
        #   3줄: 동작 버튼 (편집 / 실행 / 세션)
        self.status_label.setObjectName("statusReadout")
        self.status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # --- 2줄: 설정 그룹 2개 ---
        setup_box = QtWidgets.QGroupBox("경로 설정  ·  바꾸면 다시 검증해야 합니다")
        setup_grid = QtWidgets.QGridLayout(setup_box)
        setup_grid.setContentsMargins(12, 14, 12, 10)
        setup_grid.setHorizontalSpacing(8)
        setup_grid.setVerticalSpacing(6)
        setup_grid.addWidget(self._field_label("실행 방식"), 0, 0)
        setup_grid.addWidget(self.mode_combo, 1, 0)
        setup_grid.addWidget(self._field_label("측정"), 0, 1)
        setup_grid.addWidget(self.measure_combo, 1, 1)
        setup_grid.addWidget(self._field_label("TCP 속도"), 0, 2)
        setup_grid.addWidget(self.speed_spin, 1, 2)
        setup_grid.setColumnStretch(0, 3)
        setup_grid.setColumnStretch(1, 2)
        setup_grid.setColumnStretch(2, 1)

        fb_box = QtWidgets.QGroupBox("실시간 보정  ·  검증 없이 바로 적용됩니다")
        fb_grid = QtWidgets.QGridLayout(fb_box)
        fb_grid.setContentsMargins(12, 14, 12, 10)
        fb_grid.setHorizontalSpacing(8)
        fb_grid.setVerticalSpacing(6)
        # 위: 켜고 끄는 것 / 아래: 숫자로 조절하는 것 - 성격이 다르므로 줄을 나눈다.
        # 열마다 "스위치와 그 값"이 세로로 짝을 이룬다(P 피드백↔k, 소프트스타트는
        # 값이 없어 빈칸). 남는 폭은 맨 끝 스페이서가 먹어서 위젯들이 왼쪽으로
        # 모이게 한다 - 안 그러면 열이 늘어나 스위치와 값이 멀어져 짝이 안 보인다.
        fb_grid.addWidget(self.correction_check, 0, 0)
        fb_grid.addWidget(self.pfb_check, 0, 1)
        fb_grid.addWidget(self.pfb_gain_spin, 1, 1)
        fb_grid.addWidget(self.pfb_auto_check, 0, 2)
        fb_grid.addWidget(self.pfb_deadband_spin, 1, 2)
        fb_grid.addWidget(self.pfb_soft_check, 0, 3)
        fb_grid.addWidget(self.pfb_smooth_spin, 1, 3)
        fb_grid.addWidget(self.pfb_kd_spin, 1, 4)
        fb_grid.addWidget(self.err_model_check, 0, 5)
        fb_grid.addWidget(self.lag_comp_check, 1, 5)
        fb_grid.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding,
                                              QtWidgets.QSizePolicy.Minimum), 0, 6, 2, 1)

        settings_row = QtWidgets.QHBoxLayout()
        settings_row.setSpacing(10)
        settings_row.addWidget(setup_box, 3)
        settings_row.addWidget(fb_box, 5)

        # --- 3줄: 동작 버튼 ---
        # 왼쪽=편집(되돌리기/지우기), 가운데 여백, 오른쪽=실행 흐름(검증→실행)과
        # 세션(불러오기/종료). 로봇이 실제로 움직이는 '실행'만 강조색을 준다.
        self.execute_btn.setObjectName("primaryAction")
        self.validate_btn.setObjectName("validateAction")
        self.exit_btn.setObjectName("quietAction")
        self.undo_btn.setText("되돌리기  ⌃Z")
        self.clear_btn.setText("전체 지우기")
        self.execute_btn.setText("실행  ▸  로봇 이동")
        self.load_curve_btn.setText("경로 불러오기")
        self.exit_btn.setText("종료")
        self.exit_btn.setToolTip("정렬자세로 복귀한 뒤 창을 닫습니다.")

        action_row = QtWidgets.QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.undo_btn)
        action_row.addWidget(self.clear_btn)
        action_row.addWidget(self.load_curve_btn)
        action_row.addStretch(1)
        action_row.addWidget(self.validate_btn)
        action_row.addWidget(self.execute_btn)
        action_row.addWidget(self.exit_btn)

        bottom = QtWidgets.QVBoxLayout()
        bottom.setSpacing(10)
        bottom.addWidget(self.status_label)
        bottom.addLayout(settings_row)
        bottom.addLayout(action_row)
        grid.addLayout(bottom, 3, 0, 1, 4)

        self._apply_theme()

        for canvas, view in [(self.canvas_xy, "xy"), (self.canvas_yz, "yz"), (self.canvas_xz, "xz")]:
            canvas.mpl_connect("button_press_event", lambda e, v=view: self._on_click(e, v))
            canvas.mpl_connect("motion_notify_event", lambda e, v=view: self._on_motion(e, v))
            canvas.mpl_connect("button_release_event", lambda e, v=view: self._on_release(e, v))

        self.resize(1560, 900)

    # ---------------------------------------------------------
    # [9차 세션, §29.18] 외관 - "실험실 계측기 패널" 방향
    # ---------------------------------------------------------
    # 이 프로그램은 예쁜 앱이 아니라 **정밀 측정 도구**다(로봇을 실제로
    # 움직이고, 매 실행마다 ms/mm 단위 숫자를 읽는다). 그래서 소비자 앱
    # 스타일이 아니라 계측기 앞판을 기준으로 잡았다:
    #   - 차가운 슬레이트 계열 회색조. matplotlib 캔버스가 흰 배경이라
    #     주변 크롬은 한 톤 낮춰 도면/플롯이 주인공이 되게 한다.
    #   - 강조색은 딱 하나, 로봇이 실제로 움직이는 '실행' 버튼에만 준다.
    #     산업용 기계에서 기동 계열 조작부를 호박색으로 표시하는 관례를
    #     그대로 쓴다 - 되돌릴 수 없는 물리적 동작이라는 신호.
    #   - 상태줄은 등폭 글꼴 + 어두운 판으로 계측기 표시창처럼. 실제로
    #     여기 출력이 숫자 덩어리라 등폭이 읽기에 유리하다(기능적 선택).
    PALETTE = {
        "bg":        "#E9ECF1",   # 창 배경 - 차가운 밝은 슬레이트
        "panel":     "#FFFFFF",
        "line":      "#C6CEDA",   # 테두리
        "line_soft": "#DDE3EC",
        "ink":       "#232B36",   # 본문
        "ink_dim":   "#68758A",   # 라벨/보조
        "readout_bg": "#222A35",  # 상태 표시창
        "readout_fg": "#D9E2EF",
        "amber":     "#D98A16",   # 실행(로봇 이동) - 단 하나의 강조색
        "amber_dark": "#B87310",
        "teal":      "#0F6E7B",   # 검증 - 강조는 아니고 '주요 동작' 표시
        "teal_dark": "#0B5A64",   # 검증 버튼 테두리/글자
    }

    def _field_label(self, text):
        """설정 그룹 안의 작은 항목 라벨. 값 위에 얹는 캡션 역할."""
        lb = QtWidgets.QLabel(text)
        lb.setObjectName("fieldLabel")
        return lb

    def _apply_theme(self):
        p = self.PALETTE
        # 등폭 글꼴 후보 - 앞에서부터 설치된 것을 Qt가 고른다. 한글이 섞이므로
        # CJK 등폭을 먼저 두고, 없으면 일반 등폭으로 떨어진다.
        mono = "'D2Coding', 'Noto Sans Mono CJK KR', 'DejaVu Sans Mono', monospace"
        self.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background: {p['bg']};
            color: {p['ink']};
            font-size: 12px;
        }}
        /* 상태 표시창 - 계측기 LCD처럼 */
        QLabel#statusReadout {{
            background: {p['readout_bg']};
            color: {p['readout_fg']};
            border: 1px solid #161C24;
            border-radius: 6px;
            padding: 10px 13px;
            font-family: {mono};
            font-size: 12px;
            line-height: 150%;
        }}
        QGroupBox {{
            background: {p['panel']};
            border: 1px solid {p['line']};
            border-radius: 7px;
            margin-top: 9px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 11px;
            padding: 0 5px;
            color: {p['ink_dim']};
            font-weight: 600;
        }}
        QLabel#fieldLabel {{
            color: {p['ink_dim']};
            font-size: 11px;
            padding-left: 1px;
        }}
        QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {p['panel']};
            border: 1px solid {p['line']};
            border-radius: 5px;
            padding: 5px 8px;
            min-height: 19px;
            selection-background-color: {p['teal']};
        }}
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {p['ink_dim']};
        }}
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {p['teal']};
        }}
        QComboBox::drop-down {{ border: none; width: 18px; }}
        QSpinBox, QDoubleSpinBox {{ font-family: {mono}; }}
        QCheckBox {{ spacing: 7px; padding: 3px 0; }}
        QCheckBox::indicator {{
            width: 15px; height: 15px;
            border: 1px solid {p['line']};
            border-radius: 4px;
            background: {p['panel']};
        }}
        QCheckBox::indicator:hover {{ border-color: {p['ink_dim']}; }}
        QCheckBox::indicator:checked {{
            background: {p['teal']};
            border-color: {p['teal_dark']};
        }}
        /* 기본 버튼 - 조용하게 */
        QPushButton {{
            background: {p['panel']};
            border: 1px solid {p['line']};
            border-radius: 5px;
            padding: 7px 14px;
            min-height: 19px;
        }}
        QPushButton:hover {{ background: #F4F7FB; border-color: {p['ink_dim']}; }}
        QPushButton:pressed {{ background: {p['line_soft']}; }}
        QPushButton:disabled {{ color: #A9B3C2; background: #F2F4F8; border-color: {p['line_soft']}; }}
        /* 검증 - 실행 직전 단계라 한 단계 강조 */
        QPushButton#validateAction {{
            border: 1px solid {p['teal']};
            color: {p['teal_dark']};
            font-weight: 600;
        }}
        QPushButton#validateAction:hover {{ background: #E8F3F5; }}
        /* 실행 - 로봇이 실제로 움직인다. 유일한 채움 강조색. */
        QPushButton#primaryAction {{
            background: {p['amber']};
            border: 1px solid {p['amber_dark']};
            color: #FFFFFF;
            font-weight: 700;
            padding: 7px 20px;
        }}
        QPushButton#primaryAction:hover {{ background: {p['amber_dark']}; }}
        QPushButton#primaryAction:disabled {{
            background: #EDEFF3; border-color: {p['line_soft']}; color: #A9B3C2;
        }}
        /* 종료 - 파괴적이진 않지만 실행 흐름과 무관하므로 한 톤 낮춘다.
           단, 비활성 버튼처럼 보이면 안 되므로 테두리는 그대로 두고
           글자만 살짝 물린다(disabled의 #A9B3C2보다 뚜렷하게). */
        QPushButton#quietAction {{ color: #55627A; border: 1px solid {p['line']}; }}
        QPushButton#quietAction:hover {{ color: {p['ink']}; border-color: {p['ink_dim']}; }}
        QSlider::groove:horizontal {{
            height: 4px; background: {p['line_soft']}; border-radius: 2px;
        }}
        QSlider::sub-page:horizontal {{ background: {p['teal']}; border-radius: 2px; }}
        QSlider::handle:horizontal {{
            width: 13px; height: 13px; margin: -5px 0;
            background: {p['panel']};
            border: 2px solid {p['teal']};
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{ background: #E8F3F5; }}
        QToolTip {{
            background: {p['readout_bg']}; color: {p['readout_fg']};
            border: none; padding: 5px 8px; border-radius: 4px;
        }}
        """)

    def _make_canvas(self):
        fig = Figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        return fig, ax, canvas

    def _make_slider(self, lo, hi, callback):
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setMinimum(lo)
        s.setMaximum(hi)
        s.valueChanged.connect(callback)
        return s

    def _current_mode(self):
        return self.mode_combo.currentData()

    def _stream_step_mm(self, measure_mode=None):
        """[6차] 지금 선택된 측정모드에서 목표 TCP 속도가 실제로 나오는 웨이포인트 간격.

        간격 = 목표속도 x '실제' 사이클시간. 예전엔 실제 사이클 대신 하한(28ms)을
        써서, '매번 실측' 모드(사이클 33ms)에서는 구조적으로 목표속도가 나올 수
        없었다(45 목표인데 38만 나옴). 위 STEP_MATCHES_REAL_CYCLE 주석 참고.
        """
        if not STEP_MATCHES_REAL_CYCLE:
            return STREAM_STEP_MM
        if measure_mode is None:
            measure_mode = self.measure_combo.currentData()
        # [9차, §25 - 채택됨] 켜져 있으면 실측 사이클(33ms) 대신 흡수 목표주기로
        # 간격을 잡는다 - 그래야 주기를 48ms로 늘려도 속도(45mm/s)가 유지된다.
        if ABSORB_SPIKES_TEST and measure_mode == "full":
            period = ABSORB_TARGET_PERIOD_SEC
        else:
            period = MEASURE_CYCLE_SEC.get(measure_mode, MIN_DISPATCH_PERIOD_SEC)
        return float(min(6.0, max(0.5, STREAM_TCP_SPEED_MMS * period)))

    def _on_measure_changed(self, _idx):
        """[6차] 측정모드가 바뀌면 목표 간격도 바뀌므로(위 _stream_step_mm) 재검증이 필요.
        안 그러면 예전 간격으로 만든 웨이포인트를 새 주기로 쏘게 되어 속도가 어긋난다."""
        self.curve_waypoints = []
        self.execute_btn.setEnabled(False)
        if self._current_mode() == "stream" and STEP_MATCHES_REAL_CYCLE:
            step = self._stream_step_mm()
            self.status_label.setText(
                f"측정 방식이 바뀌었습니다 - 이 모드의 웨이포인트 간격은 {step:.2f}mm입니다 "
                f"(목표 {STREAM_TCP_SPEED_MMS:.0f}mm/s 기준). 다시 '경로 검증'을 눌러주세요.")
        else:
            self.status_label.setText("측정 방식이 바뀌었습니다. 다시 '경로 검증'을 눌러주세요.")

    def _on_correction_toggled(self, _state):
        """[6차 신규] 체크박스로 관절오프셋보정을 즉시 켜고 끈다.

        모듈 전역변수(APPLY_JOINT_OFFSET_CORRECTION)를 직접 덮어쓴다 - 이 값을
        참조하는 곳(execute_path 3곳, 상태줄 표시 1곳)이 전부 '읽는 시점'에
        전역을 다시 조회하므로, 여기서 값을 바꾸는 즉시 다음 실행부터 반영된다.
        검증(validate_path)은 좌표만 계산할 뿐 이 값을 안 쓰므로 재검증 불필요 -
        체크박스를 바꾸고 바로 '실행'을 눌러 같은 곡선을 ON/OFF로 비교하면 된다
        (§9 이슈9 - 실제 곡선에서 보정 효과를 A/B로 확인하는 용도).
        """
        global APPLY_JOINT_OFFSET_CORRECTION
        APPLY_JOINT_OFFSET_CORRECTION = self.correction_check.isChecked()
        state_txt = "ON" if APPLY_JOINT_OFFSET_CORRECTION else "OFF"
        self.correction_check.setText(f"관절오프셋보정 {state_txt}")
        self.status_label.setText(
            f"🔧 관절오프셋보정을 {state_txt}로 전환했습니다. 검증을 다시 할 필요 없이 "
            f"'실행'을 누르면 이 상태로 곧장 반영됩니다 (같은 곡선 ON/OFF 비교용).")

    def _on_speed_changed(self, value):
        """[9차 세션, §28.6] STREAM_TCP_SPEED_MMS를 실행 중 바로 바꾼다 - 예전엔
        mycobot_stream_exec.py 소스를 손으로 고치고 재시작해야 했다.

        [from-import 함정, §24 프로파일러 스크립트의 CallCounter와 같은 문제]
        이 파일은 `from mycobot_stream_exec import ... STREAM_TCP_SPEED_MMS ...`로
        값을 '복사'해왔다 - `se.STREAM_TCP_SPEED_MMS`만 바꾸면 이 파일의 로컬
        사본(`_stream_step_mm()`이 읽는 이름)은 그대로다. 그래서 두 군데를 같이
        바꾼다: 이 모듈의 전역(global)과 `se` 모듈 객체의 속성.

        [정지오차 보정 체크박스와 다른 점] 보정은 값이 바뀌어도 곡선 좌표
        계산과 무관해 재검증이 불필요했지만, 속도는 웨이포인트 간격 자체를
        바꾸므로(_stream_step_mm) 정확히 mode_combo/measure_combo가 바뀔 때와
        같은 이유로 재검증이 필요하다 - curve_waypoints를 비우고 실행 버튼을
        잠근다.
        """
        global STREAM_TCP_SPEED_MMS
        STREAM_TCP_SPEED_MMS = value
        se.STREAM_TCP_SPEED_MMS = value
        self.curve_waypoints = []
        self.execute_btn.setEnabled(False)
        self.status_label.setText(
            f"🚀 목표 TCP 속도를 {value}mm/s로 바꿨습니다 - 웨이포인트 간격이 "
            f"달라지므로 다시 '경로 검증'을 눌러주세요.")

    def _on_pfb_toggled(self, _state):
        """[9차 세션, §29] P 피드백(y=k·error) 즉시 켜고 끈다 - 관절오프셋보정
        체크박스와 같은 이유로 좌표 계산과 무관해 재검증 불필요.

        [from-import 함정] APPLY_P_FEEDBACK_CONTROL을 실제로 읽는 곳
        (run_streaming_dispatch)은 mycobot_stream_exec.py 안에 있으므로,
        이 파일의 전역만 바꾸면 반영되지 않는다 - se 모듈 속성도 같이 바꾼다
        (§28.6 _on_speed_changed와 동일 패턴).
        """
        global APPLY_P_FEEDBACK_CONTROL
        APPLY_P_FEEDBACK_CONTROL = self.pfb_check.isChecked()
        se.APPLY_P_FEEDBACK_CONTROL = APPLY_P_FEEDBACK_CONTROL
        state_txt = "ON" if APPLY_P_FEEDBACK_CONTROL else "OFF"
        self.pfb_check.setText(f"P 피드백 {state_txt}")
        note = ""
        if APPLY_P_FEEDBACK_CONTROL and self.measure_combo.currentData() != "full":
            note = "  ⚠️ '매번 실측' 모드가 아니면 실시간 측정이 없어 아무 효과가 없습니다."
        self.status_label.setText(
            f"🎛️ P 피드백 제어를 {state_txt}로 전환했습니다 (게인 k={self.pfb_gain_spin.value():.2f}). "
            f"검증 없이 바로 '실행'하면 반영됩니다.{note}")

    def _on_pfb_gain_changed(self, value):
        """[9차 세션, §29.8] P 게인(k)을 실행 중 바로 조절 - 6관절 전부에
        동일하게 broadcast한다(리스트로 재할당). 두 모듈 동기화는
        _on_pfb_toggled와 같은 이유(from-import 함정)."""
        global P_FEEDBACK_GAIN
        P_FEEDBACK_GAIN = [value] * 6
        se.P_FEEDBACK_GAIN = P_FEEDBACK_GAIN
        if self.pfb_check.isChecked():
            self.status_label.setText(f"🎛️ P 피드백 게인을 k={value:.2f}(전 관절 동일)로 바꿨습니다.")

    def _on_pfb_auto_toggled(self, _state):
        """[9차 세션, §29.5] 자동 게인 조정 ON/OFF. 켜면 pfb_gain_spin의 값은
        '시작 게인'이 되고, 실행 중 mycobot_stream_exec.run_streaming_dispatch가
        클램프 발동 여부를 보며 알아서 올리고/내린다(§29.5 실측 기반 상한
        적용 - 상수 설명은 mycobot_stream_exec.py 참고). 두 모듈 동기화는
        _on_pfb_toggled와 같은 이유(from-import 함정)."""
        global P_FEEDBACK_AUTO_GAIN
        P_FEEDBACK_AUTO_GAIN = self.pfb_auto_check.isChecked()
        se.P_FEEDBACK_AUTO_GAIN = P_FEEDBACK_AUTO_GAIN
        state_txt = "ON" if P_FEEDBACK_AUTO_GAIN else "OFF"
        self.pfb_auto_check.setText(f"자동조정 {state_txt}")
        if P_FEEDBACK_AUTO_GAIN:
            # [9차, §29.8 후속 수정] P_FEEDBACK_GAIN_MAX가 관절별 리스트로 바뀐 뒤로
            # 여기서 `:.2f`를 쓰면 TypeError가 난다(리스트에는 float 포맷 불가).
            # 지금은 전 관절 균일값(§29.12에서 되돌림)이지만 다시 갈릴 수 있으므로
            # 균일하면 값 하나로, 다르면 전부 나열하도록 처리한다.
            gmax = P_FEEDBACK_GAIN_MAX
            gmax_txt = (f"{gmax[0]:.2f}" if len(set(gmax)) == 1
                        else " ".join(f"J{j+1}={gmax[j]:.2f}" for j in range(6)))
            self.status_label.setText(
                f"🎛️ P 게인 자동조정 ON - 시작 k={self.pfb_gain_spin.value():.2f}에서 "
                f"실행 중 최대 {gmax_txt}까지 스스로 올리고, 클램프가 "
                f"보이면 즉시 낮춥니다. '실행'을 누르면 반영됩니다.")
        else:
            self.status_label.setText("🎛️ P 게인 자동조정 OFF - 스핀박스 값을 고정 게인으로 씁니다.")

    def _on_pfb_softstart_toggled(self, _state):
        """[9차 세션, §29.15] 게인 소프트스타트 ON/OFF - 시작 과도응답 억제.
        게인을 처음 P_GAIN_SOFTSTART_CYCLES 사이클 동안 0에서 선형으로 올린다.
        정상 구간 동작은 100% 그대로라 §29.13에서 확인한 개선 효과는 유지된다.
        두 모듈 동기화는 _on_pfb_toggled와 같은 이유(from-import 함정)."""
        global P_GAIN_SOFTSTART
        P_GAIN_SOFTSTART = self.pfb_soft_check.isChecked()
        se.P_GAIN_SOFTSTART = P_GAIN_SOFTSTART
        state_txt = "ON" if P_GAIN_SOFTSTART else "OFF"
        self.pfb_soft_check.setText(f"소프트스타트 {state_txt}")
        if P_GAIN_SOFTSTART:
            secs = P_GAIN_SOFTSTART_CYCLES * ABSORB_TARGET_PERIOD_SEC
            self.status_label.setText(
                f"🎛️ 게인 소프트스타트 ON - 처음 {P_GAIN_SOFTSTART_CYCLES}사이클"
                f"(약 {secs:.1f}초) 동안 게인을 0에서 서서히 올립니다. "
                f"시작 구간 과도응답(Figure 3 초반 튐)을 줄이는 용도입니다.")
        else:
            self.status_label.setText("🎛️ 게인 소프트스타트 OFF - 첫 사이클부터 풀게인으로 갑니다.")

    def _on_pfb_deadband_changed(self, value):
        """[9차 세션, §29.15-1] 데드밴드 - |오차|가 이 값 이하면 보정을 건너뛴다.
        읽기 노이즈(§23, ~0.1도)를 진짜 오차로 착각해 쫓아가며 떠는 것을 막는
        용도. 0.0이면 비활성(예전 동작 그대로). 두 모듈 동기화는
        _on_pfb_toggled와 같은 이유(from-import 함정)."""
        global P_FEEDBACK_DEADBAND_DEG
        P_FEEDBACK_DEADBAND_DEG = value
        se.P_FEEDBACK_DEADBAND_DEG = value
        if self.pfb_check.isChecked():
            if value > 0.0:
                self.status_label.setText(
                    f"🎛️ P 피드백 데드밴드 {value:.2f}도 - 이보다 작은 오차는 "
                    f"보정하지 않습니다(노이즈 추종에 의한 떨림 억제).")
            else:
                self.status_label.setText("🎛️ P 피드백 데드밴드 해제 - 모든 오차를 보정합니다.")

    def _on_pfb_smooth_changed(self, value):
        """[9차 세션, §29.15-2] 보정량 평활화(EMA) - 보정값을 직전 사이클과 섞어
        급변을 눌러 고주파 진동을 줄인다. 1.0이면 비활성(평활화 없음).
        [주의] 낮출수록 위상지연이 붙으므로(데드타임이 큰 시스템) 과하게 낮추면
        오히려 불안정해질 수 있다."""
        global P_FEEDBACK_SMOOTH_ALPHA
        P_FEEDBACK_SMOOTH_ALPHA = value
        se.P_FEEDBACK_SMOOTH_ALPHA = value
        if self.pfb_check.isChecked():
            if value < 1.0:
                self.status_label.setText(
                    f"🎛️ P 피드백 보정 평활화 α={value:.2f} - 보정 급변을 눌러 "
                    f"떨림을 줄입니다(낮출수록 강하지만 반응이 느려짐).")
            else:
                self.status_label.setText("🎛️ P 피드백 보정 평활화 해제.")

    def _on_pfb_kd_changed(self, value):
        """[9차 세션, §29.20] D항 게인(Kd, 단위=초). 0.0이면 순수 P제어.
        두 모듈 동기화는 _on_pfb_toggled와 같은 이유(from-import 함정)."""
        global P_FEEDBACK_KD
        P_FEEDBACK_KD = value
        se.P_FEEDBACK_KD = value
        if self.pfb_check.isChecked():
            if value > 0.0:
                self.status_label.setText(
                    f"🎛️ PD 제어 - D항 Kd={value:.3f}초. 오차가 급격히 벌어지는 구간"
                    f"(관절 속도 반전점 등)에 추가로 반응합니다.")
            else:
                self.status_label.setText("🎛️ D항 해제 - 순수 P제어로 동작합니다.")

    def _on_err_model_toggled(self, _state):
        """[9차 세션, §30.7] 오차모델 피드포워드 ON/OFF - J1/J4/J6만 적용.
        모델 파일(error_model_weights.json)이 없으면 켜도 조용히 아무 효과가
        없다(§30.7-[4]를 아직 안 돌린 경우). 두 모듈 동기화는 _on_pfb_toggled와
        같은 이유(from-import 함정)."""
        global APPLY_ERROR_MODEL_FEEDFORWARD
        APPLY_ERROR_MODEL_FEEDFORWARD = self.err_model_check.isChecked()
        se.APPLY_ERROR_MODEL_FEEDFORWARD = APPLY_ERROR_MODEL_FEEDFORWARD
        state_txt = "ON" if APPLY_ERROR_MODEL_FEEDFORWARD else "OFF"
        self.err_model_check.setText(f"오차모델 FF {state_txt}")
        if APPLY_ERROR_MODEL_FEEDFORWARD:
            n = len(se._ERROR_MODEL)
            if n == 0:
                self.status_label.setText(
                    "⚠️ 오차모델 피드포워드를 켰지만 저장된 모델이 없습니다 - "
                    "지금 켜도 효과가 없습니다. mycobot_error_model_fit.py의 "
                    "[4]로 먼저 J1/J4/J6 모델을 저장하세요.")
            else:
                joints = ", ".join(f"J{j+1}" for j in sorted(se._ERROR_MODEL))
                self.status_label.setText(
                    f"🎛️ 오차모델 피드포워드 ON - {joints}에 적용됩니다 "
                    f"(§30.7, 실기 미검증 - A/B로 확인할 것).")
        else:
            self.status_label.setText("🎛️ 오차모델 피드포워드 OFF.")

    def _on_lag_comp_toggled(self, _state):
        """[10차 세션, §37/§43] 자세의존 지연(τ) look-ahead 보정 ON/OFF -
        J2/J3/J4만 적용(mycobot_tau_table.py에 표가 있는 관절). 오차모델 FF와
        완전히 별개 기능 - 서로 안 겹친다. 두 모듈 동기화는 _on_err_model_toggled와
        같은 이유(from-import 함정)."""
        global APPLY_POSE_DEPENDENT_LAG_COMPENSATION
        APPLY_POSE_DEPENDENT_LAG_COMPENSATION = self.lag_comp_check.isChecked()
        se.APPLY_POSE_DEPENDENT_LAG_COMPENSATION = APPLY_POSE_DEPENDENT_LAG_COMPENSATION
        state_txt = "ON" if APPLY_POSE_DEPENDENT_LAG_COMPENSATION else "OFF"
        self.lag_comp_check.setText(f"자세지연보정 {state_txt}")
        if APPLY_POSE_DEPENDENT_LAG_COMPENSATION:
            self.status_label.setText(
                "🎛️ 자세지연보정 ON - J2/J3/J4에 적용됩니다 "
                "(§27.7-3/§37, 실기 미검증 - A/B로 확인할 것).")
        else:
            self.status_label.setText("🎛️ 자세지연보정 OFF.")

    def _on_mode_changed(self, _idx):
        """실행 방식이 바뀌면 웨이포인트 구성이 달라지므로 재검증이 필요"""
        self.curve_waypoints = []
        self.execute_btn.setEnabled(False)
        mode = self._current_mode()
        if mode == "stream":
            self.status_label.setText(
                "실행 방식: 스트리밍(브레이크 없음). 웨이포인트를 촘촘히 만들어 "
                "감속 없이 이어서 쫓아갑니다. 다시 '경로 검증'을 눌러주세요."
            )
        else:
            self.status_label.setText(
                "실행 방식: 큐 모드. 웨이포인트마다 정확히 멈췄다 가므로 정확하지만 톱니가 생깁니다. "
                "다시 '경로 검증'을 눌러주세요."
            )

    def _update_slider_labels(self):
        self.z_slider_label.setText(f"{self.z_value:.0f} mm")
        self.x_slider_label.setText(f"{self.x_value:.0f} mm")
        self.y_slider_label.setText(f"{self.y_value:.0f} mm")

    def _on_z_slider(self, v):
        self.z_value = float(v)
        self._update_slider_labels()
        self._redraw_all()

    def _on_x_slider(self, v):
        self.x_value = float(v)
        self._update_slider_labels()
        self._redraw_all()

    def _on_y_slider(self, v):
        self.y_value = float(v)
        self._update_slider_labels()
        self._redraw_all()

    # [8차 세션, §23.13] _bg_slice_xy/yz/xz, _sample_curve, _rdp_indices,
    # _draw_axis_indicator는 mycobot_canvas_draw.py로 뺐다(순수 함수, self 안 씀).

    # ---------------------------------------------------------
    # 시작 / 종료 - 정렬자세(ALIGN_ANGLES) 이동
    # ---------------------------------------------------------
    def _ensure_connected(self):
        if self.mc is not None:
            return True
        self.status_label.setText("🔌 로봇 연결 중...")
        QtWidgets.QApplication.processEvents()
        self.mc, port = find_robot_port()
        if self.mc is None:
            self.status_label.setText("❌ 로봇 연결 실패 (USB/전원/Transponder 모드 확인)")
            return False
        self.mc.power_on()
        time.sleep(2)
        self.mc.clear_error_information()
        self.mc.set_fresh_mode(0)
        return True

    def _move_to_angles(self, target_angles):
        """임의의 관절각으로 이동 + 완료 대기. _move_to_align도 이걸 재사용한다."""
        if not self._ensure_connected():
            return
        self.mc.send_angles(list(target_angles), SPEED)
        time.sleep(SETTLE_DELAY_SEC)
        t0 = time.time()
        while self.mc.is_moving():
            if time.time() - t0 > MOVE_TIMEOUT_SEC:
                break
            time.sleep(0.1)

    def _move_to_align(self):
        self._move_to_angles(ALIGN_ANGLES)

    def _on_load_curve_from_log(self):
        """[8차 세션 추가, §23.12] 저장된 경로(점 목록 전체)를 불러와
        현재 편집 화면에 그대로 복원한다 - 이후 드래그로 수정, '경로 검증',
        '실행'을 전부 다시 할 수 있다. 불러오기 자체도 '마지막 행동 취소'로
        되돌릴 수 있도록 undo_last()와 같은 패턴(_push_undo -> 교체 ->
        _invalidate_after_edit -> _redraw_all)을 그대로 따른다."""
        records = curve_log.list_recent(limit=curve_log.DEFAULT_LIST_LIMIT)
        if not records:
            QtWidgets.QMessageBox.information(
                self, "경로 로그 비어있음",
                "아직 저장된 경로가 없습니다 (curve_path_log.jsonl).\n"
                "곡선을 한 번 실행하면 자동으로 쌓입니다."
            )
            return

        items = []
        for rec in records:
            extra = rec.get("extra") or {}
            note = ""
            if extra.get("lag_sec") is not None:
                note = f"  [밀림 {extra['lag_sec']:.1f}s]"
            items.append(f"{rec.get('timestamp', '?')}  [{rec.get('label', '?')}]"
                        f"  ({rec.get('n_points', '?')}점){note}")

        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "저장된 경로 불러오기", "불러올 경로를 선택하세요 (최신 5개):",
            items, 0, False
        )
        if not ok or not choice:
            return

        idx = items.index(choice)
        rec = records[idx]

        self._push_undo("경로 불러오기 전")
        self.points = [PathPoint.from_snapshot(t) for t in curve_log.snapshots_as_tuples(rec)]
        self.pending = {}
        self.drag_index = None
        self._invalidate_after_edit()
        self._redraw_all()
        self.status_label.setText(
            f"📂 경로 불러옴: [{rec.get('label', '?')}] ({rec.get('timestamp', '?')}, "
            f"{rec.get('n_points', '?')}점) - 필요하면 드래그로 수정 후 '경로 검증'을 눌러주세요."
        )

    def _startup_align(self):
        self.status_label.setText("🤖 시작: 정렬자세로 이동 중...")
        QtWidgets.QApplication.processEvents()
        self._move_to_align()
        self.status_label.setText("✅ 정렬자세 도착. 새 점을 클릭해서 경로를 그려보세요.")

    def _on_exit(self):
        self._exit_handled = True
        if SKIP_EXIT_ALIGN_FOR_TESTING:
            self.status_label.setText("🤖 종료: [테스트모드] 정렬자세 복귀 건너뜀 - 현재 자세 유지")
            QtWidgets.QApplication.processEvents()
        else:
            self.status_label.setText("🤖 종료: 정렬자세로 복귀 중...")
            QtWidgets.QApplication.processEvents()
            self._move_to_align()
        self.close()

    def closeEvent(self, event):
        """창을 X 버튼으로 직접 닫는 경우에도 동일하게 정렬자세로 복귀 (종료 버튼과 중복 실행 방지).
        [8차 세션] SKIP_EXIT_ALIGN_FOR_TESTING=True면 복귀를 건너뛴다 - §23 진동 조사용."""
        if not getattr(self, "_exit_handled", False):
            self._exit_handled = True
            if SKIP_EXIT_ALIGN_FOR_TESTING:
                self.status_label.setText("🤖 종료: [테스트모드] 정렬자세 복귀 건너뜀 - 현재 자세 유지")
                QtWidgets.QApplication.processEvents()
            else:
                self.status_label.setText("🤖 종료: 정렬자세로 복귀 중...")
                QtWidgets.QApplication.processEvents()
                self._move_to_align()
        event.accept()

    # ---------------------------------------------------------
    # 클릭 / 드래그 처리
    # ---------------------------------------------------------
    def _axis_of(self, view):
        return {"xy": self.ax_xy, "yz": self.ax_yz, "xz": self.ax_xz}[view]

    # ---------------------------------------------------------
    # 앵커/제어점 관리
    #   불변식: points = [앵커, 제어, 앵커, 제어, 앵커, ...]
    #           항상 앵커로 시작하고 앵커로 끝난다 (길이 = 2*앵커수 - 1).
    #   구간 k(0-based)는 points[2k], points[2k+1], points[2k+2] 세 점으로 이루어진
    #   2차 베지어다. 즉 "구간마다 제어점 1개"가 구조적으로 보장된다.
    # ---------------------------------------------------------
    def _n_anchors(self):
        return (len(self.points) + 1) // 2

    def _n_segments(self):
        return max(0, self._n_anchors() - 1)

    def _append_anchor(self, p):
        """새 앵커를 경로 끝에 붙이고, 직전 앵커와의 사이에 제어점을 자동 생성한다.
        제어점 초기 위치는 두 앵커의 중점 - 즉 처음엔 직선으로 보이고,
        그 제어점을 드래그하면 그 구간만 휘어진다."""
        p.role = "anchor"
        if not self.points:
            self.points.append(p)
            return
        last = self.points[-1]
        ctrl = PathPoint((last.x + p.x) / 2.0, (last.y + p.y) / 2.0, (last.z + p.z) / 2.0,
                         role="ctrl")
        self.points.append(ctrl)
        self.points.append(p)

    def _split_segment(self, seg, u):
        """구간 seg를 파라미터 u(0~1)에서 De Casteljau로 분할한다.

        분할 전:  P0 --C-- P2          (한 구간, 제어점 1개)
        분할 후:  P0 --L0-- Q --L1-- P2 (두 구간, 제어점 2개)

        L0 = lerp(P0,C,u), L1 = lerp(C,P2,u), Q = lerp(L0,L1,u)
        이렇게 하면 **곡선 모양이 전혀 바뀌지 않은 채로** 구간만 둘로 쪼개진다.
        (사용자가 보내준 그림의 L0/L1/Q0 구성이 정확히 이것)
        반환: 새로 생긴 앵커 Q
        """
        i0 = 2 * seg
        P0 = np.array(self.points[i0].coord(), dtype=float)
        C = np.array(self.points[i0 + 1].coord(), dtype=float)
        P2 = np.array(self.points[i0 + 2].coord(), dtype=float)

        L0 = P0 + (C - P0) * u
        L1 = C + (P2 - C) * u
        Q = L0 + (L1 - L0) * u

        newL0 = PathPoint(*L0, role="ctrl")
        newQ = PathPoint(*Q, role="anchor")
        newL1 = PathPoint(*L1, role="ctrl")
        # [P0, C, P2] 구간을 [P0, L0, Q, L1, P2] 로 교체
        self.points[i0 + 1:i0 + 2] = [newL0, newQ, newL1]
        return newQ

    def _on_click(self, event, view):
        ax = self._axis_of(view)
        if event.inaxes != ax or event.button != 1:
            return

        idx = self._find_near_point(event, view)
        if idx is not None:
            # [6차] 드래그 시작 - 움직이기 '전' 상태를 저장해야 되돌릴 수 있다.
            self._push_undo(f"{idx+1}번 점 이동")
            self.drag_index = idx
            return

        seg = self._find_near_segment(event, view)
        if seg is not None:
            seg_i, u = seg
            self._push_undo(f"{seg_i+1}번 구간 분할")
            newp = self._split_segment(seg_i, u)
            self._validate_point(newp)
            self.curve_waypoints = []
            self.bad_point_idx = None
            self.bad_segment = None
            self.suggestion = None
            self.suggestion_region = []
            self.execute_btn.setEnabled(False)
            self._redraw_all()
            self.status_label.setText(
                f"경로가 두 구간으로 나뉘었습니다 (곡선 모양은 그대로). "
                f"새 앵커: ({newp.x:.0f}, {newp.y:.0f}, {newp.z:.0f}) · "
                f"구간 {self._n_segments()}개 / 제어점 {self._n_segments()}개")
            return

        # 새 점 - CAD 방식 2클릭
        if view == "xy":
            self.pending["y"] = event.xdata
            self.pending["x"] = event.ydata
        elif view == "yz":
            self.pending["y"] = event.xdata
            self.pending["z"] = event.ydata
        else:  # xz (X 가로, Z 세로)
            self.pending["x"] = event.xdata
            self.pending["z"] = event.ydata

        if all(k in self.pending for k in ("x", "y", "z")):
            p = PathPoint(self.pending["x"], self.pending["y"], self.pending["z"])
            self._push_undo("점 추가")
            self._append_anchor(p)   # 직전 앵커와의 사이에 제어점이 자동 생성됨
            self.pending = {}
            self._validate_point(p)
            self.curve_waypoints = []
            self.bad_point_idx = None
            self.bad_segment = None
            self.suggestion = None
            self.suggestion_region = []
            self.execute_btn.setEnabled(False)
            if self._n_segments() > 0:
                self.status_label.setText(
                    f"앵커 추가됨: ({p.x:.0f}, {p.y:.0f}, {p.z:.0f}) · "
                    f"구간 {self._n_segments()}개 / 제어점 {self._n_segments()}개"
                    f" — 다이아몬드(◇) 제어점을 끌면 그 구간이 휩니다")
            else:
                self.status_label.setText(
                    f"앵커 추가됨: ({p.x:.0f}, {p.y:.0f}, {p.z:.0f}) — 점을 하나 더 찍으면 구간이 생깁니다")
        else:
            have = set(self.pending.keys())
            missing = {"x", "y", "z"} - have
            self.status_label.setText(f"다른 평면을 클릭해서 {', '.join(missing)} 값을 지정해주세요")
        self._redraw_all()

    def _on_motion(self, event, view):
        ax = self._axis_of(view)
        if self.drag_index is None or event.inaxes != ax:
            return
        p = self.points[self.drag_index]
        if view == "xy":
            p.y, p.x = event.xdata, event.ydata
        elif view == "yz":
            p.y, p.z = event.xdata, event.ydata
        else:
            p.x, p.z = event.xdata, event.ydata
        p.valid = None
        self._redraw_all()

    def _on_release(self, event, view):
        if self.drag_index is not None:
            p = self.points[self.drag_index]
            # [6차] 점을 '집기만 하고 안 움직인' 경우엔 되돌리기 항목을 남기지
            # 않는다 - 안 그러면 클릭만 해도 스택이 쌓여 되돌리기를 여러 번
            # 눌러야 실제 변화가 취소되는 답답한 동작이 된다.
            moved = True
            if self.undo_stack:
                _lbl, snap = self.undo_stack[-1]
                if self.drag_index < len(snap):
                    ox, oy, oz = snap[self.drag_index][0], snap[self.drag_index][1], snap[self.drag_index][2]
                    moved = math.dist((p.x, p.y, p.z), (ox, oy, oz)) > 1e-9
            # [6차] 스냅을 유지하되 '범위' 표시에 맞게 확장한다.
            # 스냅을 뺄지 고민했지만 유지하는 게 맞다고 판단했다 - 추천 후보는
            # 데드존 격자 위의 이산적인 좌표(_step 간격)라, 드래그로 그 좌표에
            # 정확히 놓는 건 사실상 불가능하다. 스냅이 없으면 사용자가 "추천
            # 근처"에 놨는데 실제로는 격자 사이 빈 곳이라 또 실패하는 일이
            # 생긴다. 다만 이제 대상이 점 1개가 아니라 여러 개이므로,
            # **반경 안에서 가장 가까운 후보**로 스냅한다(확정 후보가 동률
            # 범위 안에 있으면 그쪽을 우선).
            snap_targets = []
            if self.suggestion is not None:
                sx, sy, sz = self.suggestion
                snap_targets.append((float(sx), float(sy), float(sz), True))
            snap_targets.extend(self.suggestion_region)

            # 확정 후보를 '무조건' 우선하면, 커서 바로 밑에 유력 후보가 있는데도
            # 20mm 떨어진 확정으로 끌려가 버려서 조작감이 나쁘다. 그래서 절대
            # 우선순위 대신 **거리에 보너스를 주는 방식**으로 부드럽게 선호한다 -
            # 확정은 실제보다 VERIFIED_SNAP_BONUS_MM만큼 가까운 것처럼 계산되므로,
            # 비슷한 거리면 확정이 이기고 유력 후보가 확연히 가까우면 그쪽이 이긴다.
            VERIFIED_SNAP_BONUS_MM = 8.0
            best = None   # (유효거리, x, y, z, verified)
            for cx, cy, cz, ver in snap_targets:
                dist = math.dist((p.x, p.y, p.z), (cx, cy, cz))
                if dist > SNAP_RADIUS_MM:
                    continue
                eff = dist - (VERIFIED_SNAP_BONUS_MM if ver else 0.0)
                if best is None or eff < best[0]:
                    best = (eff, cx, cy, cz, ver)
            if best is not None:
                _k, cx, cy, cz, ver = best
                p.x, p.y, p.z = cx, cy, cz
                tag = "확정 추천" if ver else "유력 후보"
                self.status_label.setText(
                    f"📌 {tag} 위치로 스냅됨: ({cx:.0f}, {cy:.0f}, {cz:.0f})"
                    + ("" if ver else " — 이 후보는 국소검사만 통과했으니 '경로 검증'을 다시 눌러 확인하세요"))
            self._validate_point(p)
            self.drag_index = None
            self.curve_waypoints = []
            self.bad_point_idx = None
            self.bad_segment = None
            self.suggestion = None
            self.suggestion_region = []
            self.execute_btn.setEnabled(False)  # 좌표가 바뀌었으니 재검증 전까지 실행 비활성화
            if not moved and self.undo_stack:
                self.undo_stack.pop()   # 실제로 안 움직였으니 되돌리기 항목 제거
            self._redraw_all()

    def _find_near_point(self, event, view, tol_px=12):
        ax = self._axis_of(view)
        click = np.array([event.x, event.y])
        for i, p in enumerate(self.points):
            if view == "xy":
                d = (p.y, p.x)
            elif view == "yz":
                d = (p.y, p.z)
            else:
                d = (p.x, p.z)
            disp = np.array(ax.transData.transform(d))
            if np.hypot(*(disp - click)) < tol_px:
                return i
        return None

    def _find_near_segment(self, event, view, tol_px=8):
        """곡선 위에서 클릭에 가장 가까운 지점을 찾아 (구간번호, 구간내 u 0~1)를 반환.
        De Casteljau 분할에 쓰이므로 전역 파라미터 t가 아니라 '구간 기준 u'가 필요하다."""
        if self._n_segments() < 1:
            return None
        ax = self._axis_of(view)
        click = np.array([event.x, event.y])
        curve = cd.sample_curve(self.points, n_samples=400)
        if curve is None:
            return None
        tt, xx, yy, zz = curve

        if view == "xy":
            proj = np.stack([yy, xx], axis=1)
        elif view == "yz":
            proj = np.stack([yy, zz], axis=1)
        else:
            proj = np.stack([xx, zz], axis=1)

        disp = ax.transData.transform(proj)
        d = np.hypot(*(disp - click).T)
        i_min = int(np.argmin(d))
        if d[i_min] >= tol_px:
            return None

        t = float(tt[i_min])
        n_seg = self._n_segments()
        seg = min(int(t // 2), n_seg - 1)
        u = (t - 2 * seg) / 2.0
        # 앵커와 거의 겹치는 지점에서 쪼개면 길이 0짜리 구간이 생기므로 무시
        if u < 0.06 or u > 0.94:
            return None
        return seg, u

    # ---------------------------------------------------------
    # 검증
    # ---------------------------------------------------------
    # ---------------------------------------------------------------
    # [6차 분할] 곡선 수학/검증 로직은 mycobot_curve_math.py로 옮겼다.
    # 아래는 기존 호출부를 하나도 안 건드려도 되게 하는 **얇은 위임**이다.
    # GUI 상태(점 목록, 측정모드에서 유도되는 간격, 허용오차)를 읽어서
    # 순수 함수에 넘겨주는 역할만 한다 - 실제 로직은 저쪽에 한 벌만 있다.
    # ---------------------------------------------------------------
    def _ik_seed_q(self):
        return cm.ik_seed_q()

    # [8차 점검] _segment_range_for_index / _build_orientations 위임 메서드는
    # 삭제했다 - §17.7 분할 이후 호출부가 전부 cm.* 직접 호출로 바뀌면서
    # 아무도 안 부르는 껍데기만 남아 있었다(AST 전수조사로 확인).

    def _resample_by_chord(self, t_fine, P_fine, base_step):
        return cm.resample_by_chord(t_fine, P_fine, base_step)

    def _curve_ok_with_substitution(self, idx, cand, q_seed):
        return cm.curve_ok_with_substitution(
            self.points, idx, cand, q_seed,
            CURVE_POS_TOL_MM, CURVE_ORIENT_TOL_DEG, self._stream_step_mm())

    def _run_curve_check(self, pts_xyz, roles, q_seed, t_lo=None, t_hi=None,
                         step_mm=None):
        """[6차 분할] 실제 로직은 mycobot_curve_math.run_curve_check에 있다.

        여기서는 GUI 쪽에서만 알 수 있는 값(허용오차 상수, 측정모드에서 유도되는
        웨이포인트 간격)을 채워 넘긴다. **간격을 여기서 채우는 게 핵심**이다 -
        validate_path와 추천 시뮬레이션이 서로 다른 간격을 쓰면 §6.3의 "추천대로
        옮겨도 재실패"가 형태를 바꿔 재발하는데, 두 경로 모두 이 메서드를 지나
        같은 self._stream_step_mm()을 쓰므로 구조적으로 어긋날 수 없다.
        """
        return cm.run_curve_check(
            pts_xyz, roles, q_seed,
            CURVE_POS_TOL_MM, CURVE_ORIENT_TOL_DEG,
            t_lo=t_lo, t_hi=t_hi,
            step_mm=step_mm if step_mm else self._stream_step_mm())

    def _suggest_validated_safe(self, x, y, z, max_candidates=400, seed=None,
                                point_idx=None, time_budget_sec=25.0):
        """실패한 점을 어디로 옮기면 되는지 추천한다.

        [예전 구현의 한계와 이번 개선]
          1. 점 하나만 검사했다 -> 곡선까지 검사한다 (_curve_ok_with_substitution).
             추천대로 옮겼는데 또 실패하던 주된 원인이었다.
          2. 후보 40개만 봤다 -> 400개까지 훑는다 (거리순).
          3. 조건을 만족하는 '첫' 후보를 즉시 반환했다 -> 여러 후보를 모아 점수로
             고른다. 첫 후보는 대개 실현 가능 영역의 '경계'에 아슬아슬하게 걸쳐 있어
             조금만 손이 떨려도 다시 실패했다. 이제 안전여유와 특이점 여유까지 보고
             '가깝고 + 여유 있고 + 특이점에서 먼' 지점을 고른다.

        시간이 걸리는 대신 정확도를 택했다(사용자 요청). 진행상황을 상태줄에 표시한다.

        [6차 계측 추가] §6.5에서 고친 시간예산은 '다음 후보 시작 전'에만
        확인한다 - 후보 1개의 `_run_curve_check()` 자체(전체 경로 재검증)가
        오래 걸리면 그 한 번은 끝까지 기다린다(주석에 이미 명시됐던 한계).
        경로가 복잡해지면(웨이포인트/구간 증가) 이 '1회 소요시간'이 커져서
        체감상 "검증이 다시 오래 걸린다"로 느껴질 수 있다. 원인을 눈으로
        보려고 각 단계 소요시간을 콘솔에 그대로 출력한다.
        """
        if not _grid_loaded or len(_safe_indices) == 0:
            return None, []
        t_start = time.time()
        print(f"[추천탐색] 시작 - 시간예산 {time_budget_sec:.0f}s, "
              f"경로 구간수 {self._n_segments()}, 점 개수 {len(self.points)}")

        target_idx = np.array([int(round((x - _x_min) / _step)),
                               int(round((y - _y_min) / _step)),
                               int(round((z - _z_min) / _step))])
        d = np.linalg.norm(_safe_indices - target_idx, axis=1)
        order = np.argsort(d)[:max_candidates]

        q_seed = seed if seed is not None else self._ik_seed_q()

        candidates_ok = []   # [(score, cand), ...] - 2차 통과 후보들
        n_curve_ok = 0
        checked = 0
        for oi in order:
            if time.time() - t_start > time_budget_sec:
                break
            nidx = _safe_indices[oi]
            cand = (_x_min + nidx[0] * _step, _y_min + nidx[1] * _step, _z_min + nidx[2] * _step)

            # --- 1차: 값싼 필터 (격자 안전성 + 그 점 자체의 도달 가능성) ---
            ok, _ = quick_prefilter(*cand)
            if not ok:
                continue
            q_c, _R = natural_pose_at(list(cand), q_seed)
            if q_c is None:
                continue
            sol_c, _pe, _oe = solve_pose_ik(
                q_c, list(cand), _R,
                pos_tol_mm=CURVE_POS_TOL_MM, orient_tol_deg=CURVE_ORIENT_TOL_DEG)
            if sol_c is None:
                continue
            cond_pt = jacobian_condition_number(analytic_jacobian_6d(sol_c)[:3, :])
            if cond_pt > CONDITION_NUMBER_MAX:
                continue

            checked += 1
            if checked % 5 == 0:
                self.status_label.setText(
                    f"💡 대체 위치 탐색 중... (후보 {checked}개 검사, 통과 {n_curve_ok}개)")
                QtWidgets.QApplication.processEvents()

            # --- 2차: 진짜 검사 - 이 후보로 바꿨을 때 곡선이 통과하는가 ---
            # (실제 검증과 완전히 같은 알고리즘: _run_curve_check)
            worst_cond = cond_pt
            if point_idx is not None and self._n_segments() >= 1:
                passed, wc = self._curve_ok_with_substitution(point_idx, cand, q_seed)
                if not passed:
                    continue
                worst_cond = max(cond_pt, wc if wc else cond_pt)
            n_curve_ok += 1

            # --- 점수: 가까울수록 / 안전여유 클수록 / 특이점에서 멀수록 좋다 ---
            dist_mm = math.dist((x, y, z), cand)
            margin = cd.margin_at(*cand)
            cond_ratio = worst_cond / max(CONDITION_NUMBER_MAX, 1e-6)   # 0~1, 작을수록 여유
            score = (dist_mm
                     - 2.0 * min(margin, 40.0)     # 여유 1mm당 2mm어치 가산점 (40mm에서 포화)
                     + 60.0 * cond_ratio)          # 특이점에 가까우면 강하게 감점
            candidates_ok.append((score, cand))

            # 충분히 후보를 모았으면 조기 종료 (거리순이라 더 가면 멀어지기만 한다)
            # [버그 수정] 예전엔 8개까지 모았는데, 아래 최종 검증 단계가 각각
            # '전체 경로'를 통째로 다시 훑는(=validate_path 한 번과 맞먹는 비용)
            # 무거운 작업이라 8번 반복하면 검증이 사실상 안 끝나는 것처럼 느껴질
            # 수 있었다("검증이 끝나질 않는다" 버그). 4개로 줄이고, 아래서
            # 시간예산도 강제한다.
            # [6차] 이제 '범위 표시'를 위해 더 많이(REGION_MAX) 모은다 - 다만
            # 비싼 전체경로 재검증은 여전히 상위 FULLCHECK_MAX개만 돌린다.
            # 국소검사(_curve_ok_with_substitution)는 6차 시드 통일 이후
            # 실제 검증과 같은 기준이라, 여기 모인 후보들은 '유력 영역'으로
            # 보여줘도 오해를 주지 않는다.
            if len(candidates_ok) >= REGION_MAX:
                break

        if not candidates_ok:
            print(f"[추천탐색] 1차 후보 탐색 종료 - {checked}개 검사, 통과 0개, "
                  f"소요 {time.time()-t_start:.1f}s (여기서 실패 - 아래 최종재검증 단계는 시작도 안 됨)")
            return None, []
        print(f"[추천탐색] 1차 후보 탐색 종료 - {checked}개 검사, 통과 {len(candidates_ok)}개, "
              f"소요 {time.time()-t_start:.1f}s -> 이제 후보별 전체경로 재검증 시작")

        # ┌─ 최종 관문: 점수 좋은 순으로 '전체 경로'를 통째로 재검증 ──────────────┐
        # │ 위 2차 검사는 속도를 위해 '영향받는 구간'만 봤다. 그런데 실제 검증은   │
        # │ 경로 전체를 처음부터 끝까지, 웜스타트 시드를 이어가며 훑는다. 앞 구간  │
        # │ 에서 어떤 관절해로 수렴했느냐가 뒤 구간의 성패를 바꾸므로, 부분 검사만 │
        # │ 통과하고 전체에서 실패하는 경우가 남는다.                              │
        # │ 그래서 최종 후보만큼은 실제 검증과 **완전히 동일한 전체 경로 검사**를  │
        # │ 통과시킨다. 여기를 통과한 좌표는 정의상 validate_path도 통과한다.      │
        # │                                                                          │
        # │ [버그 수정 - 시간예산 누락] 이 단계가 원래 시간제한이 전혀 없었다.      │
        # │ 후보 하나당 전체 경로 IK를 처음부터 다 푸는 게 validate_path() 한 번과  │
        # │ 맞먹는 비용인데, 그걸 후보 최대 8개에 대해 무제한으로 반복했다 -        │
        # │ 웨이포인트가 많은 경로에서는 "검증이 끝나질 않는다"로 느껴질 만큼       │
        # │ 오래 걸렸을 것이다. 이제 위 탐색 단계와 **같은 time_budget_sec을        │
        # │ 합산으로** 지킨다 - 탐색에서 시간을 많이 썼으면 여기서 쓸 시간이        │
        # │ 그만큼 줄어든다(총 상한은 항상 time_budget_sec).                        │
        # └────────────────────────────────────────────────────────────────────────┘
        candidates_ok.sort(key=lambda t: t[0])
        base_seed = self._ik_seed_q()
        roles = [p.is_anchor for p in self.points]
        # [6차] 범위 표시용: 국소검사를 통과한 후보 전체를 좌표만 뽑아 둔다.
        # 아래에서 전체검증까지 통과한 1개는 verified=True로 승격한다.
        region = [(float(c[0]), float(c[1]), float(c[2]), False)
                  for _s, c in candidates_ok]
        verified = None
        for rank, (score, cand) in enumerate(candidates_ok[:FULLCHECK_MAX]):
            if time.time() - t_start > time_budget_sec:
                print(f"[추천탐색] 시간예산 초과 - {rank}/{min(len(candidates_ok), FULLCHECK_MAX)}개만 재검증하고 중단 "
                      f"(총 경과 {time.time()-t_start:.1f}s)")
                self.status_label.setText(
                    f"💡 시간예산({time_budget_sec:.0f}s) 초과로 전체검증을 "
                    f"{rank}/{min(len(candidates_ok), FULLCHECK_MAX)}개만 확인하고 중단했습니다.")
                QtWidgets.QApplication.processEvents()
                break
            self.status_label.setText(
                f"💡 최종 후보 전체 경로 재검증 중... ({rank+1}/{min(len(candidates_ok), FULLCHECK_MAX)})")
            QtWidgets.QApplication.processEvents()
            pts_xyz = [[p.x, p.y, p.z] for p in self.points]
            if point_idx is not None:
                pts_xyz[point_idx] = [float(cand[0]), float(cand[1]), float(cand[2])]
            t_cand = time.time()
            passed, _wc, _why = self._run_curve_check(pts_xyz, roles, base_seed)
            cand_sec = time.time() - t_cand
            print(f"[추천탐색] 후보 {rank+1}/{min(len(candidates_ok), FULLCHECK_MAX)} 전체재검증: "
                  f"{'통과' if passed else '실패'}, 이 후보 1개에 {cand_sec:.1f}s 소요 "
                  f"(누적 {time.time()-t_start:.1f}s)")
            if cand_sec > time_budget_sec / 4:
                print(f"  ⚠️ 후보 1개 재검증이 {cand_sec:.1f}s로 유독 오래 걸렸습니다 - "
                      "경로 구간수/웨이포인트가 늘어난 게 원인일 가능성이 높습니다 "
                      "(전체 경로를 처음부터 다시 IK 푸는 비용이라 구간수에 비례).")
            if passed:
                verified = (float(cand[0]), float(cand[1]), float(cand[2]))
                region[rank] = (verified[0], verified[1], verified[2], True)
                print(f"[추천탐색] 완료 - 총 소요 {time.time()-t_start:.1f}s, "
                      f"범위 후보 {len(region)}개 (확정 1개)")
                return verified, region

        # 전체 검증까지 통과하는 후보가 없었다 - 확정 추천은 못 주지만,
        # [6차] 국소검사를 통과한 '유력 영역'은 그대로 돌려준다. 사용자가
        # 그 안에서 직접 골라 옮겨볼 수 있게 하는 게 아무것도 안 주는 것보다 낫다.
        print(f"[추천탐색] 전체검증 통과 후보 없음 - 총 소요 {time.time()-t_start:.1f}s, "
              f"유력 영역 {len(region)}개만 반환(확정 없음)")
        return None, region

    def _validate_point(self, p):
        ok, _ = quick_prefilter(p.x, p.y, p.z)
        p.valid = ok

    def validate_path(self):
        if len(self.points) < 2:
            self.status_label.setText("점을 2개 이상 찍어야 경로를 검증할 수 있습니다.")
            return

        self.execute_btn.setEnabled(False)
        self.curve_waypoints = []
        self.bad_point_idx = None
        self.bad_segment = None
        self.suggestion = None
        self.suggestion_region = []
        for p in self.points:
            p.valid = None
        self.status_label.setText("검증 중... (곡선 전체를 촘촘히 확인하는 중이라 몇 초 걸릴 수 있어요)")
        QtWidgets.QApplication.processEvents()

        # 1) 각 점의 자연스러운 자세(방향) 확보
        # [경로 방식 변경] 짝수 인덱스(0,2,4...) = 앵커: 로봇이 실제로 방문하므로
        # 반드시 도달 가능해야 한다(기존과 동일하게 강하게 검증).
        # 홀수 인덱스(1,3,5...) = 베지어 제어점: 곡선이 그 점을 지나지 않으므로
        # IK 도달 여부를 강제하지 않는다 - 자세 보간의 가이드로만 쓰인다.
        # (강제했다면 "제어점 자체는 안 닿아도 되는데 검증만 실패하는" 불필요한
        # 실패가 잦아졌을 것 - 추천 위치로 옮겨도 계속 실패하던 문제의 연장선)
        n = len(self.points)
        q_seed = self._ik_seed_q()   # [고정 시드] 로봇 현재자세에 의존하면 검증이 재현되지 않는다
        orientations = []
        for i, p in enumerate(self.points):
            is_anchor = p.is_anchor   # [명시적 role] 패리티 추측 대신 점 자신이 들고 있음
            q, Rm = natural_pose_at(p.coord(), q_seed)
            if q is None:
                if is_anchor:
                    p.valid = False
                    self.bad_point_idx = i
                    self.suggestion, self.suggestion_region = self._suggest_validated_safe(
                        p.x, p.y, p.z, seed=q_seed, point_idx=i)
                    msg = (f"❌ {i+1}번 점 ({p.x:.0f},{p.y:.0f},{p.z:.0f})에 도달할 수 있는 자세가 없습니다. "
                           f"이 점을 옮겨주세요.")
                    msg += self._suggestion_msg()
                    self.status_label.setText(msg)
                    self._redraw_all()
                    return
                # 제어점: 검증 대상이 아님 - 회색으로 표시하고 직전 자세를 그대로 이어붙여 진행
                p.valid = None
                orientations.append(orientations[-1] if orientations else np.eye(3))
                continue
            p.valid = True
            orientations.append(Rm)
            q_seed = q

        # 2) 곡선을 촘촘하게 샘플링한 뒤 Douglas-Peucker로 필요한 점만 남김
        #    -> 굽은 구간은 촘촘하게, 곧은 구간은 듬성듬성
        n = len(self.points)
        ts = np.arange(n, dtype=float)
        xs = np.array([p.x for p in self.points])
        ys = np.array([p.y for p in self.points])
        zs = np.array([p.z for p in self.points])
        mode = self._current_mode()
        # 스트리밍 모드는 코너 잘라먹기를 줄이기 위해 균일하고 촘촘한 간격을 그대로 사용,
        # 큐 모드는 곡률 기반(RDP)으로 꼭 필요한 점만 남겨 구간을 길게 확보한다.
        base_step = self._stream_step_mm() if mode == "stream" else DENSE_SAMPLE_STEP_MM

        # [경로 방식 변경] 이제 곡선은 '모든 점을 지나는 보간'이 아니라 앵커-제어점을
        # 번갈아 잇는 2차 베지어 체인이다 (_bezier_chain_eval 참고).
        def _eval(tq):
            return _bezier_chain_eval(xs, ys, zs, tq)

        # ┌─ [재진단] 호길이 균일화는 애초에 틀린 기준이었다 ────────────────────┐
        # │ 처음엔 "사전 샘플링이 성겨서 노치가 생긴다"고 보고 표본을 곡률에      │
        # │ 맞춰 늘렸는데, 실측(합성 급반전 곡선)으로 재현해보니 표본을          │
        # │ 500개→20000개로 늘려도 노치 크기가 똑같았다. 원인이 표본 부족이      │
        # │ 아니라는 뜻이었다.                                                    │
        # │                                                                        │
        # │ 진짜 원인: 로봇은 웨이포인트 사이를 '직선'으로 움직인다. 그런데       │
        # │ 호길이 균일화는 곡선을 따라간 거리(arc length)를 균일하게 만들 뿐,    │
        # │ 급하게 꺾이는 구간에서는 곡선이 안쪽으로 휘어 들어가므로 '직선거리    │
        # │ (현, chord)'가 짧아진다 - 호길이는 같아도 현은 짧아지는 게 당연한     │
        # │ 기하학이다. 그 현 길이가 곧 그 사이클의 TCP 이동거리이므로, 그게      │
        # │ 짧아지면 속도가 그대로 떨어진다 - 이게 노치의 진짜 정체였다.          │
        # │                                                                        │
        # │ 그래서 균일하게 맞춰야 할 건 호길이가 아니라 **현**이다. 곡선을       │
        # │ 따라 걸으면서, 직전 웨이포인트로부터의 직선거리가 base_step에         │
        # │ 도달하는 지점마다 새 웨이포인트를 찍는다(_resample_by_chord).         │
        # │ 급반전 지점에서는 자동으로 더 촘촘하게 찍힌다(직선거리가 잘 안        │
        # │ 벌어지니 여러 스텝이 필요해짐) - 표본을 몇 개 쓸지 미리 정하지        │
        # │ 않고 결과(현 길이)로 직접 제어하는 방식이라 원리적으로 노치가 생길    │
        # │ 수 없다.                                                              │
        # │                                                                        │
        # │ 실측: 같은 급반전 곡선에서 최대편차 61.5% -> 0.0% (합성 테스트).      │
        # └────────────────────────────────────────────────────────────────────────┘
        n_seg = self._n_segments()
        t_fine = np.linspace(0.0, n - 1.0, max(4000, 1000 * n_seg))
        fx, fy, fz = _eval(t_fine)
        P_fine = np.stack([fx, fy, fz], axis=1)

        tt_dense = self._resample_by_chord(t_fine, P_fine, base_step)
        xx_dense, yy_dense, zz_dense = _eval(tt_dense)

        dense_pts = np.stack([xx_dense, yy_dense, zz_dense], axis=1)
        if mode == "stream":
            tt = tt_dense
            xx, yy, zz = xx_dense, yy_dense, zz_dense
        else:
            keep_idx = cd.rdp_indices(dense_pts, CURVE_FIDELITY_MM)
            tt = tt_dense[keep_idx]
            xx, yy, zz = xx_dense[keep_idx], yy_dense[keep_idx], zz_dense[keep_idx]

        # ┌─ [패치 5] 자세 보간을 C1으로 ─────────────────────────────────────┐
        # │ 이전에는 웨이포인트마다 '자기가 속한 구간의 양 끝'만 Slerp 했다.   │
        # │ 위치는 CubicSpline(C2)인데 자세는 C0라, 제어점을 지날 때마다        │
        # │ 각속도가 계단처럼 튀었다.                                          │
        # │  -> Figure 1의 beta(ry) t≈2.0/4.1 꺾임, gamma(rz) t≈4.1 뾰족한     │
        # │     봉우리, 그리고 Figure 2 J2/J3의 잔물결이 전부 이것.            │
        # │ RotationSpline은 제어점 '전체'를 한 번에 보간해 각속도가 연속이다. │
        # │ (제어점이 2개뿐이면 스플라인이 의미 없으므로 Slerp로 폴백)         │
        # └───────────────────────────────────────────────────────────────────┘
        R_ctrl = Rotation.from_matrix(np.stack(orientations))
        if _HAS_ROT_SPLINE and n >= 3:
            _rot_interp = RotationSpline(ts, R_ctrl)
            def _R_at(t_):
                return _rot_interp(t_).as_matrix()
        else:
            _slerp_all = Slerp(ts, R_ctrl)
            def _R_at(t_):
                return _slerp_all(np.clip(t_, ts[0], ts[-1])).as_matrix()

        R_all = _R_at(tt)          # (len(tt), 3, 3) - 루프 밖에서 한 번에 계산

        # 3) 각 샘플마다 6자유도 IK + 안전검사 (이전 해에서 이어서, 연속성 확보)
        q_seed = self._ik_seed_q()   # [고정 시드] 로봇 현재자세에 의존하면 검증이 재현되지 않는다
        waypoints = []
        ik_errors = []      # 각 웨이포인트에서 IK가 실제로 도달한 위치오차(mm)
        # [참고: linear_x.py 계열 외부 스크립트] 웨이포인트 사이 최대 관절각 변화량을
        # 실행 전에 진단으로 보여준다. 지금 웨이포인트 간격은 1.26mm로 아주 촘촘해서
        # 정상이라면 스텝당 변화가 1도도 안 된다 - 그런데 IK가 다중해 문제라 시드가
        # 조금만 어긋나도 다른 관절해(팔꿈치 위/아래 등)로 튈 수 있다. 그 경우 스텝당
        # 변화량이 갑자기 몇 도씩 튀므로, 이 값이 크면 "부드러워 보이는 곡선인데 실제로는
        # 관절이 점프하는 구간이 있다"는 신호다.
        max_step_delta_deg = 0.0
        prev_deg = None

        for k in range(len(tt)):
            seg_i = min(int(np.floor(tt[k])), n - 2)
            frac = tt[k] - seg_i
            R_k = R_all[k]
            target = np.array([xx[k], yy[k], zz[k]])

            fail_reason = None
            sol, pos_err, orient_err = solve_pose_ik(
                q_seed, target, R_k, pos_tol_mm=CURVE_POS_TOL_MM, orient_tol_deg=CURVE_ORIENT_TOL_DEG
            )
            if sol is None:
                fail_reason = f"그 자세로는 팔이 닿지 않음 (위치오차 {pos_err:.0f}mm / 자세오차 {orient_err:.0f}도)"
            else:
                ok, bad = within_joint_limits(sol)
                if not ok:
                    fail_reason = f"{bad+1}번 관절이 회전 한계를 넘어야 함"
                elif jacobian_condition_number(analytic_jacobian_6d(sol)[:3, :]) > CONDITION_NUMBER_MAX:
                    fail_reason = "팔이 거의 다 펴진 특이점 자세라 제어가 불안정해짐"
                elif check_self_collision(sol):
                    fail_reason = "로봇이 자기 몸체와 충돌함"
                elif target[2] < DESK_SAFETY_MARGIN_MM:
                    fail_reason = f"책상과 충돌 위험 (z={target[2]:.0f}mm)"

            if fail_reason:
                self._mark_failure(seg_i, frac, target, fail_reason, seed=q_seed)
                self._redraw_all()
                return

            # 실제 수렴 오차 기록 (명령값이 계획과 얼마나 어긋나는지의 직접 원인)
            fk_chk = chain.forward_kinematics(sol)
            ik_errors.append(float(np.linalg.norm(fk_chk[:3, 3] * 1000.0 - target)))

            deg_now = [math.degrees(sol[i]) for i in active_indices]
            if prev_deg is not None:
                d = max(abs(a - b) for a, b in zip(deg_now, prev_deg))
                max_step_delta_deg = max(max_step_delta_deg, d)
            prev_deg = deg_now

            waypoints.append((sol, target.tolist()))
            q_seed = sol

        self.curve_waypoints = waypoints
        self.execute_btn.setEnabled(True)

        # [패치 4 검증용] 실제 목표점들의 간격이 정말 균일해졌는지 자가진단.
        #   호길이 재매개화가 제대로 먹었다면 편차가 1% 미만이어야 한다.
        tgt = np.array([t for (_, t) in waypoints], dtype=float)
        gaps = np.linalg.norm(np.diff(tgt, axis=0), axis=1) if len(tgt) >= 2 else np.array([0.0])

        jump_note = f" · 스텝당 최대 관절변화 {max_step_delta_deg:.2f}°"
        jump_warn = ""
        if max_step_delta_deg > STEP_JUMP_WARN_DEG:
            jump_warn = (f"  ⚠️ 웨이포인트 사이 관절각이 {max_step_delta_deg:.1f}°나 튀는 구간이 있습니다"
                         f" (IK가 다른 해로 넘어갔을 수 있음 - 그 부근 점을 조금 옮겨보세요)")

        if mode == "stream":
            step_now = self._stream_step_mm()
            exp_speed = step_now / max(1e-6, MEASURE_CYCLE_SEC.get(
                self.measure_combo.currentData(), MIN_DISPATCH_PERIOD_SEC))
            self.status_label.setText(
                f"✅ 검증 통과! [IK오차 평균 {np.mean(ik_errors):.3f}mm / 최대 {np.max(ik_errors):.3f}mm] "
                f"스트리밍 모드 · {len(waypoints)}개 웨이포인트"
                f"(간격 {gaps.mean():.2f}±{gaps.std():.2f}mm, 목표 {step_now:.2f}mm)"
                f"{jump_note} · 예상 TCP 약 {exp_speed:.0f}mm/s. 실행 가능" + jump_warn
            )
        else:
            self.status_label.setText(
                f"✅ 검증 통과! [IK오차 평균 {np.mean(ik_errors):.2f}mm / 최대 {np.max(ik_errors):.2f}mm] "
                f"큐 모드 · {len(waypoints)}개 웨이포인트(곡률 기반 자동 간격)"
                f"{jump_note}. 실행 가능" + jump_warn
            )
        self._redraw_all()

    def _mark_failure(self, seg_i, frac, target, reason, seed=None):
        """실패 위치를 가장 가까운 점/구간에 귀속시키고, 화면 표시용 상태를 설정.
        seed: 실패한 그 지점에서 실제로 쓰이고 있던 웜스타트 시드 (있으면 추천에도 그대로 사용)."""
        # 실패 지점이 어느 제어점에 더 가까운지로 '문제의 점'을 결정
        near_idx = seg_i if frac < 0.5 else seg_i + 1
        near_idx = max(0, min(len(self.points) - 1, near_idx))
        self.bad_point_idx = near_idx
        self.bad_segment = seg_i
        self.points[near_idx].valid = False

        bad_p = self.points[near_idx]
        self.suggestion, self.suggestion_region = self._suggest_validated_safe(
            bad_p.x, bad_p.y, bad_p.z, seed=seed, point_idx=near_idx)

        msg = (f"❌ {near_idx+1}번 점 부근에서 실패: {reason}\n"
               f"   문제 위치 ({target[0]:.0f},{target[1]:.0f},{target[2]:.0f}) — "
               f"{seg_i+1}→{seg_i+2}번 점 사이 구간입니다. "
               f"{near_idx+1}번 점을 옮기거나 이 구간에 점을 추가해 경로를 완만하게 바꿔보세요.")
        msg += self._suggestion_msg()
        self.status_label.setText(msg)

    def _suggestion_msg(self):
        """[6차 신규] 추천 결과(확정 1개 + 유력 영역)를 상태줄 문구로 만든다.

        확정(전체검증 통과)이 있으면 그 좌표를 알려주고, 확정이 없어도 국소검사를
        통과한 '유력 영역'이 있으면 그것만이라도 안내한다 - 예전엔 확정이 없으면
        아무 정보도 안 줘서 사용자가 어디로 옮겨야 할지 전혀 알 수 없었다.
        """
        n_region = len(self.suggestion_region)
        if self.suggestion:
            s_ = self.suggestion
            extra = (f" (주변 유력 영역 {n_region}곳도 옅은 주황으로 표시)"
                     if n_region > 1 else "")
            return (f"  💡 추천 위치: ({s_[0]:.0f}, {s_[1]:.0f}, {s_[2]:.0f}) "
                    f"— 진한 주황 원. 근처({SNAP_RADIUS_MM}mm)로 끌어놓으면 자동 스냅됩니다{extra}")
        if n_region:
            return (f"  💡 전체 경로까지 통과하는 '확정' 위치는 못 찾았지만, 국소적으로는 "
                    f"통과하는 유력 후보 {n_region}곳을 옅은 주황으로 표시했습니다 - "
                    f"그 근처로 옮긴 뒤 다시 검증해보세요 (근처로 끌면 가장 가까운 후보로 스냅됩니다). "
                    f"전부 실패하면 이 점이 아니라 경로의 다른 구간이 원인일 수 있습니다.")
        return "  (근처에서 실제로 도달 가능한 대체 위치를 찾지 못했습니다. 좀 더 멀리 옮겨보세요)"

    # ---------------------------------------------------------
    # 실행
    # ---------------------------------------------------------
    def execute_path(self):
        if not getattr(self, "curve_waypoints", None):
            self.status_label.setText("먼저 '경로 검증'을 통과해야 실행할 수 있습니다.")
            return

        if not self._ensure_connected():
            return

        # 실행 방식에 맞춰 펌웨어 모드 설정
        #   0 = 큐(대기열): 명령을 순서대로 완결 실행 -> 웨이포인트마다 감속
        #   1 = refresh:   최신 명령만 즉시 실행 -> 감속하지 않고 계속 쫓아감
        self.mc.set_fresh_mode(1 if self._current_mode() == "stream" else 0)
        time.sleep(0.2)

        # [참고: linear_x.py 계열 외부 스크립트 - 항상 홈을 거쳐 시작 자세로 이동]
        # 백래시(기어 유격)는 '어느 방향에서 왔느냐'에 따라 정지오차가 달라진다는 게
        # 이미 확인됐다. 이전 실행이 로봇을 어디에 세워놨든, 매번 같은 정렬자세를
        # 거쳐서 시작점에 접근하면 적어도 시작 시점의 접근 방향은 항상 같아지므로
        # 실행 간 정지오차 재현성이 좋아질 것으로 기대한다. 시간이 조금(1~2초) 더
        # 걸리는 대가이므로, 필요 없으면 아래 상수를 False로.
        if ROUTE_VIA_ALIGN_BEFORE_START:
            self.status_label.setText("🧭 정렬자세를 거쳐 시작점으로 이동 중... (접근 방향 재현성)")
            QtWidgets.QApplication.processEvents()
            self.mc.send_angles(ALIGN_ANGLES, SPEED)
            time.sleep(SETTLE_DELAY_SEC)
            t_align_wait = time.time()
            while self.mc.is_moving():
                if time.time() - t_align_wait > MOVE_TIMEOUT_SEC:
                    break
                time.sleep(0.1)

        # --- 1단계: 경로의 시작점까지 먼저 이동 (이 구간은 측정하지 않음) ---
        self.status_label.setText("➡️ 경로 시작점으로 이동 중... (이 구간은 측정에서 제외됩니다)")
        QtWidgets.QApplication.processEvents()

        first_sol = self.curve_waypoints[0][0]
        first_angles = [math.degrees(first_sol[i]) for i in active_indices]
        # [정지오차 보정] 여기서 보정된 값으로 한 번 정해두고, 아래 캘리브레이션의
        # '더미 재전송'도 반드시 같은 값을 써야 한다 - 안 그러면 로봇이 보정된
        # 위치에 서 있는데 캘리브레이션이 원래(미보정) 값을 다시 보내 미세하게
        # 움직이게 되고, 그러면 "정지 상태에서 통신비용만 잰다"는 캘리브레이션의
        # 전제가 깨진다.
        first_angles_to_send = (joint_offset_correction(first_angles)
                                if APPLY_JOINT_OFFSET_CORRECTION else first_angles)
        self.mc.send_angles(first_angles_to_send, SPEED)
        time.sleep(SETTLE_DELAY_SEC)
        t_wait = time.time()
        while self.mc.is_moving():
            if time.time() - t_wait > MOVE_TIMEOUT_SEC:
                break
            time.sleep(0.1)
        # [원복] 시작 전 정지시간을 1.5초로 늘려봤지만(이전 실행 결과), t=0 정지오차가
        #   거의 그대로였다 - 예상대로 진동/settling 문제가 아니라 계통오차였다는
        #   뜻이므로 되돌린다. 효과 없는 대기시간을 늘려봐야 실행시간만 늘어난다.
        time.sleep(0.4)   # 완전히 정지해 안정된 뒤부터 측정 시작

        # --- 1.5단계: 자동 주기 보정 (로봇은 이미 여기 서 있으므로 움직이지 않음) ---
        mode = self._current_mode()
        measure_mode = self.measure_combo.currentData()
        calib_period = calib_floor = None
        _calib_std = None   # 반환 3번째 값(표준편차)은 현재 표시에 안 쓴다 - 자리만 받음
        if mode == "stream":
            self.status_label.setText("🔧 통신 주기 자동 보정 중... (로봇은 움직이지 않습니다)")
            QtWidgets.QApplication.processEvents()
            calib_period, calib_floor, _calib_std = se.calibrate_dispatch_period(
                self.mc, measure_mode, first_angles_to_send)
            # [9차, §25 - 실기 검증 후 채택] 자동 보정값(보통 32~33ms) 대신
            # 스파이크까지 덮는 목표주기로 실행한다. 자동보정을 대체하는 게
            # 아니라 그 위에 명시적으로 얹는 것 - 로그/상태줄에 남겨서 나중에
            # "이날은 왜 자동보정값이랑 다르지" 헷갈리지 않게 한다.
            if ABSORB_SPIKES_TEST:
                print(f"🔧 [스파이크 흡수] calib_period {calib_period*1000:.1f}ms → "
                      f"{ABSORB_TARGET_PERIOD_SEC*1000:.1f}ms 적용")
                calib_period = ABSORB_TARGET_PERIOD_SEC

        # --- 2단계: 여기서부터가 실제 경로 추종 (측정 대상) ---
        self.status_label.setText(f"▶️ 곡선 경로 실행 중... ({len(self.curve_waypoints)}개 웨이포인트 스트리밍)")
        QtWidgets.QApplication.processEvents()

        times, coord_samples, angle_samples = [], [], []
        cmd_times, cmd_angles, cmd_targets = [], [], []   # 명령값 기록 (통신 불필요)
        stream_meas_times, stream_meas_angles = [], []    # [지연 FK] 스트리밍 루프 전용 원본 버퍼
        t_start = time.time()

        if mode == "stream":
            # [6차 분할] 실시간 데드라인이 있는 전송 루프는 mycobot_stream_exec로
            # 옮겼다(적응형 재보정 포함). 여기서는 결과 통계만 받아 보관한다.
            # _sample_measurement_raw는 self.mc를 쓰는 GUI쪽 메서드라 콜백으로 넘긴다.
            #
            # [10차, §45] 파이프라인 읽기 - 캘리브레이션(위 1.5단계)이 이미
            # 끝난 뒤라 self.mc를 단독으로 쓰는 구간이 지났으므로, 지금부터
            # 배경 리더를 돌려도 안전하다(락으로 fast_send_angles와 쓰기만
            # 조율됨 - AsyncAngleReader 클래스 주석 참고). try/finally로
            # 예외가 나도 반드시 stop()해서 스레드가 안 남게 한다.
            self._async_reader = None
            if se.APPLY_ASYNC_MEASUREMENT_PIPELINE and measure_mode == "full":
                self._async_reader = AsyncAngleReader(self.mc)
                self._async_reader.start()
            try:
                stats = se.run_streaming_dispatch(
                    self.mc, self.curve_waypoints, calib_period,
                    APPLY_JOINT_OFFSET_CORRECTION, measure_mode, t_start,
                    cmd_times, cmd_angles, cmd_targets,
                    stream_meas_times, stream_meas_angles,
                    self._sample_measurement_raw)
            finally:
                if self._async_reader is not None:
                    self._async_reader.stop()
                    self._async_reader = None
            self._cycle_mean = stats["cycle_mean"]
            self._cycle_std = stats["cycle_std"]
            self._late_ratio = stats["late_ratio"]
            self._target_period = stats["target_period"]
            self._adapt_events = stats["adapt_events"]
            self._calib_period = stats["calib_period"]
            self._cycle_times = stats["cycle_times"]
            self._gc_events = stats["gc_events"]
            self._cycle_end_abs = stats["cycle_end_abs"]
            self._p_fb_clamp_count = stats["p_fb_clamp_count"]
            self._p_fb_deadband_hits = stats["p_fb_deadband_hits"]
            self._error_model_joints = stats["error_model_joints"]
            self._error_model_ff_mean_deg = stats["error_model_ff_mean_deg"]
            self._p_gain_adapt_events = stats["p_gain_adapt_events"]
            self._final_p_gain = stats["final_p_gain"]
        else:
            # 큐 모드: 명령을 순서대로 쌓아 실행 (웨이포인트마다 감속 -> 톱니)
            for idx, (sol, target) in enumerate(self.curve_waypoints):
                angles_deg = [math.degrees(sol[i]) for i in active_indices]
                angles_to_send = (joint_offset_correction(angles_deg)
                                  if APPLY_JOINT_OFFSET_CORRECTION else angles_deg)
                self.mc.send_angles(angles_to_send, CURVE_SPEED)
                cmd_times.append(time.time() - t_start)
                cmd_angles.append(angles_deg)
                cmd_targets.append(list(target))
                delay = SETTLE_DELAY_SEC if idx == 0 else DISPATCH_DELAY_SEC
                time.sleep(delay)
                # [패치 6] 예전엔 여기만 get_coords()(펌웨어 기구학)를 썼다.
                #   스트리밍 모드는 FK(get_angles)를 썼으므로 두 모드의 그래프가
                #   서로 다른 자로 잰 값이었다 -> 모드 비교가 애초에 성립하지 않았다.
                self._sample_measurement(times, coord_samples, angle_samples, t_start)

        # [밀림 측정] 명령을 다 뿌린 시각과, 로봇이 실제로 다 따라잡은 시각의 차이.
        #   이 값이 크면 = 명령을 로봇이 소화할 수 있는 속도보다 빨리 쏜 것 (주기를 늘려야 함)
        dispatch_done_t = time.time() - t_start

        # 스트리밍 모드는 마지막 목표에 도달하기 전에 루프가 끝날 수 있으므로,
        # 최종 목표를 한 번 더 보내 확실히 도착하도록 보장
        if mode == "stream":
            last_sol = self.curve_waypoints[-1][0]
            self.mc.send_angles([math.degrees(last_sol[i]) for i in active_indices], CURVE_SPEED)
            time.sleep(SETTLE_DELAY_SEC)

        # 마지막 웨이포인트까지 다 도달할 때까지 대기 (계속 샘플링)
        t0 = time.time()
        while self.mc.is_moving():
            if time.time() - t0 > MOVE_TIMEOUT_SEC:
                break
            # [패치 6] 같은 배열에 get_coords()를 섞어 넣던 자리.
            #   출처가 바뀌면 그래프 꼬리 부분만 수 mm 어긋난다. FK로 통일.
            self._sample_measurement(times, coord_samples, angle_samples, t_start)
            time.sleep(TRAJ_SAMPLE_SEC)

        total_t = time.time() - t_start
        lag = total_t - dispatch_done_t

        # [지연 FK] 스트리밍 중엔 get_angles()만 쌓아뒀다(stream_meas_*).
        # 여기서 한꺼번에 순기구학을 돌려 coord_samples를 채운다.
        #
        # [중요] 반드시 dispatch_done_t / total_t / lag 를 다 잰 "다음"에 해야 한다.
        # 이 계산 자체가 (샘플 수 × FK 1회 비용)만큼 실제 시간이 걸리는데, 만약
        # lag를 재기 "전"에 여기서 시간을 써버리면 그 몇 초가 고스란히 밀림으로
        # 잡혀서 진단이 오염된다. 로봇 쪽 명령 전송과 물리적 이동은 이미 다
        # 끝난 뒤이므로, 이 계산이 몇 초 걸려도 실제 실행 속도나 밀림 측정에는
        # 영향이 없다. 값 자체는 commanded 쪽과 동일한 URDF FK라 정확도 손실도 없다.
        for a, tm in zip(stream_meas_angles, stream_meas_times):
            q_meas = [0.0] * len(chain.links)
            for i, idx in enumerate(active_indices):
                q_meas[idx] = math.radians(a[i])
            fk_m = chain.forward_kinematics(q_meas)
            coord_samples.append((fk_m[:3, 3] * 1000.0).tolist() +
                                 matrix_to_rxryrz(fk_m[:3, :3]).tolist())
            angle_samples.append(a)
            times.append(tm)
        # 스트리밍 구간 샘플을 먼저 넣었으니 시간순 정렬을 보장해야 한다
        # (마무리 대기 루프 쪽 샘플은 이 시점 이후에 이미 뒤에 붙어 있었다 ->
        #  전체를 시간 기준으로 한 번 정렬해 순서를 확정한다).
        if times:
            order = np.argsort(times)
            times = [times[i] for i in order]
            coord_samples = [coord_samples[i] for i in order]
            angle_samples = [angle_samples[i] for i in order]

        # [9차 세션, §29.16] 상태줄 문자열 조립은 mycobot_status_msg.py로 분리했다
        # (순수 함수, 로봇/Qt 무관 - 최근 버그 2건이 전부 이 블록에서 났는데
        #  정작 로봇을 완주해야만 발견됐기 때문. 이제 오프라인 검증이 된다).
        # 여기서는 값만 모아서 넘긴다.
        pf_info = None
        if APPLY_P_FEEDBACK_CONTROL and mode == "stream":
            # P 피드백은 run_streaming_dispatch(stream 모드) 안에서만 실제로
            # 동작한다 - queue 모드면 None을 넘겨 표시 자체를 생략한다.
            pf_info = {
                "enabled": True,
                "gain": P_FEEDBACK_GAIN,
                "auto_gain": P_FEEDBACK_AUTO_GAIN,
                "final_gain": getattr(self, "_final_p_gain", None),
                "gain_adapt_events": getattr(self, "_p_gain_adapt_events", []),
                "softstart_cycles": P_GAIN_SOFTSTART_CYCLES if P_GAIN_SOFTSTART else 0,
                "deadband_deg": P_FEEDBACK_DEADBAND_DEG,
                "deadband_hits": getattr(self, "_p_fb_deadband_hits", 0),
                "smooth_alpha": P_FEEDBACK_SMOOTH_ALPHA,
                "kd": P_FEEDBACK_KD,
                "clamp_count": getattr(self, "_p_fb_clamp_count", 0),
                # [9차, §30.7]
                "error_model_joints": getattr(self, "_error_model_joints", []),
                "error_model_ff_mean_deg": getattr(self, "_error_model_ff_mean_deg", None),
            }

        stream_info = None
        if mode == "stream":
            stream_info = {
                "dispatch_done_t": dispatch_done_t,
                "total_t": total_t,
                "lag": lag,
                "cycle_mean": getattr(self, "_cycle_mean", 0.0),
                "cycle_std": getattr(self, "_cycle_std", 0.0),
                "target_period": getattr(self, "_target_period", 0.0),
                "late_ratio": getattr(self, "_late_ratio", 0.0),
                "calib_floor": calib_floor,
                "calib_period": getattr(self, "_calib_period",
                                        getattr(self, "_target_period", 0.0)),
                "adapt_events": getattr(self, "_adapt_events", []),
                "est_speed": self._stream_step_mm() / max(1e-6, getattr(self, "_cycle_mean", 1.0)),
                "measure_mode": self.measure_combo.currentData(),
            }

        msg = status_msg.build_execution_status(
            offset_correction_on=APPLY_JOINT_OFFSET_CORRECTION,
            measure_mode=self.measure_combo.currentData(),
            p_feedback=pf_info,
            stream=stream_info,
            tcp_speed_mms=STREAM_TCP_SPEED_MMS,
            absorb_target_sec=ABSORB_TARGET_PERIOD_SEC if ABSORB_SPIKES_TEST else None,
            lag_compensation_joints=(list(se.POSE_DEPENDENT_LAG_JOINTS)
                                     if APPLY_POSE_DEPENDENT_LAG_COMPENSATION else None),
        )
        self.status_label.setText(msg)

        # [8차 세션 추가, §23.7] 실행 직후 최종 자세를 공유 로그에 남긴다 -
        # mycobot_vibration_diagnostic.py와 같은 로그를 쓴다. "움직이고 나서
        # 특정 상황에서 멈춰 진동"하는 순간을 사람이 따로 챙기지 않아도
        # 자동으로 남게 하려는 목적. 로그 실패가 실행 결과 표시를 막으면
        # 안 되므로 예외를 삼킨다.
        try:
            final_angles = self.mc.get_angles()
            if isinstance(final_angles, list) and len(final_angles) == 6:
                pose_log.log_pose(
                    label=f"curve_stop ({CODE_VERSION})",
                    joint_means=final_angles,
                    extra={
                        "lag_sec": lag if mode == "stream" else None,
                        "late_ratio": getattr(self, "_late_ratio", None) if mode == "stream" else None,
                        "measure_mode": self.measure_combo.currentData(),
                    },
                    announce=False,  # GUI 콘솔은 조용히 - 필요하면 vibration_diagnostic에서 조회
                )
        except Exception as e:
            # [8차 점검] 로그 실패가 결과 그래프 표시를 막으면 안 되므로 계속
            # 진행하되, 예전처럼 완전히 조용히 넘기지는 않는다 - 그러면 로그가
            # 계속 실패해도 사용자가 영영 모른다(자세 로그가 안 쌓이는데 원인을
            # 알 수 없는 상황). 콘솔에만 알린다.
            print(f"⚠️ 자세 로그 기록 실패(실행 자체는 정상 완료): {e}")

        # [8차 세션 추가, §23.12] 최종 자세뿐 아니라 경로 자체(self.points)도
        # 같이 남긴다 - "예전에 그렸던 곡선을 다시 불러와 수정·재검증·재실행"
        # 하고 싶다는 요청에 따른 것. 자세 로그와는 별도 파일(curve_path_log.jsonl)
        # 이지만 라벨에 같은 CODE_VERSION을 붙여 서로 연관 지을 수 있게 했다.
        #
        # [9차 세션] 라벨이 그동안 f"curve_stop ({CODE_VERSION})"로 항상 똑같았다 -
        # 실제로 어떤 곡선을 그렸든 로그에 전부 같은 이름으로 쌓여서, "불러오기"
        # 목록에서 서로 다른 곡선을 구분할 방법이 timestamp/점개수뿐이었다.
        # 점 목록(self.points) 자체에서 8자리 지문을 만들어 라벨에 넣는다 -
        # 좌표가 하나라도 다르면 다른 지문이 나오고, 완전히 같은 곡선을 다시
        # 실행하면 항상 같은 지문이 나온다(부동소수점 표현오차 방지를 위해
        # 소수점 3자리로 반올림 후 해시).
        import hashlib
        _fp_src = "|".join(
            repr(tuple(round(v, 3) if isinstance(v, float) else v for v in p.snapshot()))
            for p in self.points
        )
        curve_fp = hashlib.md5(_fp_src.encode("utf-8")).hexdigest()[:8]
        curve_label = f"curve_{curve_fp} ({CODE_VERSION})"

        # [9차 세션] '명령값만' 모드는 실측 없이 빠르게 훑어보기 위한 용도라
        # 실행할 때마다 로그에 쌓이면 정작 의미 있는(실측 데이터가 붙은)
        # 기록들 사이에 잡음처럼 끼어든다 - '매번 실측'일 때만 남긴다.
        if measure_mode == "full":
            try:
                curve_log.save_curve(
                    label=curve_label,
                    points_snapshots=[p.snapshot() for p in self.points],
                    extra={
                        "lag_sec": lag if mode == "stream" else None,
                        "mode": mode,
                        "measure_mode": measure_mode,
                        "correction_on": bool(APPLY_JOINT_OFFSET_CORRECTION),
                    },
                )
            except Exception as e:
                print(f"⚠️ 경로 로그 기록 실패(실행 자체는 정상 완료): {e}")

        # [9차 세션, §30] 오차모델 학습 데이터 적재.
        # 곡선/속도와 무관하게 **모든 실측 실행이 학습 데이터가 된다** -
        # 이게 ILC(곡선별 웨이포인트 표)와 갈리는 지점이다. 원본 시계열을
        # 그대로 저장하므로 나중에 특징 설계를 바꿔도 과거 데이터를 다시 쓸 수
        # 있다(mycobot_error_model.py docstring 참고).
        # 실시간 비용 0 - 실행이 다 끝난 뒤에 메모리에 있는 배열을 쓸 뿐이다.
        if measure_mode == "full" and mode == "stream" and cmd_times and stream_meas_times:
            try:
                err_model.save_run(
                    curve_label=curve_label,
                    tcp_speed_mms=STREAM_TCP_SPEED_MMS,
                    cmd_times=cmd_times,
                    cmd_angles=cmd_angles,
                    meas_times=stream_meas_times,
                    meas_angles=stream_meas_angles,
                    tau_ms_per_joint=None,   # 저장 안 함 - §27 스텝응답 실측값을 학습 스크립트에서 적용(격자탐색은 §30.3에서 원리적으로 불가능함을 확인, 폐기)
                    extra={
                        "p_feedback_on": bool(APPLY_P_FEEDBACK_CONTROL),
                        # [9차, §30.9] 이 실행이 오차모델 피드포워드를 이미 켠
                        # 채로 돌았는지 표시 - 켜져 있었다면 저장된 오차는
                        # "피드포워드가 이미 잡고 남은 잔차"라 순수 PD 시절
                        # 데이터와 성격이 다르다. 나중에 재학습할 때 섞을지
                        # 말지 판단하려면 이 표시가 있어야 한다.
                        "error_model_ff_applied": bool(APPLY_ERROR_MODEL_FEEDFORWARD),
                        # se._ERROR_MODEL(모듈 import 시점에 적재된 것)을 참조한다 -
                        # load_model()을 여기서 새로 부르면 그 사이 파일이 바뀌었을
                        # 때 "이 실행 중 실제로 쓰인 모델"과 어긋날 수 있다.
                        "error_model_ff_joints": (list(se._ERROR_MODEL.keys())
                                                  if APPLY_ERROR_MODEL_FEEDFORWARD else []),
                        # [10차, §43] 같은 이유로 자세지연보정 상태도 표시 - 이게
                        # 켜진 채로 돌았으면 웨이포인트 자체가 look-ahead로 이미
                        # 당겨져 있었으므로, 순수 PD/FF만의 궤적과 성격이 다르다.
                        "lag_compensation_on": bool(APPLY_POSE_DEPENDENT_LAG_COMPENSATION),
                        "gain": list(P_FEEDBACK_GAIN),
                        "kd": P_FEEDBACK_KD,
                        "deadband_deg": P_FEEDBACK_DEADBAND_DEG,
                        "smooth_alpha": P_FEEDBACK_SMOOTH_ALPHA,
                        "correction_on": bool(APPLY_JOINT_OFFSET_CORRECTION),
                    },
                )
            except Exception as e:
                print(f"⚠️ 오차모델 학습데이터 기록 실패(실행 자체는 정상 완료): {e}")

        # [9차 세션, 전송 리듬 조사] "빠름-빠름-빠름-느림"처럼 리듬이 느껴진다는
        # 관찰을 확정하려면 평균/표준편차가 아니라 사이클별 원본 시계열이
        # 필요하다. 스트리밍 모드로 실측(get_angles 왕복 포함) 실행했을 때만
        # 의미가 있다 - 큐 모드나 '명령값만' 모드는 애초에 이 루프를 안 쓰거나
        # 사이클 구조가 달라서 self._cycle_times가 없다.
        if mode == "stream" and getattr(self, "_cycle_times", None):
            try:
                # [9차, GC 조사] GC 발동 절대시각(self._gc_events)을 "몇 번째
                # 사이클에 끼었는가"로 바꾼다 - bisect로 그 시각 이후 처음 끝난
                # 사이클을 찾는다(mycobot_stub_test로 정확성 확인됨).
                import bisect
                cycle_end_abs = getattr(self, "_cycle_end_abs", None) or []
                gc_cycle_indices = []
                if cycle_end_abs:
                    for ev in getattr(self, "_gc_events", []):
                        idx = bisect.bisect_left(cycle_end_abs, ev)
                        idx = min(idx, len(cycle_end_abs) - 1)
                        gc_cycle_indices.append(idx)

                cycle_log.save_cycle_times(
                    label=curve_label,   # [9차 세션] curve_log와 같은 지문 - 같은
                                         # 곡선이면 두 로그의 라벨이 항상 일치한다.
                    cycle_times=self._cycle_times,
                    send_speed=STREAM_SEND_SPEED,
                    extra={
                        "measure_mode": self.measure_combo.currentData(),
                        "late_ratio": self._late_ratio,
                        "target_period_ms": self._target_period * 1000.0,
                        "gc_cycle_indices": gc_cycle_indices,
                        "absorb_spikes_test": ABSORB_SPIKES_TEST,
                    },
                )
            except Exception as e:
                print(f"⚠️ 사이클 리듬 로그 기록 실패(실행 자체는 정상 완료): {e}")

        self._show_result_plots(times, coord_samples, angle_samples,
                                cmd_times, cmd_angles, cmd_targets,
                                dispatch_done_t=dispatch_done_t, measure_mode=measure_mode)

    # [8차 점검] _next_trace_dir / _save_figure 위임 메서드는 삭제했다 -
    # §22에서 _show_result_plots가 mycobot_result_plots.py로 통째로 옮겨가면서
    # 그 안에서만 쓰이던 이 둘의 호출부도 같이 사라져, 껍데기만 남아 있었다
    # (AST 전수조사로 확인). 실제 구현은 rp.next_trace_dir / rp.save_figure에 있다.

    def _sample_measurement(self, times, coord_samples, angle_samples, t_start):
        """[패치 6] 실측 1회. **모든 실측 경로가 반드시 이 함수를 거친다.**

        get_coords()는 펌웨어 자체 기구학(툴 오프셋·DH가 URDF와 다를 수 있음)이라
        commanded 쪽(URDF FK)과 직접 비교하면 수 mm의 계통오차가 섞인다.
        여기서는 get_angles() 하나만 읽고 좌표는 항상 같은 URDF 체인으로 계산한다.
        덤으로 읽기가 2회 -> 1회로 줄어 전송 리듬 방해도 절반이다.
        """
        a = read_angles(self.mc)   # [10차, §39] 스위치 OFF면 mc.get_angles()와 100% 동일
        if not (isinstance(a, list) and len(a) == 6):
            return False
        q_meas = [0.0] * len(chain.links)
        for i, idx in enumerate(active_indices):
            q_meas[idx] = math.radians(a[i])
        fk_m = chain.forward_kinematics(q_meas)
        coord_samples.append((fk_m[:3, 3] * 1000.0).tolist() +
                             matrix_to_rxryrz(fk_m[:3, :3]).tolist())
        angle_samples.append(a)
        times.append(time.time() - t_start)
        return True

    def _sample_measurement_raw(self, times, angle_samples, t_start):
        """[지연 FK] get_angles()만 하고 순기구학은 하지 않는다.
        스트리밍 루프처럼 사이클 시간이 곧 TCP 속도로 직결되는 자리 전용 -
        FK는 나중에 한꺼번에 돌린다(execute_path의 '스트리밍 실측 일괄 FK 변환' 참고).
        큐 모드/마무리 대기 루프처럼 실시간 데드라인이 없는 곳은 그냥
        _sample_measurement를 계속 쓴다 - 굳이 나눌 이유가 없다.

        [10차, §45] se.APPLY_ASYNC_MEASUREMENT_PIPELINE이 켜져 있고 배경
        리더(self._async_reader)가 돌고 있으면, 블로킹 읽기 대신 그 리더의
        '가장 최근 측정값'을 즉시 가져다 쓴다 - 아직 한 번도 못 읽었으면
        (None, None)이 오므로 이번 사이클은 건너뛴다(스위치 OFF일 때와
        동일하게 100% 안전한 폴백).
        """
        if se.APPLY_ASYNC_MEASUREMENT_PIPELINE and getattr(self, "_async_reader", None) is not None:
            a, meas_t = self._async_reader.get_latest()
            if not (isinstance(a, list) and len(a) == 6):
                return False
            angle_samples.append(a)
            # [주의] 배경 스레드가 잰 실제 측정시각(meas_t)이 아니라 '지금'
            # 시각을 쓴다 - stream_meas_times는 다른 배열(cmd_times 등)과
            # 같은 t_start 기준 상대시각 체계를 공유해야 하고, 그 정합성이
            # 이 실험의 정확한 비교 대상(피드백 신선도)이므로 여기서
            # 임의로 바꾸면 §45 A/B 결과 해석이 오염된다 - 원래 방식대로
            # "이 사이클이 그 값을 실제로 소비한 시각"을 기록한다.
            times.append(time.time() - t_start)
            return True
        a = read_angles(self.mc)   # [10차, §39] 스위치 OFF면 mc.get_angles()와 100% 동일
        if not (isinstance(a, list) and len(a) == 6):
            return False
        angle_samples.append(a)
        times.append(time.time() - t_start)
        return True

    def _show_result_plots(self, times, coord_samples, angle_samples,
                           cmd_times=None, cmd_angles=None, cmd_targets=None,
                           dispatch_done_t=None, measure_mode=None):
        """[7차 분할, §22] mycobot_result_plots.show_result_plots로 위임.
        GUI 상태(status_label 텍스트, curve_waypoints)와 모듈 상수만 값으로
        넘겨준다 - 순수 함수 쪽에서는 self를 전혀 모른다."""
        return rp.show_result_plots(
            times, coord_samples, angle_samples,
            cmd_times, cmd_angles, cmd_targets,
            dispatch_done_t, measure_mode,
            self.status_label.text(),
            getattr(self, "curve_waypoints", None),
            chain, active_indices, matrix_to_rxryrz,
            JOINT_LIMITS_DEG, STREAM_TCP_SPEED_MMS,
            RESULTS_ROOT_DIR, CODE_VERSION,
        )

    # ---------------------------------------------------------
    # 초기화 / 다시 그리기
    # ---------------------------------------------------------
    def clear_path(self):
        self._push_undo("전체 지우기")
        self.points = []
        self.pending = {}
        self.curve_waypoints = []
        self.bad_point_idx = None
        self.bad_segment = None
        self.suggestion = None
        self.suggestion_region = []
        self.execute_btn.setEnabled(False)
        self.status_label.setText("전체 지움. 새 점을 클릭해서 시작하세요. "
                                  "('마지막 행동 취소'로 되살릴 수 있습니다)")
        self._redraw_all()

    def _push_undo(self, label):
        """[6차] 상태를 바꾸는 행동 **직전에** 현재 점 목록을 통째로 저장한다.
        label은 되돌릴 때 사용자에게 "무엇을 되돌렸는지" 알려주기 위한 것."""
        self.undo_stack.append((label, [p.snapshot() for p in self.points]))
        if len(self.undo_stack) > self.UNDO_MAX:
            self.undo_stack.pop(0)

    def _invalidate_after_edit(self):
        """점이 바뀌었으니 검증 결과/추천을 모두 무효화한다 (여러 곳에서 반복되던 코드)."""
        self.curve_waypoints = []
        self.bad_point_idx = None
        self.bad_segment = None
        self.suggestion = None
        self.suggestion_region = []
        self.execute_btn.setEnabled(False)

    def undo_last(self):
        """[6차 재작성] 마지막으로 한 '행동'을 되돌린다.

        예전엔 무슨 행동을 했든 무조건 끝 앵커를 제거해서, 방금 점을 옮겼거나
        곡선을 분할했는데도 엉뚱한 점이 사라졌다. 이제 스냅샷 스택에서 직전
        상태를 그대로 복원하므로 '방금 한 그 행동'만 정확히 취소된다.
        """
        if not self.undo_stack:
            self.status_label.setText("되돌릴 행동이 없습니다.")
            return
        label, snap = self.undo_stack.pop()
        self.points = [PathPoint.from_snapshot(t) for t in snap]
        self.pending = {}          # 2클릭 조합 중이던 좌표도 초기화(어정쩡한 상태 방지)
        self.drag_index = None
        self._invalidate_after_edit()
        self._redraw_all()
        remain = len(self.undo_stack)
        self.status_label.setText(
            f"↩️ 되돌림: {label} · 구간 {self._n_segments()}개 / 제어점 {self._n_segments()}개"
            + (f" (되돌리기 {remain}단계 더 가능)" if remain else " (더 되돌릴 행동 없음)"))

    def _redraw_all(self):
        # [8차 세션, §23.13] cd.draw_view_2d/3d로 위임 - self.points 등
        # 편집 상태를 명시적 인자로 넘긴다(§22의 _show_result_plots와 같은 패턴).
        cd.draw_view_2d(self.ax_xy, self.canvas_xy, "xy", self.points,
                        self.x_value, self.y_value, self.z_value,
                        self.bad_point_idx, self.bad_segment,
                        self.suggestion, self.suggestion_region)
        cd.draw_view_2d(self.ax_yz, self.canvas_yz, "yz", self.points,
                        self.x_value, self.y_value, self.z_value,
                        self.bad_point_idx, self.bad_segment,
                        self.suggestion, self.suggestion_region)
        cd.draw_view_2d(self.ax_xz, self.canvas_xz, "xz", self.points,
                        self.x_value, self.y_value, self.z_value,
                        self.bad_point_idx, self.bad_segment,
                        self.suggestion, self.suggestion_region)
        cd.draw_view_3d(self.ax_3d, self.canvas_3d, self.points,
                        self.bad_point_idx, self.bad_segment,
                        self.suggestion, self.suggestion_region)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PathEditor()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()