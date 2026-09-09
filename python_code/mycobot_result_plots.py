# -*- coding: utf-8 -*-
"""
mycobot_result_plots.py
========================
[7차 분할, §22] `_show_result_plots` / `_next_trace_dir` / `_save_figure`를
`mycobot_path_editor.py`에서 분리했다. §17.7(6차)에서 이미 `mycobot_curve_math`,
`mycobot_stream_exec`로 GUI 무관 로직을 뺀 것과 같은 이유·같은 패턴이다:

  - `_show_result_plots`는 300줄 넘는 순수 후처리(그래프 4장 생성)인데
    GUI 이벤트와는 무관하다 - `self.mc`(로봇 핸들)도, `self.<qt위젯>`도
    실제로는 안 건드린다(상태줄 "텍스트"만 읽을 뿐).
  - PathEditor에는 얇은 위임 메서드만 남긴다(기존 호출부 무변경).

[6차 §18.2 4번, §21.3 2번에서 예고된 작업. 실기 불필요 - 순수 리팩토링.]

이 모듈은 Qt를 import하지 않는다 - `self.status_label.text()`,
`self.curve_waypoints` 등 GUI 상태는 전부 호출자가 값으로 넘겨준다.
"""

import os
import re
import math
import numpy as np


# ---------------------------------------------------------------------------
# [10차 세션, §47] 각가속도 계산 - 비균일 국소 다항식 미분
# ---------------------------------------------------------------------------
#   `mycobot_accel_plot.py`(§46)와 **같은 수식**이지만 여기에 따로 둔다:
#   그 모듈은 최상단에서 matplotlib.use("TkAgg")를 하고 대화형 메뉴를 가진
#   독립 실행 도구라, 여기서 import하면 백엔드 설정까지 끌려온다. 이 모듈은
#   "Qt/무거운 의존성을 안 들인다"는 원칙(위 docstring)으로 분리된 것이므로
#   40줄짜리 순수 수치함수 하나를 복제하는 편이 낫다고 판단했다.
#   **수식을 고칠 일이 생기면 두 곳을 같이 고칠 것.**
#
#   왜 이 방식인가(요약, 자세한 근거는 §46):
#     - 엔코더 양자화 0.08도를 그냥 두 번 미분하면 가짜 α가 ~35 deg/s² 생겨
#       신호가 노이즈에 묻힌다(실측: 계획값과의 상관 0.202).
#     - 측정 시각 간격이 30/60/90ms 삼봉분포(48ms는 0개)라 균일 샘플링을
#       가정하는 표준 Savitzky-Golay는 쓸 수 없다.
#     - 각 점 주변에서 **실제 상대시각**으로 다항식을 적합하고 그 계수에서
#       해석적으로 미분을 뽑는다: y(t_i+τ) ≈ Σ c_k τ^k  ->  y^(d) = d!·c_d
ACCEL_HALF_WINDOW_SEC = 0.50   # §46에서 35mm/s 기준 튜닝(상관 0.915, 진폭 보존)
ACCEL_POLYORDER = 2


def local_poly_deriv(t, y, half_window_sec=ACCEL_HALF_WINDOW_SEC,
                     polyorder=ACCEL_POLYORDER, deriv=2):
    """비균일 샘플링에서 국소 다항식 적합으로 deriv차 미분을 구한다.
    (mycobot_accel_plot.local_poly_deriv와 동일 - 위 주석 참고)"""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    out = np.zeros(n)
    if n <= polyorder + 1:
        return out
    for i in range(n):
        mask = np.abs(t - t[i]) <= half_window_sec
        if mask.sum() < polyorder + 1:
            mask = np.zeros(n, dtype=bool)
            mask[np.argsort(np.abs(t - t[i]))[:min(polyorder + 2, n)]] = True
        try:
            c = np.polyfit(t[mask] - t[i], y[mask], polyorder)
        except Exception:
            continue
        if len(c) > deriv:
            out[i] = math.factorial(deriv) * c[-(deriv + 1)]
    return out


def next_trace_dir(measure_mode, results_root_dir, code_version):
    """[분류저장] 측정모드별로 따로 번호를 매겨 하위 폴더를 만든다.
      - 'full'(매번 실측) -> everytime_trace_N
    ('none'/명령값만 모드는 show_result_plots에서 그래프 자체를 생략하고
     조기 반환하므로 이 함수까지 오지 않는다 - only_trace_N은 더 이상 만들지 않는다)

    같은 버전 폴더 안에서 여러 번 실행해도 매번 새 번호의 폴더가 생기므로
    예전처럼 figure_1.png가 덮어써지는 일이 없다.

    [주의] 한 번의 실행(run) 안에서는 반드시 한 번만 호출해서 그 결과를
    재사용해야 한다. figure마다 매번 새로 부르면, 먼저 만든 폴더를 자기가
    스캔에서 또 발견해 번호가 하나씩 밀린다."""
    prefix = "everytime_trace" if measure_mode == "full" else "trace"
    version_dir = os.path.join(results_root_dir, f"{code_version}_figure")
    os.makedirs(version_dir, exist_ok=True)

    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    existing_n = []
    try:
        for name in os.listdir(version_dir):
            m = pat.match(name)
            if m:
                existing_n.append(int(m.group(1)))
    except OSError:
        pass
    n = max(existing_n, default=0) + 1

    run_dir = os.path.join(version_dir, f"{prefix}_{n}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_figure(fig, n, run_dir):
    """[버전 자동저장] run_dir/figure_{n}.png 로 저장.
    저장 실패(폴더 접근 불가 등)해도 창 표시 자체는 막지 않는다 - 조용히 경고만."""
    if run_dir is None:
        return None
    try:
        out_path = os.path.join(run_dir, f"figure_{n}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return out_path
    except Exception as e:
        print(f"⚠️ 결과 이미지 저장 실패(figure_{n}): {e}")
        return None


def show_result_plots(times, coord_samples, angle_samples,
                       cmd_times, cmd_angles, cmd_targets,
                       dispatch_done_t, measure_mode,
                       status_text, curve_waypoints,
                       chain, active_indices, matrix_to_rxryrz,
                       joint_limits_deg, stream_tcp_speed_mms,
                       results_root_dir, code_version):
    """명령값(통신 불필요·항상 존재)을 기본으로 그리고, 실측 샘플이 있으면 겹쳐 그린다.

    [분할 시 시그니처 변경] 원래 PathEditor 메서드는 self.status_label,
    self.curve_waypoints 등 GUI 상태를 암묵적으로 참조했다. 순수 함수로
    만들면서 그 값들을 명시적 인자로 받는다 - 호출부(위임 메서드)에서
    self.status_label.text() / self.curve_waypoints / 모듈 상수를 그대로
    넘겨주면 동작은 완전히 동일하다.
    """
    import matplotlib.pyplot as plt

    # [명령값만 모드는 그래프 자체를 생략] 이 모드는 로봇을 전혀 읽지 않으므로
    # commanded 쪽 IK/스플라인 계산값을 자기 자신과 비교하는 셈이다.
    # 실제로 뜨는 cross-track ~0.01mm, tracking ~0.6mm 같은 수치는 로봇 오차가
    # 아니라 IK 잔차·시간이산화 아티팩트일 뿐이라 정보가 없다.
    # -> 만들지도, only_trace_N 폴더에 저장하지도 않는다.
    if measure_mode == "none":
        print("ℹ️ 명령값만 모드: 실제 측정이 없어 결과 그래프를 생략합니다.")
        return

    run_dir = next_trace_dir(measure_mode, results_root_dir, code_version)

    # [상태줄 텍스트도 같이 저장] 실행 완료 메시지(밀림/주기/오차 요약)를
    # 이미지 4장과 같은 폴더에 status.txt로 남긴다. execute_path()가 이미
    # self.status_label.setText(msg)를 호출한 뒤 이 함수를 부르므로,
    # 그 텍스트를 status_text 인자로 그대로 받는다.
    try:
        with open(os.path.join(run_dir, "status.txt"), "w", encoding="utf-8") as f:
            f.write(status_text)
    except Exception as e:
        print(f"⚠️ 상태 텍스트 저장 실패: {e}")

    has_cmd = bool(cmd_times) and len(cmd_times) >= 2
    has_meas = len(coord_samples) >= 2
    if not has_cmd and not has_meas:
        return

    # --- 명령값으로부터 위치/자세/관절각/속도 계산 (순기구학, 통신 불필요) ---
    if has_cmd:
        ct = np.array(cmd_times, dtype=float)
        ca = np.array(cmd_angles, dtype=float)          # (N,6) deg
        cpos, crpy = [], []
        for row in ca:
            q = [0.0] * len(chain.links)
            for i, idx in enumerate(active_indices):
                q[idx] = math.radians(row[i])
            fk = chain.forward_kinematics(q)
            cpos.append(fk[:3, 3] * 1000.0)
            crpy.append(matrix_to_rxryrz(fk[:3, :3]))
        cpos = np.array(cpos)
        crpy = np.array(crpy)

    if has_meas:
        mcoords = np.array(coord_samples, dtype=float)
        mangles = np.array(angle_samples, dtype=float)
        mt = np.array(times, dtype=float)

    # ┌─ [패치 7] 오일러각 ±180 랩어라운드 해제 ─────────────────────────┐
    # │ matrix_to_rxryrz는 [-180,180]을 돌려준다. 실제 자세가 연속이어도  │
    # │ rx가 180을 넘는 순간 -180으로 점프해 그래프에 거대한 수직선이     │
    # │ 그어진다(Figure 1 alpha 서브플롯이 읽을 수 없었던 이유).          │
    # │ 실행에는 영향 없다 - 보간은 회전행렬로 하므로. 표시만 고친다.     │
    # └───────────────────────────────────────────────────────────────────┘
    def _unwrap_deg(a2d):
        return np.degrees(np.unwrap(np.radians(np.asarray(a2d, dtype=float)), axis=0))

    if has_cmd:
        crpy_plot = _unwrap_deg(crpy)
    if has_meas:
        mrpy_plot = _unwrap_deg(mcoords[:, 3:6])
        if has_cmd:
            # [주의] 두 계열을 따로 unwrap하면 서로 다른 분기(±360)에 앉을 수 있다.
            #   예: 같은 자세인데 commanded=+178, measured=-179 -> 그래프가 357도
            #   벌어져 보인다. 첫 샘플 기준으로 measured를 commanded 쪽에 맞춰준다.
            for i in range(3):
                k = round((crpy_plot[0, i] - mrpy_plot[0, i]) / 360.0)
                mrpy_plot[:, i] += k * 360.0

    # ┌─ [패치 8] 측정속도의 미분 노이즈 억제 ────────────────────────────┐
    # │ 인접 두 샘플을 28ms로 나누면, get_angles의 각도 양자화(~0.1도,    │
    # │ 위치로 ~0.5mm)가 ±18mm/s의 잡음으로 증폭된다. Figure 3의 빨간     │
    # │ 점이 ±20mm/s로 지글거린 게 전부 이것이고, 그 그래프로는 아무      │
    # │ 판단도 할 수 없었다. 중심차분 윈도우로 넓게 미분한다.             │
    # └───────────────────────────────────────────────────────────────────┘
    def _windowed_speed(t, p, win=5):
        t = np.asarray(t, dtype=float)
        p = np.asarray(p, dtype=float)
        w = int(max(1, min(win, len(t) - 1)))
        d = np.linalg.norm(p[w:] - p[:-w], axis=1)
        dt = np.maximum(t[w:] - t[:-w], 1e-4)
        return (t[w:] + t[:-w]) * 0.5, d / dt        # 시각은 창의 중앙

    # ---- Figure 1: 위치 & 자세 vs 시간 ----
    fig1, axes = plt.subplots(2, 3, figsize=(16, 8))
    for i, name in enumerate(["X", "Y", "Z"]):
        ax = axes[0, i]
        if has_cmd:
            ax.plot(ct, cpos[:, i], "-", color="tab:blue", linewidth=1.2, label="commanded")
        if has_meas:
            ax.plot(mt, mcoords[:, i], ".", color="tab:red", markersize=4, label="measured")
        ax.set_title(f"{name} (mm) vs time")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)
    for i, name in enumerate(["alpha (rx)", "beta (ry)", "gamma (rz)"]):
        ax = axes[1, i]
        if has_cmd:
            ax.plot(ct, crpy_plot[:, i], "-", color="tab:green", linewidth=1.2, label="commanded")
        if has_meas:
            ax.plot(mt, mrpy_plot[:, i], ".", color="tab:red", markersize=4, label="measured")
        ax.set_title(f"{name} (deg) vs time")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig1.suptitle("Position & Orientation over time  (line=commanded, dots=measured)")
    fig1.tight_layout()
    save_figure(fig1, 1, run_dir)
    fig1.show()

    # ---- Figure 2: 관절 각도 vs 시간 ----
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8))
    for j in range(6):
        ax = axes2[j // 3, j % 3]
        lo, hi = joint_limits_deg[j]
        ax.axhspan(lo, hi, color="green", alpha=0.08)
        ax.axhline(lo, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(hi, color="red", linestyle="--", linewidth=0.8)
        if has_cmd:
            ax.plot(ct, ca[:, j], "-", color="m", linewidth=1.2, label="commanded")
        if has_meas:
            ax.plot(mt, mangles[:, j], ".", color="tab:red", markersize=4, label="measured")
        ax.set_title(f"Joint {j+1} angle vs time")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("deg")
        ax.grid(alpha=0.3)
    axes2[0, 0].legend(fontsize=8)

    # ┌─ [패치 10] §9-5 "Z축 오차 가설" 자동 판정 ────────────────────────┐
    # │ 문서의 가설: J2/J3는 중력을 이고 있어 추종지연이 J1보다 크다.      │
    # │ 그런데 Figure 2를 눈으로 보면 J1의 간격도 비슷해 보인다.           │
    # │ 관절별로 (명령-실측 오차) 대 (명령 각속도)의 기울기를 재면 결론이  │
    # │ 난다. 기울기 = 그 관절의 유효 추종지연(ms).                        │
    # │   · 6축이 다 비슷하다 -> 순수 속도비례 지연. 속도를 낮추면 해결.   │
    # │   · J2/J3만 크다      -> 중력 요인. 소프트웨어로는 한계.           │
    # └───────────────────────────────────────────────────────────────────┘
    if has_cmd and has_meas and len(mt) > 10:
        # [6차] 콘솔에만 찍히면 매번 복사-붙여넣기 해야 해서 §7.1(Z축 오차
        # 가설) 데이터가 잘 안 쌓였다. status.txt처럼 파일로도 남긴다 -
        # 표를 만드는 김에 문자열로도 조립해서 print와 파일 저장에 그대로 쓴다.
        lag_lines = []
        # [9차 세션, §27.7 후속 - τ 하한 실험 준비] 지금까지는 이 표에 목표
        # 속도가 안 남아서, STREAM_TCP_SPEED_MMS를 바꿔가며 여러 번 실행해도
        # 어느 결과가 어느 속도였는지 나중에 폴더 순서로 기억에 의존해야
        # 했다. 이미 이 함수가 받는 stream_tcp_speed_mms 인자를 한 줄
        # 추가하는 것만으로 해결된다 - 속도별 τ 비교(§23.16-3/§26.9-4)를
        # joint_lag.txt만 모아서 바로 할 수 있게.
        lag_lines.append(f"[목표 TCP 속도] {stream_tcp_speed_mms}mm/s")
        # [9차 세션, §28.5 후속] "속도를 바꾸면 재검증해야 하는데, 그러면
        # 웨이포인트 간격도 같이 바뀌어서 진짜 '같은 곡선'으로 비교하는 게
        # 맞는지" 질문에 대한 답 - 간격 자체는 같은 곡선(같은 제어점)을
        # 다른 밀도로 재샘플링한 것뿐이라 §14/§27의 "다른 곡선을 섞으면
        # 오염된다"는 문제와는 다르다. 다만 간격이 넓을수록 코너를 더
        # 잘라먹어서(§4.3) 급커브 구간의 순수 서보지연과 코너컷 효과가
        # 섞일 수 있다 - 나중에 τ-속도 그래프가 이상하면 이 줄로 그때
        # 간격이 유독 넓었는지 바로 확인할 수 있게 남긴다.
        if curve_waypoints and len(curve_waypoints) > 1:
            targets = np.array([wp[1] for wp in curve_waypoints], dtype=float)
            step_mm = float(np.mean(np.linalg.norm(np.diff(targets, axis=0), axis=1)))
            lag_lines.append(f"[웨이포인트] {len(curve_waypoints)}개, 평균 간격 {step_mm:.3f}mm")
        lag_lines.append("[관절별 추종 특성]  (명령-실측 오차 vs 명령 각속도의 회귀)")
        lag_lines.append("  관절 | 평균오차(도) | 최대오차(도) | 유효지연(ms) | 각속도범위(도/s)")
        # np.gradient는 좌표가 '엄격히' 증가해야 한다. 같은 ms에 두 샘플이
        # 찍히면 0으로 나눠 nan이 번진다(빠른 루프에서 실제로 발생 가능).
        mt_mono = np.maximum.accumulate(mt)
        mt_mono += np.arange(len(mt_mono)) * 1e-9
        for j in range(6):
            cmd_i = np.interp(mt, ct, ca[:, j])
            err_j = np.abs(mangles[:, j] - cmd_i)
            vel_j = np.abs(np.gradient(cmd_i, mt_mono))
            ok_m = np.isfinite(vel_j) & np.isfinite(err_j)
            if ok_m.sum() > 3 and vel_j[ok_m].max() > 1.0:
                k = float(np.polyfit(vel_j[ok_m], err_j[ok_m], 1)[0])   # 초 단위
            else:
                k = float("nan")
            lag_lines.append(f"   J{j+1}  | {err_j.mean():11.3f}  | {err_j.max():11.3f}  |"
                              f" {k*1000:11.1f}  | {vel_j.max():14.1f}")
        lag_lines.append("  → 6축의 유효지연이 비슷하면 '속도비례 추종지연'(속도를 낮추면 개선),"
                          "\n    J2/J3만 크면 '중력 부하'가 맞다(소프트웨어로는 한계).")
        lag_text = "\n".join(lag_lines)
        print("\n" + lag_text + "\n")
        try:
            with open(os.path.join(run_dir, "joint_lag.txt"), "w", encoding="utf-8") as f:
                f.write(lag_text + "\n")
        except Exception as e:
            print(f"⚠️ 관절별 추종 특성 표 저장 실패: {e}")

    fig2.suptitle("Joint angles over time (red dashed = hard limits)")
    fig2.tight_layout()
    save_figure(fig2, 2, run_dir)
    fig2.show()

    # ---- Figure 3: 속도 프로파일 ----
    fig3, ax3 = plt.subplots(figsize=(11, 4))
    if has_cmd and len(ct) > 2:
        # [주의] 명령 시각을 time.time()으로 그대로 쓰면 파이썬 sleep의 타이밍 지터가
        #        속도로 증폭되어(간격이 거의 0인 사이클 -> 수천 mm/s) 실제와 무관한
        #        톱니가 나타난다. 로봇 내부 서보는 이 지터를 흡수하므로,
        #        '계획된 균일 스케줄' 기준으로 그려야 의도한 속도 프로파일이 보인다.
        nominal_t = np.linspace(ct[0], ct[-1], len(ct))
        dt = np.diff(nominal_t); dt[dt <= 1e-4] = 1e-4
        v_cmd = np.linalg.norm(np.diff(cpos, axis=0), axis=1) / dt
        ax3.plot(nominal_t[1:], v_cmd, "-", color="tab:blue", linewidth=1.2,
                 label=f"commanded, nominal schedule (avg {v_cmd.mean():.0f}mm/s)")
    if has_meas and len(mt) > 2:
        # 원본(1샘플 차분)은 양자화 잡음이라 흐리게만 깔고,
        # 실제로 읽는 건 창 미분(patch 8) 쪽이다.
        dtm = np.diff(mt); dtm[dtm <= 1e-4] = 1e-4
        v_raw = np.linalg.norm(np.diff(mcoords[:, :3], axis=0), axis=1) / dtm
        ax3.plot(mt[1:], v_raw, "-", color="tab:red", linewidth=0.6, alpha=0.20,
                 label="measured (1-sample diff, quantization noise)")
        ts_m, v_meas = _windowed_speed(mt, mcoords[:, :3], win=5)
        ax3.plot(ts_m, v_meas, ".-", color="tab:red", markersize=3, linewidth=1.2,
                 label=f"measured (5-sample window diff, avg {v_meas.mean():.0f}mm/s)")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("speed (mm/s)")
    ax3.axhline(stream_tcp_speed_mms, color="k", linestyle=":", linewidth=1,
                label=f"target = {stream_tcp_speed_mms}mm/s")
    ax3.set_ylim(bottom=0)
    ax3.set_title("TCP speed profile  (commanded=nominal schedule / measured=actual)")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    save_figure(fig3, 3, run_dir)
    fig3.show()

    # ---- Figure 4: 계획 경로 vs 실제(또는 명령) 경로 ----
    planned = np.array([t for (_, t) in curve_waypoints], dtype=float) \
        if curve_waypoints else None
    track = mcoords[:, :3] if has_meas else (cpos if has_cmd else None)
    track_label = "measured path" if has_meas else "commanded path"
    track_t = mt if has_meas else ct
    if planned is not None and len(planned) >= 2 and track is not None:
        devs = []
        for pt in track:
            seg_d = []
            for i in range(len(planned) - 1):
                a, b = planned[i], planned[i + 1]
                ab = b - a
                L2 = float(ab @ ab)
                if L2 < 1e-9:
                    seg_d.append(np.linalg.norm(pt - a)); continue
                u = float(np.clip((pt - a) @ ab / L2, 0, 1))
                seg_d.append(np.linalg.norm(pt - (a + ab * u)))
            devs.append(min(seg_d))
        devs = np.array(devs)

        # ┌─ [패치 9] 오차 지표 3종 분리 ─────────────────────────────────┐
        # │ 기존 devs는 '측정점에서 계획 폴리라인까지의 최단거리'다.       │
        # │ 즉 cross-track(옆으로 벗어난 양)만 재고, along-track(경로      │
        # │ 위에서 얼마나 뒤처졌나)은 정의상 0으로 나온다.                 │
        # │ 실제로 종점을 12mm 남기고 멈춘 실행에서도 이 지표는 3.3mm로    │
        # │ 끝나 오차를 완전히 숨겼다.                                     │
        # │                                                               │
        # │ 이제 세 가지를 따로 낸다:                                      │
        # │   ① cross-track  : 경로 '모양'이 맞는가        (기존 devs)     │
        # │   ② tracking     : 같은 시각에 같은 자리인가  (지연 포함)      │
        # │   ③ shape-only   : 최적 시간지연 τ를 보정한 뒤 남는 오차        │
        # │ ②−③ 의 차이가 곧 '순수한 추종지연'이고, τ가 그 크기다.        │
        # │ 종점오차는 별도로 suptitle에 찍는다.                           │
        # └───────────────────────────────────────────────────────────────┘
        err_track = err_shape = None
        tau_best = 0.0
        if dispatch_done_t and dispatch_done_t > 1e-3 and len(planned) >= 2:
            planned_t = np.linspace(0.0, float(dispatch_done_t), len(planned))

            def _plan_at(tq):
                tq = np.clip(tq, planned_t[0], planned_t[-1])
                return np.stack([np.interp(tq, planned_t, planned[:, i]) for i in range(3)],
                                axis=1)

            err_track = np.linalg.norm(track - _plan_at(track_t), axis=1)
            # 로봇이 τ만큼 뒤처져 있다면, 시각 t의 로봇은 계획의 t-τ 지점에 있어야 한다.
            taus = np.linspace(0.0, min(1.5, float(dispatch_done_t) * 0.5), 151)
            means = [np.linalg.norm(track - _plan_at(track_t - tau), axis=1).mean()
                     for tau in taus]
            tau_best = float(taus[int(np.argmin(means))])
            err_shape = np.linalg.norm(track - _plan_at(track_t - tau_best), axis=1)

        err_final = float(np.linalg.norm(track[-1] - planned[-1]))

        ncol = 5 if err_track is not None else 4
        fig4, axes4 = plt.subplots(1, ncol, figsize=(4.6 * ncol, 5))
        for ax4, (i, j, ni, nj) in zip(axes4[:3],
                                       [(0, 1, "X", "Y"), (1, 2, "Y", "Z"), (0, 2, "X", "Z")]):
            ax4.plot(planned[:, i], planned[:, j], "r--", linewidth=1.5, label="planned path")
            ax4.plot(track[:, i], track[:, j], "b.-", markersize=3, linewidth=1, label=track_label)
            ax4.plot(planned[0, i], planned[0, j], "go", markersize=8, label="start")
            ax4.plot(planned[-1, i], planned[-1, j], "r*", markersize=12, label="target")
            ax4.set_xlabel(f"{ni} (mm)"); ax4.set_ylabel(f"{nj} (mm)")
            ax4.set_title(f"{ni}{nj} plane"); ax4.grid(alpha=0.3)
            ax4.set_aspect("equal", adjustable="datalim")
            # [좌우 반전] 로봇을 실제로 바라보는 방향 기준 좌/우가 X축 부호와
            # 반대였다는 피드백 -> X가 가로축인 두 평면(XY, XZ)만 뒤집는다.
            # YZ 평면은 가로축이 Y라 대상이 아니다. 데이터/오차 계산에는
            # 영향 없음 - 축 표시 방향만 바꾼다.
            if ni == "X":
                ax4.invert_xaxis()
        axes4[0].legend(loc="best", fontsize=8)

        ax_dev = axes4[3]
        ax_dev.plot(track_t, devs, "m.-", markersize=3, linewidth=1)
        ax_dev.axhline(devs.max(), color="orange", linestyle="--", linewidth=1,
                       label=f"max = {devs.max():.2f}mm")
        ax_dev.axhline(devs.mean(), color="gray", linestyle=":", linewidth=1,
                       label=f"mean = {devs.mean():.2f}mm")
        ax_dev.set_xlabel("time (s)")
        ax_dev.set_ylabel("cross-track deviation (mm)")
        ax_dev.set_title("Cross-track only (lag not captured)")
        ax_dev.grid(alpha=0.3); ax_dev.legend(fontsize=8); ax_dev.set_ylim(bottom=0)

        if err_track is not None:
            ax_tr = axes4[4]
            ax_tr.plot(track_t, err_track, "-", color="tab:red", linewidth=1.2,
                       label=f"tracking (avg {err_track.mean():.2f} / max {err_track.max():.2f}mm)")
            ax_tr.plot(track_t, err_shape, "-", color="tab:blue", linewidth=1.2,
                       label=f"tau={tau_best*1000:.0f}ms corrected "
                             f"(avg {err_shape.mean():.2f} / max {err_shape.max():.2f}mm)")
            ax_tr.set_xlabel("time (s)")
            ax_tr.set_ylabel("3D error vs time-aligned plan (mm)")
            ax_tr.set_title("Tracking error (raw / lag-corrected)")
            ax_tr.grid(alpha=0.3); ax_tr.legend(fontsize=7); ax_tr.set_ylim(bottom=0)

        head = (f"Planned vs {track_label}  |  final error {err_final:.2f}mm  |  "
                f"cross-track avg {devs.mean():.2f} / max {devs.max():.2f}mm")
        if err_track is not None:
            head += (f"  |  tracking avg {err_track.mean():.2f}mm "
                     f"(lag tau={tau_best*1000:.0f}ms -> corrected {err_shape.mean():.2f}mm)")
        fig4.suptitle(head, fontsize=11)
        fig4.tight_layout()
        save_figure(fig4, 4, run_dir)
        fig4.show()

    # ---- [10차 세션, §47] Figure 5: 관절 각가속도 (측정 vs 계획) ----
    # 실측 각가속도는 양자화 노이즈 때문에 나이브 이중미분으로는 못 뽑는다
    # (§46). 위 local_poly_deriv로 비균일 국소 다항식 미분을 쓴다.
    # 이 블록은 fig4의 if 안이 아니라 바깥이다 - 실측만 있고 명령값 쪽
    # 경로계산이 실패한 경우에도 각가속도는 그릴 수 있어야 하므로.
    saved5 = None
    try:
        if has_meas and len(times) >= 8:
            ma = np.asarray(angle_samples, dtype=float)   # (N,6) deg
            mt = np.asarray(times, dtype=float)
            fig5, axes5 = plt.subplots(2, 3, figsize=(18, 9))
            for j in range(6):
                ax = axes5[j // 3][j % 3]
                al_m = local_poly_deriv(mt, ma[:, j])

                # [원본도 숨기지 않는다] figure_2(TCP 속도)가 1샘플 차분을 흐리게
                # 깔고 창 미분을 겹쳐 그리는 것과 같은 관례다 - 필터가 얼마나
                # 뭉갰는지 보는 사람이 직접 판단할 수 있어야 한다.
                # **보조 y축을 쓰는 이유**: 나이브 이중미분은 양자화(0.08도)
                # 때문에 진폭이 필터 결과의 ~5배(J4 실측 σ 93 vs 20)라, 같은
                # 축에 그리면 정작 읽어야 할 필터 곡선이 납작해져 버린다.
                al_naive = np.gradient(np.gradient(ma[:, j], mt), mt)
                ax_raw = ax.twinx()
                ax_raw.plot(mt, al_naive, "-", color="tab:red",
                            linewidth=0.6, alpha=0.20)
                ax_raw.set_yticks([])          # 눈금은 지운다 - 배경 참고용이므로
                ax_raw.set_zorder(0)
                ax.set_zorder(1)
                ax.patch.set_visible(False)    # 본 축이 위로 오도록

                if has_cmd and len(cmd_times) >= 5:
                    # 계획값은 매끈하므로 단순 미분으로 충분(기준선 역할)
                    ct_a = np.asarray(cmd_times, dtype=float)
                    ca_a = np.asarray(cmd_angles, dtype=float)
                    om_c = np.gradient(ca_a[:, j], ct_a)
                    al_c = np.gradient(om_c, ct_a)
                    ax.plot(ct_a, al_c, "-", linewidth=1.1, alpha=0.7,
                            color="tab:blue", label="commanded")
                ax.plot(mt, al_m, "-", linewidth=1.4, color="tab:orange",
                        label="measured (filtered)")
                # 배경 나이브는 보조축에 그려서 ax.legend()에 안 잡히므로
                # 범례용 더미를 본 축에 추가한다(데이터는 안 그림).
                ax.plot([], [], "-", color="tab:red", linewidth=0.6, alpha=0.35,
                        label="measured (naive 2nd diff, quantization noise)")
                ax.set_title(f"Joint {j+1} angular acceleration")
                ax.set_xlabel("time (s)")
                if j % 3 == 0:
                    ax.set_ylabel("alpha (deg/s^2)")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=6)
            fig5.suptitle(
                f"Joint angular acceleration  |  local polynomial fit "
                f"+/-{ACCEL_HALF_WINDOW_SEC*1000:.0f}ms, order {ACCEL_POLYORDER} "
                f"(non-uniform sampling aware)  |  measured is low-pass filtered "
                f"- not instantaneous  |  faint red = unfiltered (separate scale)",
                fontsize=11)
            fig5.tight_layout()
            saved5 = save_figure(fig5, 5, run_dir)
            fig5.show()
    except Exception as e:
        # 각가속도는 부가정보다 - 실패해도 기존 결과 4장/로그를 막으면 안 된다
        print(f"⚠️ 각가속도 그래프(figure_5) 생성 실패: {e}")

    n_fig = 5 if saved5 else 4
    print(f"💾 결과 이미지 {n_fig}장 + status.txt(+joint_lag.txt) 저장됨: {run_dir}")
