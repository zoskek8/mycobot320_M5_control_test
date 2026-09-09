# -*- coding: utf-8 -*-
"""
mycobot_status_msg.py
======================
[9차 세션, §29.16] `mycobot_path_editor.execute_path()` 끝부분에서 상태줄
문자열을 조립하던 ~80줄을 그대로 떼어낸 순수 함수 모듈이다.

**왜 하필 이 블록을 떼었나 - 버그가 실제로 여기서만 두 번 났다:**
  - §29.6: `APPLY_P_FEEDBACK_AUTO_GAIN`(존재하지 않는 이름) -> NameError
  - §29.14: `P_FEEDBACK_GAIN_MAX`가 리스트가 된 뒤 `:.2f` 포맷 -> TypeError

둘 다 "P 피드백을 켜고 스트리밍 실행을 끝까지 돌려야" 비로소 터지는
자리였다. 즉 **로봇을 연결하고 곡선을 완주해야만 발견되는 버그**였고,
정작 내용은 문자열 조립뿐이라 로봇과 아무 상관이 없었다. 순수 함수로
빼면 `python3 mycobot_status_msg.py` 한 줄로 전 조합을 검증할 수 있다 -
아래 `__main__`의 자체 점검이 그 두 버그를 모두 잡아낸다.

Qt도 로봇 연결도 모른다(pose_log.py/curve_log.py/tau_table.py와 같은
원칙) - 값만 받아서 문자열을 돌려줄 뿐이다. 호출자(GUI)가 그 결과를
`status_label.setText()`에 넣는다.
"""


def _fmt_p_feedback(pf):
    """P 피드백 관련 구간. pf가 비어있거나 enabled=False면 빈 문자열."""
    if not pf or not pf.get("enabled"):
        return ""

    gain = pf["gain"]            # [J1..J6] 리스트
    out = ""
    if pf.get("auto_gain"):
        # 관절마다 최종값이 다를 수 있으므로 J1~J6를 전부 나열한다.
        final_gain = pf.get("final_gain") or gain
        events = pf.get("gain_adapt_events") or []
        n_up = sum(1 for _i, _j, d, _g in events if d == "올림")
        n_down = sum(1 for _i, _j, d, _g in events if d == "낮춤")
        per_joint = " ".join(f"J{j+1}={final_gain[j]:.2f}" for j in range(6))
        out += (f"  [P 피드백 ON, 자동조정: 시작 k={gain[0]:.2f}(전 관절 동일) "
                f"→ 최종 {per_joint} (올림 {n_up}회/낮춤 {n_down}회)]")
    else:
        out += f"  [P 피드백 ON, k={gain[0]:.2f}(전 관절 동일)]"

    if pf.get("softstart_cycles"):
        out += f"  [소프트스타트 {pf['softstart_cycles']}사이클]"

    # 진동 억제(§29.15) - 중립값이면 표시하지 않는다(꺼진 것과 같으므로).
    deadband = pf.get("deadband_deg", 0.0)
    if deadband > 0.0:
        out += f"  [데드밴드 {deadband:.2f}도, {pf.get('deadband_hits', 0)}회 건너뜀]"
    alpha = pf.get("smooth_alpha", 1.0)
    if alpha < 1.0:
        out += f"  [평활화 α={alpha:.2f}]"
    kd = pf.get("kd", 0.0)
    if kd > 0.0:
        out += f"  [D항 Kd={kd:.3f}]"

    ff_joints = pf.get("error_model_joints") or []
    if ff_joints:
        names = ",".join(f"J{j+1}" for j in ff_joints)
        mean_ff = pf.get("error_model_ff_mean_deg")
        mean_txt = f", 평균 {mean_ff:.3f}도" if mean_ff is not None else ""
        out += f"  [오차모델 피드포워드: {names}{mean_txt}]"

    clamp_n = pf.get("clamp_count", 0)
    if clamp_n > 0:
        out += (f"  ⚠️ 안전클램프 {clamp_n}회 발동 - 게인이 너무 셀 수 있습니다"
                f"(k 낮춰보세요).")
    return out


def _fmt_stream(st, tcp_speed_mms, absorb_target_sec=None):
    """스트리밍 실행 구간(타이밍/사이클 통계). st가 없으면 빈 문자열."""
    if not st:
        return ""

    out = (f"  [명령 전송 {st['dispatch_done_t']:.1f}s / 로봇 완료 {st['total_t']:.1f}s "
           f"→ 밀림 {st['lag']:.1f}s]")
    if st["lag"] > 0.5:
        out += ("  ⚠️ 로봇이 명령을 못 따라갑니다. STREAM_TCP_SPEED_MMS를 낮추세요 "
                f"(현재 {tcp_speed_mms}mm/s).")

    cyc_mean_ms = st.get("cycle_mean", 0.0) * 1000
    cs = st.get("cycle_std", 0.0) * 1000
    tp = st.get("target_period", 0.0) * 1000
    late = st.get("late_ratio", 0.0)
    cf = (st.get("calib_floor") or 0.0) * 1000

    mm_label = {"none": "명령값만(부드러움)", "full": "매번 실측(거침)"}
    mode_txt = mm_label.get(st.get("measure_mode"), "")

    # 적응형 재보정이 있었으면 최초 계산값과 최종값을 같이 보여준다 - 안 그러면
    # "적용주기"가 처음 계산된 값에서 왜 달라졌는지 알 수 없다.
    adapt_events = st.get("adapt_events") or []
    if adapt_events:
        calib_p_ms = st.get("calib_period", tp / 1000.0) * 1000
        n_up = sum(1 for _i, k, _p in adapt_events if k == "상향")
        n_down = sum(1 for _i, k, _p in adapt_events if k == "복원")
        adapt_txt = (f" [적응형 재보정: 최초 {calib_p_ms:.0f}ms → 최종 {tp:.0f}ms "
                     f"(상향 {n_up}회/복원 {n_down}회)]")
    else:
        adapt_txt = ""

    out += (f"  [{mode_txt} · 자동보정 사이클바닥 {cf:.0f}ms → 적용주기 {tp:.0f}ms "
            f"/ 실측 {cyc_mean_ms:.0f}±{cs:.0f}ms → TCP 약 {st.get('est_speed', 0.0):.0f}mm/s "
            f"· 주기미달 {late*100:.0f}%]" + adapt_txt)

    if absorb_target_sec:
        # 이 실행이 기본(자동보정) 대신 스파이크 흡수 설정으로 돈다는 걸
        # 상태줄/스크린샷만 봐도 알 수 있게 남긴다(§25).
        out += f"  🔧 [스파이크 흡수 적용 - 목표주기 {absorb_target_sec*1000:.0f}ms]"
    if late > 0.2:
        out += ("  ⚠️ 자동보정 이후에도 주기를 못 따라갑니다 - USB/시리얼 상태나 "
                "다른 프로그램의 부하를 확인하세요 (STREAM_TCP_SPEED_MMS를 낮추면 완화됩니다).")
    return out


def build_execution_status(offset_correction_on, measure_mode,
                           p_feedback=None, stream=None,
                           tcp_speed_mms=0, absorb_target_sec=None,
                           lag_compensation_joints=None):
    """곡선 실행 완료 후 상태줄에 넣을 문자열을 만든다.

    offset_correction_on : bool - 관절오프셋보정 ON/OFF
    measure_mode         : "none" | "full" - 측정 모드
    p_feedback           : dict 또는 None. enabled/gain/auto_gain/final_gain/
                           gain_adapt_events/softstart_cycles/deadband_deg/
                           deadband_hits/smooth_alpha/clamp_count
                           (P 피드백은 stream 모드에서만 실제로 동작하므로,
                            queue 모드면 호출자가 None을 넘겨 표시를 생략한다)
    stream               : dict 또는 None(=queue 모드). dispatch_done_t/total_t/
                           lag/cycle_mean/cycle_std/target_period/late_ratio/
                           calib_floor/calib_period/adapt_events/est_speed/
                           measure_mode
    lag_compensation_joints : [10차, §43] 자세지연보정이 켜져 있으면 적용된
                           관절 목록(물리번호, 예 [2,3,4]), 꺼져 있으면 None.
                           오차모델 FF와 완전히 별개 표시줄 - 두 기능이
                           동시에 켜져도 각자 자기 관절만 보여준다.
    """
    msg = "✅ 곡선 경로 실행 완료."
    msg += "  [관절오프셋보정 ON]" if offset_correction_on else "  [관절오프셋보정 OFF]"
    msg += _fmt_p_feedback(p_feedback)
    if lag_compensation_joints:
        names = ",".join(f"J{j}" for j in sorted(lag_compensation_joints))
        msg += f"  [자세지연보정: {names}]"
    msg += _fmt_stream(stream, tcp_speed_mms, absorb_target_sec)
    if measure_mode == "none":
        msg += "  (실측 없음 - 결과 그래프는 생략됩니다)"
    return msg


if __name__ == "__main__":
    # 로봇 없이 도는 자체 점검 - 위 docstring에 적은 실제 버그 2건을 모두 잡는다.
    print("자체 점검 시작...")

    base = dict(offset_correction_on=True, measure_mode="full", tcp_speed_mms=35)

    # 1) 최소 조합 - P 피드백 OFF, queue 모드
    m = build_execution_status(offset_correction_on=False, measure_mode="none")
    assert "관절오프셋보정 OFF" in m and "실측 없음" in m
    assert "P 피드백" not in m
    print("  ✅ P 피드백 OFF / queue / 실측없음")

    # 2) P 피드백 고정게인 - §29.14 회귀검증(gain이 리스트여도 터지지 않아야)
    pf = dict(enabled=True, gain=[0.8]*6, auto_gain=False)
    m = build_execution_status(p_feedback=pf, **base)
    assert "[P 피드백 ON, k=0.80(전 관절 동일)]" in m
    print("  ✅ 고정게인 표시")

    # 3) 자동조정 - §29.6 회귀검증(존재하지 않는 이름 참조 시 NameError로 터짐)
    pf = dict(enabled=True, gain=[0.3]*6, auto_gain=True,
              final_gain=[0.7, 0.5, 0.5, 0.6, 0.7, 0.7],
              gain_adapt_events=[(1, 1, "올림", 0.35), (2, 2, "낮춤", 0.25)])
    m = build_execution_status(p_feedback=pf, **base)
    assert "J1=0.70" in m and "J4=0.60" in m
    assert "올림 1회/낮춤 1회" in m
    print("  ✅ 자동조정 관절별 최종게인 표시")

    # 4) final_gain 누락 시 gain으로 폴백(실행 중 예외로 stats를 못 받은 경우)
    pf = dict(enabled=True, gain=[0.8]*6, auto_gain=True)
    m = build_execution_status(p_feedback=pf, **base)
    assert "J6=0.80" in m
    print("  ✅ final_gain 누락 시 폴백")

    # 5) 진동 억제 옵션 - 중립값이면 표시 안 함
    pf = dict(enabled=True, gain=[0.8]*6, auto_gain=False,
              deadband_deg=0.0, smooth_alpha=1.0)
    m = build_execution_status(p_feedback=pf, **base)
    assert "데드밴드" not in m and "평활화" not in m
    pf.update(deadband_deg=0.2, deadband_hits=137, smooth_alpha=0.5,
              softstart_cycles=40, clamp_count=3)
    m = build_execution_status(p_feedback=pf, **base)
    assert "[데드밴드 0.20도, 137회 건너뜀]" in m
    assert "[평활화 α=0.50]" in m
    assert "[소프트스타트 40사이클]" in m
    assert "안전클램프 3회" in m
    assert "D항" not in m, "Kd=0(기본)인데 D항이 표시됨"
    print("  ✅ 진동억제/소프트스타트/클램프 표시 (중립값이면 생략)")

    # 5b) D항(§29.20) - 0.0이면 표시 안 함, 켜면 표시
    pf.update(kd=0.05)
    m = build_execution_status(p_feedback=pf, **base)
    assert "[D항 Kd=0.050]" in m
    print("  ✅ D항 표시 (0.0이면 생략)")

    # 5c) [§30.7] 오차모델 피드포워드 표시 - 관절 없으면 생략
    pf.update(error_model_joints=[0, 3, 5], error_model_ff_mean_deg=0.214)
    m = build_execution_status(p_feedback=pf, **base)
    assert "[오차모델 피드포워드: J1,J4,J6, 평균 0.214도]" in m
    print("  ✅ 오차모델 피드포워드 표시 (관절 없으면 생략)")

    # 5d) [10차, §43] 자세지연보정 표시 - 오차모델 FF와 별개 표시줄, 동시에도 공존
    m = build_execution_status(lag_compensation_joints=[2, 3, 4], **base)
    assert "[자세지연보정: J2,J3,J4]" in m
    m = build_execution_status(lag_compensation_joints=None, **base)
    assert "자세지연보정" not in m
    m = build_execution_status(p_feedback=pf, lag_compensation_joints=[2, 3, 4], **base)
    assert "[오차모델 피드포워드: J1,J4,J6, 평균 0.214도]" in m
    assert "[자세지연보정: J2,J3,J4]" in m
    print("  ✅ 자세지연보정 표시 (없으면 생략, 오차모델 FF와 동시 표시 가능)")

    # 6) 스트리밍 통계 + 경고 문턱
    st = dict(dispatch_done_t=25.5, total_t=25.8, lag=0.3, cycle_mean=0.048,
              cycle_std=0.0, target_period=0.048, late_ratio=0.0,
              calib_floor=0.030, est_speed=35.0, measure_mode="full")
    m = build_execution_status(stream=st, absorb_target_sec=0.048, **base)
    assert "밀림 0.3s" in m and "주기미달 0%" in m and "스파이크 흡수" in m
    assert "⚠️" not in m, "경고 문턱 아래인데 경고가 붙었다"
    st.update(lag=0.9, late_ratio=0.35)
    m = build_execution_status(stream=st, **base)
    assert "명령을 못 따라갑니다" in m and "주기를 못 따라갑니다" in m
    print("  ✅ 스트리밍 통계 + 경고 문턱(밀림 0.5s / 미달 20%)")

    # 7) 적응형 재보정 이력
    st.update(lag=0.3, late_ratio=0.0, calib_period=0.033,
              adapt_events=[(5, "상향", 0.04), (9, "복원", 0.033)])
    m = build_execution_status(stream=st, **base)
    assert "적응형 재보정: 최초 33ms → 최종 48ms (상향 1회/복원 1회)" in m
    print("  ✅ 적응형 재보정 이력 표시")

    print("\n모든 자체 점검 통과.")
