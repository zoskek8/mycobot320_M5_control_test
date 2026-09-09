# -*- coding: utf-8 -*-
"""
mycobot_validate_profile.py
============================
[9차 세션, 진단용 1회성 스크립트] "검증 알고리즘을 최적화하고 싶다"는
요청에 대해, **추정이 아니라 실측**으로 병목을 찾기 위한 프로파일러.
mycobot_vibration_diagnostic.py / mycobot_latency_diagnostic.py와 같은
위치의 "GUI 없는 1회 실행용 스크립트"다.

**로봇 연결이 필요 없다** - PathEditor.__init__은 self.mc = None만 하고
실제 연결은 execute_path()에서만 하므로, 검증(validate_path)과 추천
(_suggest_validated_safe)은 로봇 없이 그대로 돌릴 수 있다. Qt는
offscreen 플랫폼으로 띄우므로 화면도 안 뜬다.

재는 것 3가지:
  1) cProfile - 함수별 누적시간 (어디서 시간이 실제로 녹는가)
  2) chain.forward_kinematics() 호출 횟수 - 코드 리뷰에서 세어본
     "웨이포인트당 FK 30~40회" 추정이 맞는지 실측 확인
  3) 지점별 구간시간 - 자세결정 / 샘플링 / 메인 IK 루프 / 추천탐색

측정 대상 2가지:
  [A] 통과하는 곡선  -> validate_path()만 돈다
  [B] 실패하는 곡선  -> validate_path() 실패 후 _suggest_validated_safe()
                        까지 돈다 (여기가 진짜 시간 먹는 곳이라는 가설 검증)

사용법:
    python3 mycobot_validate_profile.py

    [1] 저장된 경로(curve_path_log.jsonl)에서 불러와 프로파일 - 권장.
        실제로 쓰던 곡선이라 가장 현실적인 숫자가 나온다.
    [2] 합성 곡선으로 프로파일 (로그가 없을 때)
    [3] 일부러 실패하는 곡선으로 추천탐색까지 프로파일
    [q] 종료

[주의] 이 스크립트는 아무것도 고치지 않는다 - 순수하게 재기만 한다.
로그 파일에도 안 쓴다(§23.14의 교훈 - 진단 도구가 로그를 오염시키면 안 된다).
"""

import os
# [중요] PyQt5를 import하기 전에 설정해야 한다 - 화면 없이 돌리기 위함.
# 실제 GUI를 띄우고 싶으면 이 줄을 주석 처리하면 된다.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import time
import cProfile
import pstats
import io

from PyQt5 import QtWidgets

import mycobot_kinematics as mk
import mycobot_curve_log as curve_log
from mycobot_curve_math import PathPoint
import mycobot_path_editor as pe


# ---------------------------------------------------------------------------
# 1. FK/IK 호출 횟수 계측 - 원본 함수를 카운터로 감싼다
# ---------------------------------------------------------------------------
# [설계 메모] 프로파일러(cProfile)는 "시간"은 잘 재지만, 호출 횟수를 함수별로
# 보려면 결국 pstats를 파싱해야 하고 모듈 경계를 넘나들면 읽기 어렵다.
# 여기서는 관심 있는 3개 함수만 직접 감싸서 명시적으로 센다 - 코드 리뷰에서
# 손으로 세어본 추정치(웨이포인트당 FK 30~40회)와 바로 대조하기 위함이다.

class CallCounter:
    """원본 함수를 감싸 호출 횟수와 누적 소요시간을 센다.

    [중요 - 실제로 겪은 함정] `mycobot_path_editor.py`와 `mycobot_curve_math.py`는
    `from mycobot_kinematics import solve_pose_ik, ...` 형태로 가져온다. 이건
    **import 시점에 각 모듈의 네임스페이스로 이름을 복사**하는 것이라,
    `mycobot_kinematics.solve_pose_ik`만 갈아끼워도 그 모듈들은 여전히 원본을
    부른다 - 계측값이 전부 0으로 나온다(처음 만들었을 때 실제로 이랬다).
    그래서 **그 이름을 가진 모든 모듈**을 찾아서 다 갈아끼운다.
    """

    def __init__(self, modules):
        # modules: 패치 대상 모듈 리스트 (이름을 가진 모듈만 실제로 패치됨)
        self.modules = modules
        self.counts = {}
        self.times = {}
        self._originals = []   # [(module, name, original), ...]

    def wrap(self, name, source_module):
        """source_module에서 원본을 가져와, 그 이름을 참조하는 모든 모듈에 씌운다."""
        original = getattr(source_module, name, None)
        if original is None:
            return
        self.counts[name] = 0
        self.times[name] = 0.0

        def wrapper(*args, **kwargs):
            self.counts[name] += 1
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.times[name] += time.perf_counter() - t0

        patched = 0
        for module in self.modules:
            # 그 모듈이 이 이름을 갖고 있고, 그게 원본과 같은 객체일 때만 교체한다
            # (우연히 같은 이름의 다른 함수를 덮어쓰지 않도록)
            if getattr(module, name, None) is original:
                self._originals.append((module, name, original))
                setattr(module, name, wrapper)
                patched += 1
        if patched == 0:
            print(f"  ⚠️ [{name}] 를 참조하는 모듈을 못 찾음 - 계측 누락 가능")

    def restore(self):
        for module, name, original in self._originals:
            setattr(module, name, original)
        self._originals.clear()

    def reset(self):
        for k in self.counts:
            self.counts[k] = 0
            self.times[k] = 0.0

    def report(self, n_waypoints=None):
        print("\n--- 호출 횟수 / 누적시간 ---")
        print(f"  {'함수':<28} {'횟수':>10} {'누적(s)':>10} {'1회평균(ms)':>13}")
        for name in self.counts:
            c = self.counts[name]
            t = self.times[name]
            avg_ms = (t / c * 1000) if c else 0.0
            print(f"  {name:<28} {c:>10,} {t:>10.3f} {avg_ms:>13.4f}")
        if n_waypoints:
            print(f"\n  웨이포인트 {n_waypoints}개 기준 1개당 호출 횟수:")
            for name in self.counts:
                print(f"    {name:<26} {self.counts[name] / n_waypoints:>8.1f}회")


# [주의] chain.forward_kinematics는 mycobot_kinematics.chain 객체의 메서드다.
# 모듈 함수가 아니라 인스턴스 메서드라서 위 CallCounter.wrap()으로는 못 감싼다
# (setattr 대상이 모듈이 아니라 객체). 별도로 처리한다.
class FKCounter:
    def __init__(self, chain_obj):
        self.chain = chain_obj
        self.original = chain_obj.forward_kinematics
        self.count = 0
        self.time = 0.0

    def install(self):
        original = self.original

        def wrapper(*args, **kwargs):
            self.count += 1
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.time += time.perf_counter() - t0

        self.chain.forward_kinematics = wrapper

    def restore(self):
        self.chain.forward_kinematics = self.original

    def reset(self):
        self.count = 0
        self.time = 0.0


# ---------------------------------------------------------------------------
# 2. 테스트용 곡선 준비
# ---------------------------------------------------------------------------

def load_curve_from_log():
    """curve_path_log.jsonl에서 최신 경로를 불러온다. 없으면 None."""
    records = curve_log.list_recent(limit=curve_log.DEFAULT_LIST_LIMIT)
    if not records:
        return None, None

    print("\n=== 저장된 경로 (최신 5개) ===")
    for i, rec in enumerate(records, start=1):
        print(f"  [{i}] {rec.get('timestamp', '?')}  [{rec.get('label', '?')}]"
              f"  ({rec.get('n_points', '?')}점)")
    choice = input("불러올 번호 (Enter=1번): ").strip() or "1"
    try:
        idx = int(choice)
    except ValueError:
        print("⚠️ 잘못된 입력 - 1번을 씁니다.")
        idx = 1
    if not (1 <= idx <= len(records)):
        print("⚠️ 범위 밖 - 1번을 씁니다.")
        idx = 1

    rec = records[idx - 1]
    pts = [PathPoint.from_snapshot(t) for t in curve_log.snapshots_as_tuples(rec)]
    return pts, rec.get("label", "?")


def synthetic_curve():
    """로그가 없을 때 쓸 합성 곡선.

    [주의] 이 좌표는 이 로봇의 작업영역 안에 있으리라 **보장하지 못한다** -
    데드존 격자와 관절한계에 따라 검증이 실패할 수도 있다. 실패하면 그건
    그것대로 [3]번(추천탐색 프로파일)과 같은 경로를 타므로 측정은 된다.
    가장 현실적인 숫자를 원하면 [1]번(실제 저장된 경로)을 쓸 것.
    """
    coords = [
        (150, 150, 450),   # anchor
        (50, 150, 420),    # ctrl
        (-50, 150, 400),   # anchor
        (-120, 150, 390),  # ctrl
        (-200, 150, 385),  # anchor
    ]
    roles = ["anchor", "ctrl", "anchor", "ctrl", "anchor"]
    return [PathPoint(x, y, z, role=r) for (x, y, z), r in zip(coords, roles)], "synthetic"


def unreachable_curve():
    """일부러 실패시키는 곡선 - 마지막 앵커를 명백히 도달 불가능한 곳에 둔다.
    validate_path가 실패하면서 _suggest_validated_safe까지 타게 만든다."""
    coords = [
        (150, 150, 450),
        (50, 150, 420),
        (-50, 150, 400),
        (-200, 150, 500),
        (-500, 150, 700),   # 팔 길이를 한참 넘어선 지점
    ]
    roles = ["anchor", "ctrl", "anchor", "ctrl", "anchor"]
    return [PathPoint(x, y, z, role=r) for (x, y, z), r in zip(coords, roles)], "unreachable"


# ---------------------------------------------------------------------------
# 3. 프로파일 실행
# ---------------------------------------------------------------------------

def profile_validate(win, points, label, top_n=30):
    win.points = [PathPoint.from_snapshot(p.snapshot()) for p in points]
    win._invalidate_after_edit()

    fk = FKCounter(mk.chain)
    # from-import로 이름을 복사해간 모듈들까지 전부 패치 대상에 넣는다 (위 주석 참고)
    import mycobot_curve_math as cmath_mod
    import mycobot_canvas_draw as cdraw_mod
    counter = CallCounter([mk, pe, cmath_mod, cdraw_mod])
    for fname in ("solve_pose_ik", "natural_pose_at", "numerical_jacobian",
                  "numerical_jacobian_6d", "check_self_collision",
                  "jacobian_condition_number", "within_joint_limits",
                  "quick_prefilter"):
        counter.wrap(fname, mk)
    fk.install()

    print(f"\n{'='*70}")
    print(f"  프로파일 시작: [{label}]  점 {len(win.points)}개")
    print(f"{'='*70}")

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    try:
        win.validate_path()
    finally:
        prof.disable()
    elapsed = time.perf_counter() - t0

    fk.restore()
    counter.restore()

    n_wp = len(getattr(win, "curve_waypoints", []) or [])
    passed = n_wp > 0

    print("\n--- 결과 요약 ---")
    print(f"  전체 소요:        {elapsed:.3f}s")
    print(f"  검증 결과:        {'✅ 통과' if passed else '❌ 실패(추천탐색까지 돌았음)'}")
    print(f"  웨이포인트 수:    {n_wp}")
    print(f"  FK 총 호출:       {fk.count:,}회  (누적 {fk.time:.3f}s, "
          f"전체의 {fk.time/max(elapsed,1e-9)*100:.1f}%)")
    if n_wp:
        print(f"  웨이포인트당 FK:  {fk.count/n_wp:.1f}회")
    print(f"  상태줄: {win.status_label.text()[:120]}")

    counter.report(n_waypoints=n_wp if n_wp else None)

    print(f"\n--- cProfile 누적시간 상위 {top_n} ---")
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    ps.print_stats(top_n)
    # 경로가 길어 읽기 힘드므로 파일경로 앞부분을 잘라낸다
    for line in s.getvalue().splitlines():
        print("  " + line)

    out = f"/tmp/validate_profile_{label.replace(' ', '_').replace(':', '_')}_{int(time.time())}.prof"
    try:
        prof.dump_stats(out)
        print(f"\n💾 프로파일 저장: {out}")
        print(f"   (자세히 보려면: python3 -m pstats {out}  또는  snakeviz {out})")
    except Exception as e:
        print(f"⚠️ 프로파일 저장 실패: {e}")

    return elapsed, fk.count, n_wp


def main():
    if not mk._grid_loaded:
        print("⚠️ 데드존 격자(.npz)가 로드되지 않았습니다.")
        print("   quick_prefilter가 무조건 통과가 되고 추천탐색은 아예 동작하지 않습니다")
        print("   (_suggest_validated_safe는 격자가 없으면 즉시 None을 반환).")
        print("   -> [3]번 추천탐색 프로파일은 의미 있는 숫자가 안 나옵니다.\n")

    # QApplication 객체는 쓰지 않지만, 위젯을 만들려면 반드시 먼저 존재해야 하고
    # 지역변수로 붙들지 않으면 가비지컬렉션돼 크래시가 난다 - 그래서 살려둔다.
    app = QtWidgets.QApplication(sys.argv)   # noqa: F841
    win = pe.PathEditor()
    print(f"✅ PathEditor 생성 완료 (로봇 연결 없음, CODE_VERSION={pe.CODE_VERSION})")
    print(f"   실행 방식: {win.mode_combo.currentData()} / "
          f"측정: {win.measure_combo.currentData()} / "
          f"웨이포인트 간격: {win._stream_step_mm():.2f}mm")

    while True:
        print("\n--- 메뉴 ---")
        choice = input(
            "  [1] 저장된 경로에서 불러와 프로파일 (권장 - 가장 현실적)\n"
            "  [2] 합성 곡선으로 프로파일\n"
            "  [3] 일부러 실패하는 곡선 - 추천탐색까지 프로파일\n"
            "  [q] 종료\n"
            "> "
        ).strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            pts, label = load_curve_from_log()
            if pts is None:
                print("⚠️ 저장된 경로가 없습니다 (curve_path_log.jsonl). "
                      "GUI에서 곡선을 한 번 실행하면 쌓입니다. [2]를 써보세요.")
                continue
            profile_validate(win, pts, f"log:{label}")
        elif choice == "2":
            pts, label = synthetic_curve()
            profile_validate(win, pts, label)
        elif choice == "3":
            pts, label = unreachable_curve()
            print("\n[안내] 이 곡선은 일부러 실패하게 만든 것입니다 - "
                  "추천탐색(_suggest_validated_safe)이 돌면서 "
                  f"최대 {pe.REGION_MAX}개 후보 검사 + 상위 {pe.FULLCHECK_MAX}개 "
                  "전체경로 재검증을 수행합니다. 시간이 좀 걸립니다.")
            profile_validate(win, pts, label)
        else:
            print("잘못된 입력입니다.")

    print("종료합니다.")


if __name__ == "__main__":
    main()
