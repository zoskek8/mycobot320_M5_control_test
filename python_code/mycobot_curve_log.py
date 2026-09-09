# -*- coding: utf-8 -*-
"""
mycobot_curve_log.py
=====================
[8차 세션, §23.12] 자세 로그(`mycobot_pose_log.py`)는 최종 정지자세
6개 관절각만 남겨서, "예전에 그렸던 곡선 자체를 다시 불러와 수정·재검증·
재실행"하는 건 못 한다는 지적에 따라 만든 별도 모듈이다.

여기서는 `PathPoint.snapshot()`이 만드는 `(x, y, z, role, valid)` 튜플의
리스트, 즉 **경로 전체**를 기록한다 - `mycobot_path_editor.py`가 곡선을
실행할 때마다 `self.points`를 통째로 저장하고, 나중에 그 목록을 다시
`self.points`에 복원하면(각 튜플을 `PathPoint.from_snapshot()`으로
되돌리면) 그 시점의 편집 화면이 그대로 재현된다 - 점을 드래그해서
고치는 것도, 재검증도, 재실행도 전부 다시 가능해진다.

이 모듈 자체는 Qt도 PathPoint 클래스도 모른다(§17.7과 같은 원칙) -
snapshot 튜플은 그냥 JSON 직렬화 가능한 값 목록일 뿐이고, 복원은
호출자(GUI)가 `PathPoint.from_snapshot()`으로 한다.
"""

import json
import os
from datetime import datetime

CURVE_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curve_path_log.jsonl")
# [9차 세션] 5 -> 20. list_recent()는 limit과 무관하게 항상 파일 전체를
# 읽고 정렬한 뒤에 자르므로, 이 값을 올려도 성능비용은 0이다(전체 로드가
# 이미 그만큼 걸림). 유일한 제약은 사람이 목록에서 고르기 편한 정도 -
# 20이면 한 화면에서 훑어볼 수 있는 선.
DEFAULT_LIST_LIMIT = 20


def load_log():
    """로그 파일 전체를 리스트로 읽는다. 파일이 없으면 빈 리스트."""
    if not os.path.exists(CURVE_LOG_PATH):
        return []
    records = []
    with open(CURVE_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def append_log(record):
    """레코드 하나를 로그 파일 끝에 추가한다 (실패해도 예외를 던지지 않는다 -
    로그 실패가 호출자의 본래 작업(경로 실행 등)을 막으면 안 된다)."""
    try:
        with open(CURVE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"⚠️ 경로 로그 저장 실패: {e}")
        return False


def save_curve(label, points_snapshots, extra=None, dedupe=True):
    """경로(점 목록의 snapshot 튜플들)를 로그에 남긴다.

    points_snapshots: [(x, y, z, role, valid), ...] - PathPoint.snapshot()의
    반환값을 그대로 리스트로 모은 것. JSON에는 튜플이 리스트로 저장된다
    (불러올 때 다시 튜플로 바꿔서 PathPoint.from_snapshot()에 넘기면 된다).

    dedupe: [9차 세션] True면 **좌표가 완전히 같은 기존 기록을 먼저 지우고**
    새 기록을 추가한다. 같은 경로로 반복 실험할 때(설정만 바꿔가며 여러 번
    실행하는 게 이 프로젝트의 기본 작업방식이다) 불러오기 목록이 똑같은
    곡선 20개로 가득 차서 정작 다른 곡선을 못 찾는 문제가 있었다.
    "지우고 다시 추가"라 최신 실행 시각으로 목록 맨 위에 올라온다 - 단순히
    저장을 건너뛰면 옛 타임스탬프가 남아 "방금 실행했는데 목록 위에 없다"가
    되어 더 헷갈린다.
    """
    ts = datetime.now().isoformat(timespec="microseconds")
    record = {
        "timestamp": ts,
        "label": label,
        "points": [list(t) for t in points_snapshots],
        "n_points": len(points_snapshots),
    }
    if extra:
        record["extra"] = extra

    if dedupe:
        removed = _remove_records_with_points(record["points"])
        if removed:
            print(f"📎 같은 좌표의 기존 경로 기록 {removed}건을 지우고 최신으로 갱신합니다.")
    append_log(record)
    return record


def _remove_records_with_points(points):
    """좌표가 완전히 같은 기록을 로그에서 제거한다. 제거한 개수를 반환.

    로그 파일을 통째로 다시 쓰므로, 실패하면 원본을 건드리지 않고 0을 반환한다
    (로그 정리 실패가 호출자의 본래 작업을 막으면 안 된다 - append_log와 같은 원칙).
    JSONL 파일이 수십 줄 규모라 통째 재작성 비용은 무시할 수 있다.
    """
    records = load_log()
    if not records:
        return 0
    kept = [r for r in records if r.get("points") != points]
    removed = len(records) - len(kept)
    if removed == 0:
        return 0
    try:
        tmp_path = CURVE_LOG_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp_path, CURVE_LOG_PATH)   # 원자적 교체 - 중간에 죽어도 원본이 남는다
        return removed
    except Exception as e:
        print(f"⚠️ 중복 경로 정리 실패(기존 기록은 그대로 둡니다): {e}")
        return 0


def list_recent(limit=DEFAULT_LIST_LIMIT):
    """최신순으로 최대 limit개를 반환한다."""
    records = load_log()
    return sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)[:limit]


def snapshots_as_tuples(record):
    """record["points"](JSON에서 읽으면 리스트의 리스트)를 PathPoint.from_snapshot()이
    기대하는 튜플 리스트로 바꿔준다."""
    return [tuple(p) for p in record.get("points", [])]
