# -*- coding: utf-8 -*-
"""
mycobot_cycle_log.py
======================
[9차 세션, 전송 리듬 조사] "전송이 빠름-빠름-빠름-느림으로 반복되는 것
같다"는 관찰을 확정하려면 사이클별 시계열이 필요한데, `run_streaming_dispatch`는
그걸 평균/표준편차로 뭉개고 원본은 버렸다. 여기서는 그 원본
(`stats["cycle_times"]`)을 실행할 때마다 자동으로 남긴다 - `mycobot_pose_log.py`
때와 같은 이유다: 사람이 "지금 이거 리듬이 이상한데" 싶을 때마다 따로
챙기지 않아도, 이미 쌓여있는 로그에서 나중에 골라 분석할 수 있게.

Qt도 로봇 연결도 몰라도 되는 순수 함수 모듈이다(pose_log.py/curve_log.py와
같은 원칙) - GUI가 실행마다 자동으로 기록하고, 진단 스크립트
(`mycobot_cycle_rhythm_diagnostic.py`)는 그 로그를 읽어서 분석만 한다.

로그 파일(cycle_rhythm_log.jsonl, 이 모듈과 같은 폴더)은 JSON Lines 형식.
"""

import json
import os
from datetime import datetime

CYCLE_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cycle_rhythm_log.jsonl")
# [9차 세션] curve_log.py와 같은 이유로 5 -> 20 (list_recent 비용은 limit과
# 무관 - mycobot_curve_log.py의 동일 주석 참고).
DEFAULT_LIST_LIMIT = 20


def load_log():
    """로그 파일 전체를 리스트로 읽는다. 파일이 없으면 빈 리스트."""
    if not os.path.exists(CYCLE_LOG_PATH):
        return []
    records = []
    with open(CYCLE_LOG_PATH, "r", encoding="utf-8") as f:
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
        with open(CYCLE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"⚠️ 사이클 리듬 로그 저장 실패: {e}")
        return False


def save_cycle_times(label, cycle_times, send_speed=None, extra=None):
    """사이클별 시계열(초 단위 리스트)을 로그에 남긴다.

    send_speed: 그 실행에 쓰인 STREAM_SEND_SPEED 값. 나중에 이 값을 바꿔가며
    여러 번 실행했을 때, 리듬이 send_speed와 어떻게 달라지는지 비교하려면
    각 기록에 그때 값이 뭐였는지 같이 남아있어야 한다.
    """
    ts = datetime.now().isoformat(timespec="microseconds")
    record = {
        "timestamp": ts,
        "label": label,
        "cycle_times_ms": [round(t * 1000.0, 4) for t in cycle_times],
        "n_cycles": len(cycle_times),
        "send_speed": send_speed,
    }
    if extra:
        record["extra"] = extra
    append_log(record)
    return record


def list_recent(limit=DEFAULT_LIST_LIMIT):
    """최신순으로 최대 limit개를 반환한다."""
    records = load_log()
    return sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)[:limit]
