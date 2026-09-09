# -*- coding: utf-8 -*-
"""
mycobot_pose_log.py
====================
[8차 세션, §23.7] "같은 자세로 다시 재보라"는 요청 자체가, 그 자세를
기록해두는 시스템 없이는 불가능하다는 지적에 따라 만든 공유 로그 모듈.

원래는 mycobot_vibration_diagnostic.py 안에 자체적으로 있었는데,
"실제로 진동이 발생하는 순간은 GUI(mycobot_path_editor.py)에서 곡선을
실행한 직후"라는 지적에 따라 공유 모듈로 뺐다 - GUI가 매 실행마다
자동으로 로그를 남기고, 진단 스크립트는 그 로그를 읽어서 비교/검색만
한다. 사람이 "지금 이 자세를 기록해야지"라고 따로 챙기지 않아도 된다.

Qt도 로봇 연결도 몰라도 되는 순수 함수 모듈이다(§17.7과 같은 원칙) -
GUI 쪽에서 호출할 때도, 진단 스크립트에서 호출할 때도 이 모듈은 그냥
파일 하나를 읽고 쓸 뿐이다.

로그 파일(vibration_pose_log.jsonl, 이 모듈과 같은 폴더)은 JSON Lines
형식 - 한 줄에 기록 하나. 사람이 직접 열어봐도 되고, 필요하면 지워도
된다(그러면 그냥 새로 쌓이기 시작한다).
"""

import json
import os
from datetime import datetime

POSE_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vibration_pose_log.jsonl")
SIMILAR_POSE_TOLERANCE_DEG = 3.0  # 이 이내면 "비슷한 자세"로 간주 (관절별 최대오차 기준)
# [9차 세션] 원래 이 상수가 없이 mycobot_vibration_diagnostic.py에 5가 직접
# 하드코딩돼 있었다 - curve_log.py/cycle_log.py와 기준을 맞추려고 신설했다.
# (load_log()도 limit과 무관하게 파일 전체를 읽으므로 값을 올려도 비용 없음.)
DEFAULT_LIST_LIMIT = 20


def load_log():
    """로그 파일 전체를 리스트로 읽는다. 파일이 없으면 빈 리스트."""
    if not os.path.exists(POSE_LOG_PATH):
        return []
    records = []
    with open(POSE_LOG_PATH, "r", encoding="utf-8") as f:
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
    로그 실패가 호출자의 본래 작업(실행 결과 표시 등)을 막으면 안 된다)."""
    try:
        with open(POSE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"⚠️ 자세 로그 저장 실패: {e}")
        return False


def find_similar_poses(joint_means, tolerance_deg=SIMILAR_POSE_TOLERANCE_DEG, exclude_ts=None):
    """로그에서 관절별 오차가 전부 tolerance_deg 이내인 과거 기록을 찾는다.
    (max_diff, record) 튜플의 리스트를 max_diff 오름차순으로 반환한다."""
    hits = []
    for rec in load_log():
        if exclude_ts is not None and rec.get("timestamp") == exclude_ts:
            continue
        past = rec.get("joint_means")
        if not past or len(past) != len(joint_means):
            continue
        diffs = [abs(a - b) for a, b in zip(joint_means, past)]
        if max(diffs) <= tolerance_deg:
            hits.append((max(diffs), rec))
    hits.sort(key=lambda x: x[0])
    return hits


def log_pose(label, joint_means, joint_std=None, joint_ptp=None, extra=None,
             tolerance_deg=SIMILAR_POSE_TOLERANCE_DEG, announce=True):
    """자세 하나를 로그에 기록하고, 그 자리에서 바로 비슷한 과거 자세를
    찾아 반환한다 (record, similar_list). announce=True면 콘솔에 요약도 찍는다.

    joint_std/joint_ptp/extra는 선택 - 진동 진단처럼 관절별 통계가 있으면
    같이 남기고, GUI처럼 순간 자세 하나만 있으면 None으로 둬도 된다.
    """
    ts = datetime.now().isoformat(timespec="microseconds")
    record = {
        "timestamp": ts,
        "label": label,
        "joint_means": [float(v) for v in joint_means],
    }
    if joint_std is not None:
        record["joint_std"] = [float(v) for v in joint_std]
    if joint_ptp is not None:
        record["joint_ptp"] = [float(v) for v in joint_ptp]
        worst = max(range(len(joint_ptp)), key=lambda i: joint_ptp[i])
        record["worst_joint"] = worst + 1
        record["worst_ptp"] = float(joint_ptp[worst])
    if extra:
        record["extra"] = extra

    # [주의] 여기서는 exclude_ts를 넘기지 않는다 - 이 함수는 항상 검색을
    # append_log()보다 먼저 하므로, 지금 만드는 이 record는 아직 로그
    # 파일에 없다(자기 자신과 매칭될 위험이 애초에 없다). 예전엔
    # exclude_ts=ts를 넘겼었는데, timespec="seconds"였을 때 같은 초에
    # 연달아 호출되면(자동 테스트, 빠른 연속 측정 등) 방금 막 쌓인
    # "진짜 비슷한 과거 기록"까지 타임스탬프가 우연히 같아서 제외되는
    # 버그가 있었다 - 마이크로초 단위로 올리고 exclude_ts 자체를 뺐다.
    similar = find_similar_poses(joint_means, tolerance_deg=tolerance_deg)

    if announce:
        if similar:
            print(f"📎 자세 로그: 비슷한 과거 자세 {len(similar)}건 발견"
                  f" (관절별 오차 {tolerance_deg}도 이내):")
            for max_diff, rec in similar[:5]:
                extra_note = ""
                if "worst_joint" in rec:
                    extra_note = (f" | 그때 최대흔들림: J{rec['worst_joint']}"
                                  f" P2P {rec['worst_ptp']:.3f}도")
                print(f"   - {rec['timestamp']} [{rec['label']}]"
                      f" 최대관절오차 {max_diff:.2f}도{extra_note}")
        else:
            print("📎 자세 로그: 비슷한 과거 자세 없음 (처음 보는 자세이거나 로그가 비어있음)")

    append_log(record)
    if announce:
        print(f"💾 자세 로그에 기록됨: {POSE_LOG_PATH}")

    return record, similar
