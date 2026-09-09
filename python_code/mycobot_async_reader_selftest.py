# -*- coding: utf-8 -*-
"""
mycobot_async_reader_selftest.py
===================================
[10차 세션, §45] `mycobot_kinematics.py`의 `AsyncAngleReader`(파이프라인
읽기, §6-d)와 그걸 위해 추가한 `_serial_write_lock`(§45)에 대한 회귀
테스트다. 로봇 연결도, ikpy/pybullet/pymycobot도 필요 없다 - 가짜 시리얼
포트로 전부 검증한다.

**왜 이 파일이 따로 있는가**: `mycobot_kinematics.py`는 최상단에서
`from ikpy.chain import Chain` / `import pybullet as p` /
`from pymycobot import MyCobot320`를 하기 때문에, 이 스레딩 로직만
가볍게 확인하려 해도 그 무거운 의존성을 전부 깔아야 한다. 그래서 실제
소스에서 필요한 부분만 잘라내(`exec`) 순수 스레딩 로직만 검증한다 -
`mycobot_ab_control_adjust.py --selftest`와 같은 원칙(진짜 소스를 그대로
쓰되 무거운 의존성만 피함).

**검증하는 것**:
  1. AsyncAngleReader의 기본 동작(시작 전 None, 시작 후 비차단 조회,
     계속 갱신, stop() 후 정지, 중복 start() 방지) - §45 개발 중 이미
     확인했던 것.
  2. **가장 중요한 것 - 동시성 안전성**: 메인 스레드가 fast_send_angles로
     계속 명령을 보내는 동안, 배경 스레드가 AsyncAngleReader로 계속
     읽어도 **두 write() 호출이 절대 시간적으로 안 겹치는지**(바이트가
     뒤섞여 프레임이 깨질 수 있는 유일한 지점, §45 주석 참고). 이게 이
     기능 전체의 안전성을 좌우하는 시험이다.

사용법:
    python3 mycobot_async_reader_selftest.py
"""

import sys
import time
import struct
import threading


def _load_kinematics_symbols():
    """mycobot_kinematics.py에서 필요한 심볼만 실행해 뽑아낸다 -
    ikpy/pybullet/pymycobot 없이도 동작한다."""
    with open("mycobot_kinematics.py", encoding="utf-8") as f:
        src = f.read()
    start = src.index("def build_send_angles_frame")
    end = src.index("\n\n\n", src.index("class AsyncAngleReader"))
    ns = {
        "struct": struct, "threading": threading, "time": time,
        "JOINT_LIMITS_DEG": [(-165, 165)] * 6,
    }
    exec(src[start:end], ns)
    return ns


class FakeSerialPort:
    """진짜 시리얼 포트를 흉내낸다. write() 호출 구간(시작~끝 시각)을
    전부 기록해서, 나중에 두 write()가 겹쳤는지(=락이 안 먹었다는 증거)
    검사한다. GET_ANGLES 요청(명령코드 0x20)에는 ~30ms 뒤 정상 응답을
    준비해 펌웨어 왕복을 흉내낸다."""

    def __init__(self, write_delay_sec=0.0005, response_delay_sec=0.03):
        self.write_intervals = []
        self._write_delay = write_delay_sec
        self._response_delay = response_delay_sec
        self._response_ready_at = None
        self._response_bytes = b""
        self.n_angle_reads = 0

    def write(self, data):
        t0 = time.perf_counter()
        time.sleep(self._write_delay)   # 실제 전송에 걸리는 시간 흉내
        if len(data) >= 4 and data[3] == 0x20:   # GET_ANGLES 요청
            self.n_angle_reads += 1
            angles = [10.0 + self.n_angle_reads] * 6
            body = b"".join(struct.pack(">h", int(round(a * 100))) for a in angles)
            self._response_bytes = bytes([0xFE, 0xFE, 0x0E, 0x20]) + body + bytes([0xFA])
            self._response_ready_at = time.perf_counter() + self._response_delay
        t1 = time.perf_counter()
        self.write_intervals.append((t0, t1, bytes(data[:4])))

    def reset_input_buffer(self):
        pass

    @property
    def in_waiting(self):
        if self._response_ready_at and time.perf_counter() >= self._response_ready_at:
            return len(self._response_bytes)
        return 0

    def read(self, n):
        b = self._response_bytes[:n]
        self._response_bytes = self._response_bytes[n:]
        if not self._response_bytes:
            self._response_ready_at = None
        return b


class FakeMC:
    def __init__(self, **kwargs):
        self._serial_port = FakeSerialPort(**kwargs)


def check_no_overlaps(intervals):
    """시간순 정렬 후 인접 구간이 겹치는 쌍이 있는지 확인. 겹친 쌍 목록 반환."""
    ordered = sorted(intervals, key=lambda x: x[0])
    overlaps = []
    for i in range(len(ordered) - 1):
        end_i = ordered[i][1]
        start_next = ordered[i + 1][0]
        if start_next < end_i:
            overlaps.append((ordered[i], ordered[i + 1]))
    return overlaps


def main():
    print("자체 점검 시작...\n")
    ns = _load_kinematics_symbols()
    fast_send_angles = ns["fast_send_angles"]
    AsyncAngleReader = ns["AsyncAngleReader"]

    # 1) 기본 동작 - 시작 전/직후/이후/정지/중복시작
    print("--- 1) 기본 동작 ---")
    mc = FakeMC(write_delay_sec=0.0001, response_delay_sec=0.03)
    reader = AsyncAngleReader(mc)

    a, t = reader.get_latest()
    assert a is None and t is None
    print("  ✅ start() 전: (None, None)")

    reader.start()
    time.sleep(0.001)
    a0, _ = reader.get_latest()
    assert a0 is None, "30ms도 안 지났는데 값이 있으면 이상함"
    print("  ✅ start() 직후(1ms): 아직 값 없음")

    time.sleep(0.05)
    t_before = time.perf_counter()
    a1, t1 = reader.get_latest()
    elapsed = time.perf_counter() - t_before
    assert a1 is not None and elapsed < 0.001, f"get_latest()가 블로킹함({elapsed*1000:.3f}ms)"
    print(f"  ✅ 50ms 후: 값 있음(angles={a1}), 조회 비차단(<1ms)")

    time.sleep(0.1)
    a2, t2 = reader.get_latest()
    assert a2[0] > a1[0] and t2 > t1, "시간이 지났는데 값이 안 갱신됨"
    print(f"  ✅ 추가 100ms 후 계속 갱신됨 ({a1[0]:.0f} -> {a2[0]:.0f})")

    reader.stop()
    a3, _ = reader.get_latest()
    time.sleep(0.1)
    a4, _ = reader.get_latest()
    assert a3 == a4, "stop() 후에도 계속 갱신되고 있음"
    print("  ✅ stop() 후 갱신 정지")

    th1 = None
    reader2 = AsyncAngleReader(mc)
    reader2.start()
    th1 = reader2._thread
    reader2.start()
    assert reader2._thread is th1, "중복 start()가 새 스레드를 만듦"
    reader2.stop()
    print("  ✅ 중복 start() 방지\n")

    # 2) [가장 중요] 동시성 안전성 - 메인 스레드 전송 + 배경 스레드 읽기 경쟁
    print("--- 2) 동시성 안전성(write-write 겹침 검사) ---")
    mc2 = FakeMC(write_delay_sec=0.0005, response_delay_sec=0.03)
    reader3 = AsyncAngleReader(mc2)
    reader3.start()

    t_end = time.perf_counter() + 0.15
    n_sends = 0
    while time.perf_counter() < t_end:
        ok = fast_send_angles(mc2, [0.0] * 6, 50)
        assert ok
        n_sends += 1
        time.sleep(0.01)

    reader3.stop()

    intervals = mc2._serial_port.write_intervals
    overlaps = check_no_overlaps(intervals)
    print(f"  전송 {n_sends}회, 배경 읽기요청 {mc2._serial_port.n_angle_reads}회, "
          f"총 write() {len(intervals)}회")
    if overlaps:
        for a, b in overlaps[:3]:
            print(f"    ⚠️ 겹침: {a} vs {b}")
    assert not overlaps, f"write() 겹침 {len(overlaps)}건 발견 - 락이 깨졌다"
    print(f"  ✅ 150ms 경쟁 동안 write() 겹침 0건 - _serial_write_lock 정상 작동")

    a_final, _ = reader3.get_latest()
    assert a_final is not None and a_final[0] > 10.0
    print(f"  ✅ 경쟁 중에도 값이 정상 갱신됨(최종 {a_final})\n")

    # 3) 더 공격적인 경쟁(전송 간격을 훨씬 좁혀 락 경합을 늘림) - 그래도 안 깨지는지
    print("--- 3) 더 촘촘한 경쟁(전송 간격 2ms)에서도 안전한지 ---")
    mc3 = FakeMC(write_delay_sec=0.0005, response_delay_sec=0.03)
    reader4 = AsyncAngleReader(mc3)
    reader4.start()
    t_end3 = time.perf_counter() + 0.1
    n_sends3 = 0
    while time.perf_counter() < t_end3:
        fast_send_angles(mc3, [0.0] * 6, 50)
        n_sends3 += 1
        time.sleep(0.002)
    reader4.stop()
    overlaps3 = check_no_overlaps(mc3._serial_port.write_intervals)
    print(f"  전송 {n_sends3}회, write() {len(mc3._serial_port.write_intervals)}회, "
          f"겹침 {len(overlaps3)}건")
    assert not overlaps3
    print("  ✅ 촘촘한 경쟁에서도 겹침 0건\n")

    print("모든 자체 점검 통과.")


if __name__ == "__main__":
    main()
