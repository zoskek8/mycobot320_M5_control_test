# -*- coding: utf-8 -*-
"""
mycobot_canvas_draw.py
========================
[8차 세션, §23.13] `_draw_view_2d` / `_draw_view_3d`와 그 보조 함수들
(배경 슬라이스, 안전여유 색상 계산, 곡선 샘플링, RDP 단순화)을
`mycobot_path_editor.py`에서 분리했다. §17.7/§22와 정확히 같은 이유·
같은 패턴이다:

  - 이 함수들은 matplotlib 축(ax)과 데이터(점 목록, 슬라이더 값 등)만
    받아서 그리기만 한다 - `self.mc`(로봇 핸들)도 안 건드리고, Qt 위젯도
    `canvas.draw_idle()` 말고는 안 건드린다(그것도 인자로 받은 canvas).
  - 안전여유(margin) 색상 계산 관련 함수들(`build_safe_margin`,
    `margin_at`, `margin_color`, `contrast_color`, `margin_marker_color`)은
    원래도 모듈 레벨 함수였다 - self를 아예 쓰지 않았다.
  - PathEditor에는 이 모듈을 부르는 코드만 남는다(직접 호출 - 얇은
    위임 메서드조차 필요 없을 만큼 단순해졌다. `_redraw_all`이 유일한
    호출부라서).

이 모듈은 mycobot_kinematics의 데드존 격자 전역값(_reachable_grid 등)을
가져다 쓴다 - PathEditor가 이미 그렇게 쓰고 있던 것과 동일하다.
"""

import numpy as np
import matplotlib
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from scipy.ndimage import distance_transform_edt

from mycobot_kinematics import (
    _reachable_grid, _collision_only_grid, _x_min, _y_min, _z_min, _step,
    _nx, _ny, _nz, _grid_loaded, DESK_SAFETY_MARGIN_MM,
)
from mycobot_curve_math import bezier_chain_eval

# ---------------------------------------------------------------------------
# 안전여유(margin) 스칼라장 - 등고선 표시용
# ---------------------------------------------------------------------------
# [주의] 데드존 격자(_reachable_grid / _collision_only_grid)는 True/False 뿐이라
#   그 자체에는 '농도'가 없다. 불리언에 컬러맵을 씌워봐야 여전히 두 가지 색이다.
#   따라서 등고선을 그리려면 연속적인 스칼라를 먼저 만들어야 한다.
#   여기서는 거리변환(EDT)으로 "각 안전 셀이 가장 가까운 위험 셀에서 몇 mm 떨어져
#   있는가" = 안전여유를 계산해 그 값을 농도/등고선으로 쓴다.
#   여유가 클수록 진한 색 = 데드존 경계에서 멀어 여유 있는 지점이라는 뜻.
#
# [한계 - 반드시 인지할 것] 이 여유는 '격자상 도달가능/자가충돌' 기준일 뿐이다.
#   실제 경로검증 실패의 상당수는 특이점(조건수)·자세(orientation) 때문에 생기므로,
#   진한 색이라고 해서 검증 통과가 보장되지는 않는다. 어디까지나 '경계에서
#   얼마나 여유 있는가'를 보여주는 보조 지표다.
MARGIN_LEVEL_FRACS = (0.15, 0.3, 0.5, 0.7, 0.9)   # 등고선을 그릴 여유 비율


def build_safe_margin():
    """안전 셀마다 '가장 가까운 위험 셀까지의 거리(mm)'를 계산. 시작 시 1회만 수행."""
    if not _grid_loaded or _reachable_grid is None or _collision_only_grid is None:
        return None, 1.0
    safe = np.asarray(_reachable_grid, dtype=bool) & ~np.asarray(_collision_only_grid, dtype=bool)
    z_coords = _z_min + np.arange(_nz) * _step
    safe[:, :, z_coords < DESK_SAFETY_MARGIN_MM] = False      # 책상 아래는 안전이 아님
    margin = distance_transform_edt(safe, sampling=_step)     # sampling=격자간격 -> 단위가 mm
    inside = margin[margin > 0]
    vmax = float(np.percentile(inside, 95)) if inside.size else 1.0
    return margin.astype(np.float32), max(vmax, 1.0)


SAFE_MARGIN_MM, MARGIN_VMAX = build_safe_margin()

try:
    MARGIN_CMAP = matplotlib.colormaps["YlGn"]          # matplotlib >= 3.6
except Exception:
    MARGIN_CMAP = matplotlib.cm.get_cmap("YlGn")        # 구버전 폴백


def margin_at(x, y, z):
    """임의의 3D 좌표에서의 안전여유(mm). 격자 밖이거나 위험지대면 0."""
    if SAFE_MARGIN_MM is None:
        return 0.0
    ix = int(round((x - _x_min) / _step))
    iy = int(round((y - _y_min) / _step))
    iz = int(round((z - _z_min) / _step))
    if not (0 <= ix < _nx and 0 <= iy < _ny and 0 <= iz < _nz):
        return 0.0
    return float(SAFE_MARGIN_MM[ix, iy, iz])


def margin_color(margin_mm):
    """안전여유를 등고선과 '같은' 컬러맵 색으로 변환 (여유 클수록 진함).
    배경(등고선) 전용. 그 위에 얹는 점 색으로는 margin_marker_color를 쓸 것."""
    return MARGIN_CMAP(float(np.clip(margin_mm / MARGIN_VMAX, 0.0, 1.0)))


def _rel_luminance(rgb):
    """WCAG 상대휘도. 대비비 계산용."""
    c = []
    for v in rgb[:3]:
        v = float(v)
        c.append(v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast_ratio(rgb1, rgb2):
    l1, l2 = _rel_luminance(rgb1), _rel_luminance(rgb2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def contrast_color(bg_rgba, target_ratio=4.5):
    """주어진 배경색에서 '가장 먼' 색을 만든다.

    1) 색상(hue)을 180도 돌려 보색을 잡는다 (채도 최대).
    2) 그런데 보색만으로는 부족하다 - 예를 들어 진초록 배경(hue 140°)의 보색인
       분홍(hue 320°)은 색상은 정반대지만 **밝기가 거의 같아** 실제로는 잘 안 보인다
       (실측 대비비 1.32:1). 그래서 보색을 흰색/검정 쪽으로 섞어가며 WCAG 대비비가
       target_ratio 이상이 되는 지점을 이진탐색으로 찾는다.
    3) 중간 밝기 배경(YlGn 중간의 연두)은 흰색으로 섞어도 4.5:1에 못 미치고
       검정으로 섞어야 도달한다(그 반대인 경우도 있다). 그래서 **양방향을 다 시도**해
       목표를 만족하는 쪽을, 둘 다 못 미치면 그나마 대비가 큰 쪽을 택한다.
    -> 색상 정보(여유 크기)는 유지하면서 가시성은 항상 최대한 확보된다.
    """
    import colorsys
    r, g, b = float(bg_rgba[0]), float(bg_rgba[1]), float(bg_rgba[2])
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    base = np.array(colorsys.hsv_to_rgb((h + 0.5) % 1.0, 1.0, 1.0))   # 보색(채도/명도 최대)
    bg = (r, g, b)

    if _contrast_ratio(bg, base) >= target_ratio:
        return (float(base[0]), float(base[1]), float(base[2]), 1.0)

    best = None       # (달성 대비비, 색)
    for anchor in (np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])):
        # anchor를 100% 섞었을 때조차 목표에 못 미치면 이 방향은 최대치만 기록
        cand_full = anchor
        ratio_full = _contrast_ratio(bg, cand_full)
        if ratio_full < target_ratio:
            if best is None or ratio_full > best[0]:
                best = (ratio_full, cand_full)
            continue
        lo, hi = 0.0, 1.0
        found = cand_full
        for _ in range(24):        # 목표를 만족하는 '가장 덜 섞은'(=가장 유채색인) 색
            mid = (lo + hi) / 2.0
            cand = base * (1 - mid) + anchor * mid
            if _contrast_ratio(bg, cand) >= target_ratio:
                found = cand
                hi = mid
            else:
                lo = mid
        rr = _contrast_ratio(bg, found)
        if best is None or (rr >= target_ratio and best[0] < target_ratio) or \
           (rr >= target_ratio and best[0] >= target_ratio and rr > best[0]):
            best = (rr, found)

    out = best[1]
    return (float(out[0]), float(out[1]), float(out[2]), 1.0)


def margin_marker_color(margin_mm):
    """점 마커용 색. 배경 등고선과 같은 컬러맵을 쓰면 점이 배경에 묻혀 안 보이므로,
    그 색의 보색을 쓴다. 여유(margin)에 따라 색이 계속 달라지므로 '여유 크기 정보'는
    그대로 유지하면서 가시성만 확보된다."""
    return contrast_color(margin_color(margin_mm))


# ---------------------------------------------------------------------------
# 배경 슬라이스 (데드존 격자를 현재 슬라이더 위치에서 자른 2D 단면)
# ---------------------------------------------------------------------------

def bg_slice_xy(z_value):
    if not _grid_loaded or _reachable_grid is None:
        return None
    if z_value < DESK_SAFETY_MARGIN_MM:
        return None
    iz = int(round((z_value - _z_min) / _step))
    iz = max(0, min(_nz - 1, iz))
    safe = _reachable_grid[:, :, iz] & (~_collision_only_grid[:, :, iz])
    collide = _collision_only_grid[:, :, iz]
    marg = SAFE_MARGIN_MM[:, :, iz] if SAFE_MARGIN_MM is not None else np.zeros(safe.shape)
    return safe, collide, marg


def bg_slice_yz(x_value):
    if not _grid_loaded or _reachable_grid is None:
        return None
    ix = int(round((x_value - _x_min) / _step))
    ix = max(0, min(_nx - 1, ix))
    safe = _reachable_grid[ix, :, :] & (~_collision_only_grid[ix, :, :])
    collide = _collision_only_grid[ix, :, :]
    marg = SAFE_MARGIN_MM[ix, :, :] if SAFE_MARGIN_MM is not None else np.zeros(safe.shape)
    return safe, collide, marg


def bg_slice_xz(y_value):
    if not _grid_loaded or _reachable_grid is None:
        return None
    iy = int(round((y_value - _y_min) / _step))
    iy = max(0, min(_ny - 1, iy))
    safe = _reachable_grid[:, iy, :] & (~_collision_only_grid[:, iy, :])
    collide = _collision_only_grid[:, iy, :]
    marg = SAFE_MARGIN_MM[:, iy, :] if SAFE_MARGIN_MM is not None else np.zeros(safe.shape)
    return safe, collide, marg


# ---------------------------------------------------------------------------
# 부드러운 곡선(스플라인) - 화면 미리보기 전용, 실제 로봇 실행과는 별개
# ---------------------------------------------------------------------------

def sample_curve(points, n_samples=200):
    """점들로 만든 2차 베지어 체인을 샘플링해서 (t, x, y, z) 배열로 반환.
    짝수번째 점(1,3,5...번, 0-index로는 0,2,4)은 곡선이 실제로 지나는 앵커,
    홀수번째 점(2,4,6...번)은 그 사이를 당기는 제어점 - 곡선이 지나지 않는다."""
    n = len(points)
    if n < 2:
        return None
    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])
    zs = np.array([p.z for p in points])
    tt = np.linspace(0, n - 1, n_samples)
    xx, yy, zz = bezier_chain_eval(xs, ys, zs, tt)
    return tt, xx, yy, zz


def rdp_indices(points, epsilon):
    """Douglas-Peucker: (N,3) 점들 중, 원곡선을 epsilon(mm) 이내로 근사하는 데
    필요한 최소한의 점 인덱스 집합을 반환 (곡률이 큰 곳엔 많이, 완만한 곳엔 적게 남음)"""
    n = len(points)
    if n < 3:
        return list(range(n))
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        start, end = points[i0], points[i1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        seg = points[i0:i1 + 1]
        if line_len < 1e-9:
            dists = np.linalg.norm(seg - start, axis=1)
        else:
            line_unit = line_vec / line_len
            vecs = seg - start
            proj_len = np.clip(vecs @ line_unit, 0, line_len)
            proj_points = start + np.outer(proj_len, line_unit)
            dists = np.linalg.norm(seg - proj_points, axis=1)
        idx_local = int(np.argmax(dists))
        if dists[idx_local] > epsilon:
            idx_global = i0 + idx_local
            keep[idx_global] = True
            stack.append((i0, idx_global))
            stack.append((idx_global, i1))
    return list(np.where(keep)[0])


def draw_axis_indicator(ax, xlabel, ylabel):
    """평면 좌측 하단에 +X/+Y(해당 평면의 두 축) 방향 화살표를 표시 (축 혼동 방지용)"""
    ax.annotate("", xy=(0.14, 0.03), xytext=(0.03, 0.03), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(0.03, 0.14), xytext=(0.03, 0.03), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate(xlabel, xy=(0.17, 0.03), xycoords="axes fraction", fontsize=8, va="center", weight="bold")
    ax.annotate(ylabel, xy=(0.03, 0.17), xycoords="axes fraction", fontsize=8, ha="center", weight="bold")


# ---------------------------------------------------------------------------
# 메인 그리기 함수 - GUI 상태(점 목록, 슬라이더 값, 실패지점, 추천 등)를
# 전부 인자로 받는다. self는 전혀 모른다.
# ---------------------------------------------------------------------------

def draw_view_2d(ax, canvas, view, points, x_value, y_value, z_value,
                  bad_point_idx, bad_segment, suggestion, suggestion_region):
    ax.clear()
    if view == "xy":
        res = bg_slice_xy(z_value)
        extent = [_y_min, _y_min + _ny * _step, _x_min, _x_min + _nx * _step]
        xlabel, ylabel = "Y (mm)", "X (mm)"
    elif view == "yz":
        res = bg_slice_yz(x_value)
        extent = [_y_min, _y_min + _ny * _step, _z_min, _z_min + _nz * _step]
        xlabel, ylabel = "Y (mm)", "Z (mm)"
    else:
        res = bg_slice_xz(y_value)
        extent = [_x_min, _x_min + _nx * _step, _z_min, _z_min + _nz * _step]
        xlabel, ylabel = "X (mm)", "Z (mm)"

    if res is None:
        ax.set_facecolor("#ffcccc")
    else:
        safe, collide, marg = res
        # 안전지대는 단색이 아니라 '안전여유(mm)'에 따른 농담으로 칠한다.
        #   진할수록 데드존 경계에서 멀다 = 여유가 크다.
        shade = MARGIN_CMAP(np.clip(marg / MARGIN_VMAX, 0.0, 1.0))[..., :3]
        bg = np.zeros(safe.shape + (3,))
        bg[..., :] = [0.88, 0.88, 0.88]     # 도달 불가 = 회색
        bg[safe] = shade[safe]              # 안전 = 여유에 따른 농담
        bg[collide] = [1.0, 0.8, 0.55]      # 자가충돌 = 주황
        # imshow는 (행=세로축, 열=가로축) 순서를 요구.
        # xy 뷰는 세로가 X이므로 (nx,ny) 그대로, 나머지는 전치해야 맞음.
        img = bg if view == "xy" else np.transpose(bg, (1, 0, 2))
        ax.imshow(img, extent=extent, origin="lower", aspect="auto", alpha=0.75)

        # --- 등고선: 같은 안전여유를 갖는 지점을 선으로 연결 ---
        m2 = marg if view == "xy" else marg.T
        if np.any(m2 > 0):
            # imshow가 실제로 셀을 배치한 중심 좌표와 정확히 일치시켜야 선이 어긋나지 않음
            nrow, ncol = m2.shape
            hh = extent[0] + (np.arange(ncol) + 0.5) * (extent[1] - extent[0]) / ncol
            vv = extent[2] + (np.arange(nrow) + 0.5) * (extent[3] - extent[2]) / nrow
            top = float(m2.max())
            levels = sorted({round(f * MARGIN_VMAX, 1) for f in MARGIN_LEVEL_FRACS})
            levels = [lv for lv in levels if 0 < lv < top]
            if levels:
                cs = ax.contour(hh, vv, m2, levels=levels,
                                colors="#20502a", linewidths=0.7, alpha=0.65)
                ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")

    # 경로: 부드러운 곡선(스플라인)으로 렌더링 (화면 미리보기 전용)
    #      문제가 있는 구간은 빨간색으로 강조
    curve = sample_curve(points, n_samples=200)
    if curve is not None:
        tt, xx, yy, zz = curve
        if view == "xy":
            cx, cy = yy, xx
        elif view == "yz":
            cx, cy = yy, zz
        else:
            cx, cy = xx, zz
        if bad_segment is not None:
            bad_mask = (tt >= bad_segment) & (tt <= bad_segment + 1)
            ax.plot(cx, cy, "b-", linewidth=1.8, alpha=0.5)
            ax.plot(cx[bad_mask], cy[bad_mask], "r-", linewidth=3.0, alpha=0.95)
        else:
            ax.plot(cx, cy, "b-", linewidth=1.8, alpha=0.8)

    for i, p in enumerate(points):
        if view == "xy":
            x, y = p.y, p.x
        elif view == "yz":
            x, y = p.y, p.z
        else:
            x, y = p.x, p.z
        # 통과한 점은 배경 등고선과 '같은' 컬러맵으로 칠해 여유 정도를 바로 비교할 수 있게 한다.
        # (실패/미검증은 상태 자체가 더 중요하므로 빨강/회색을 유지)
        if i == bad_point_idx:
            color, msize = "red", 13
        elif p.valid is None:
            color, msize = "gray", 9
        elif p.valid:
            color, msize = margin_marker_color(margin_at(p.x, p.y, p.z)), 10
        else:
            color, msize = "red", 13
        # 앵커(곡선이 실제로 지나는 점, 로봇이 방문)는 원, 구간 제어점은 다이아몬드
        is_anchor_i = p.is_anchor
        marker = "o" if is_anchor_i else "D"
        msize = msize if is_anchor_i else msize * 0.8
        ax.plot(x, y, marker, color=color, markersize=msize, markeredgecolor="black")
        # [번호] 마커 색이 초록/빨강/회색으로 다 달라 검정 글씨는 안 보일 때가 있어,
        # 흰 글씨 + 검정 테두리(outline)로 어떤 배경에서도 읽히게 한다.
        ax.annotate(str(i + 1), (x, y), textcoords="offset points", xytext=(0, 0),
                    fontsize=6.5, ha="center", va="center", color="white",
                    weight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        ax.annotate(f"{i+1}: ({p.x:.0f},{p.y:.0f},{p.z:.0f})", (x, y),
                    textcoords="offset points", xytext=(0, -14), fontsize=7, ha="center")

    # 추천 대체 위치 - [6차] 점 하나가 아니라 '범위'로 표시한다.
    #   옅은 주황 점 + 반투명 영역 = 국소검사를 통과한 유력 후보들
    #   진한 주황 원 + 연결선     = 전체 경로 재검증까지 통과한 '확정' 추천
    # 확정이 없어도(전체검증 통과 후보 없음) 유력 영역은 그려준다 - 예전엔
    # 이 경우 화면에 아무것도 안 떠서 사용자가 어디로 옮길지 알 수 없었다.
    if bad_point_idx is not None and (suggestion_region or suggestion):
        def _proj(px, py, pz):
            if view == "xy":
                return py, px
            if view == "yz":
                return py, pz
            return px, pz

        if suggestion_region:
            reg_h, reg_v = [], []
            for cx, cy, cz, _ver in suggestion_region:
                h, v = _proj(cx, cy, cz)
                reg_h.append(h); reg_v.append(v)
            # 영역감을 주려고 반투명 큰 점을 겹쳐 찍고(번지는 효과) 그 위에
            # 작은 점을 찍는다. 후보가 3개 이상이면 볼록껍질로 영역을 채운다.
            ax.scatter(reg_h, reg_v, s=260, c="darkorange", alpha=0.10,
                       edgecolors="none", zorder=3)
            ax.scatter(reg_h, reg_v, s=18, c="darkorange", alpha=0.55,
                       edgecolors="none", zorder=4)
            if len(reg_h) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    pts2d = np.column_stack([reg_h, reg_v])
                    hull = ConvexHull(pts2d)
                    ax.fill(pts2d[hull.vertices, 0], pts2d[hull.vertices, 1],
                            facecolor="darkorange", alpha=0.12, edgecolor="darkorange",
                            linewidth=1.0, linestyle="--", zorder=2)
                except Exception:
                    # 후보들이 한 직선 위에 있으면 볼록껍질이 실패할 수 있다 -
                    # 그래도 위의 점 표시만으로 충분히 범위가 보이므로 무시한다.
                    pass

        bp = points[bad_point_idx]
        b_h, b_v = _proj(bp.x, bp.y, bp.z)
        if suggestion is not None:
            sx, sy, sz = suggestion
            s_h, s_v = _proj(sx, sy, sz)
            ax.plot([b_h, s_h], [b_v, s_v], "--", color="darkorange",
                    linewidth=1.5, alpha=0.9, zorder=5)
            ax.plot(s_h, s_v, "o", markerfacecolor="none", markeredgecolor="darkorange",
                    markersize=13, markeredgewidth=2, zorder=6)
            ax.annotate(f"추천 ({sx:.0f},{sy:.0f},{sz:.0f})", (s_h, s_v),
                        textcoords="offset points", xytext=(0, 12), fontsize=7,
                        ha="center", color="darkorange")
        elif suggestion_region:
            # 확정은 없고 유력 영역만 있는 경우 - 영역이라는 걸 라벨로 밝힌다
            ax.annotate(f"유력 영역 {len(suggestion_region)}곳 (확정 아님)",
                        (float(np.mean(reg_h)), float(np.mean(reg_v))),
                        textcoords="offset points", xytext=(0, 12), fontsize=7,
                        ha="center", color="darkorange")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    draw_axis_indicator(ax, xlabel.split(" ")[0], ylabel.split(" ")[0])
    canvas.draw_idle()


def draw_view_3d(ax_3d, canvas_3d, points, bad_point_idx, bad_segment, suggestion, suggestion_region):
    ax_3d.clear()
    if points:
        curve = sample_curve(points, n_samples=200)
        if curve is not None:
            tt3, xx, yy, zz = curve
            if bad_segment is not None:
                m = (tt3 >= bad_segment) & (tt3 <= bad_segment + 1)
                ax_3d.plot(xx, yy, zz, "b-", linewidth=1.5, alpha=0.5)
                ax_3d.plot(xx[m], yy[m], zz[m], "r-", linewidth=3.0)
            else:
                ax_3d.plot(xx, yy, zz, "b-", linewidth=1.8, alpha=0.8)
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        colors = []
        sizes = []
        # [주의] scatter(c=...)는 색 이름과 RGBA 튜플을 섞은 리스트를 받지 못한다.
        #        모두 RGBA 튜플로 통일해서 넘긴다.
        for i, p in enumerate(points):
            if i == bad_point_idx or p.valid is False:
                colors.append(to_rgba("red")); sizes.append(120)
            elif p.valid is None:
                colors.append(to_rgba("gray")); sizes.append(60)
            else:
                # 2D 등고선과 동일한 컬러맵 (여유 클수록 진함)
                colors.append(to_rgba(margin_marker_color(margin_at(p.x, p.y, p.z)))); sizes.append(70)
        # [경로 방식 변경] 앵커(원)/베지어 제어점(다이아몬드) 구분 - scatter는 한 번
        # 호출에 마커 모양이 하나뿐이라 두 그룹으로 나눠 따로 호출한다.
        is_anchor = np.array([pp.is_anchor for pp in points])
        xs_a, ys_a, zs_a = np.array(xs), np.array(ys), np.array(zs)
        colors_a = np.array(colors); sizes_a = np.array(sizes)
        if is_anchor.any():
            ax_3d.scatter(xs_a[is_anchor], ys_a[is_anchor], zs_a[is_anchor],
                          c=colors_a[is_anchor], s=sizes_a[is_anchor],
                          marker="o", edgecolors="black")
        if (~is_anchor).any():
            ax_3d.scatter(xs_a[~is_anchor], ys_a[~is_anchor], zs_a[~is_anchor],
                          c=colors_a[~is_anchor], s=sizes_a[~is_anchor] * 0.8,
                          marker="D", edgecolors="black")
        # [6차] 3D에서도 추천을 '범위'로 - 유력 후보들을 옅은 주황 점으로
        # 뿌리고, 확정 추천만 큰 빈 원 + 연결선으로 강조한다.
        if bad_point_idx is not None and suggestion_region:
            rx = [c[0] for c in suggestion_region]
            ry = [c[1] for c in suggestion_region]
            rz = [c[2] for c in suggestion_region]
            ax_3d.scatter(rx, ry, rz, c="darkorange", s=28, alpha=0.35,
                          edgecolors="none")
        if suggestion is not None and bad_point_idx is not None:
            sx, sy, sz = suggestion
            bp = points[bad_point_idx]
            ax_3d.plot([bp.x, sx], [bp.y, sy], [bp.z, sz], "--", color="darkorange", linewidth=1.5)
            ax_3d.scatter([sx], [sy], [sz], facecolors="none", edgecolors="darkorange", s=140, linewidths=2)
        for i, p in enumerate(points):
            ax_3d.text(p.x, p.y, p.z, f"{i+1}: ({p.x:.0f},{p.y:.0f},{p.z:.0f})", fontsize=7)
    ax_3d.set_xlabel("X (mm)")
    ax_3d.set_ylabel("Y (mm)")
    ax_3d.set_zlabel("Z (mm)")
    # 격자 단위가 계속 바뀌지 않도록 축 범위를 작업영역 전체로 고정
    ax_3d.set_xlim(_x_min, _x_min + _nx * _step)
    ax_3d.set_ylim(_y_min, _y_min + _ny * _step)
    ax_3d.set_zlim(_z_min, _z_min + _nz * _step)
    canvas_3d.draw_idle()
