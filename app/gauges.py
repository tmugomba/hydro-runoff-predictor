"""
SVG speedometer-style gauge generator.

Builds a self-contained <svg> string for a single semicircular gauge (value arc + needle),
styled to match the dark navy / amber / cyan design language used across the portfolio's
other dashboards (Wind Calculator, Solar Yield Dashboard). No external image files or JS
libraries -- pure SVG + inline CSS, so it drops straight into st.markdown(unsafe_allow_html=True).

GEOMETRY NOTE: uses the standard SVG angle convention (angle measured clockwise from the
positive x-axis, since SVG's y-axis points down). The gauge sweeps from angle 180 deg
(left point, value = min) through 270 deg (top, value = midpoint) to 360 deg (right point,
value = max), with sweep-flag=1 so the arc always draws through the top rather than the
bottom. Verified visually before wiring into the app.
"""

import math


def _point(cx: float, cy: float, r: float, theta_deg: float) -> tuple[float, float]:
    """Cartesian point on a circle of radius r centered at (cx, cy), at angle theta_deg
    (standard SVG convention: 0 = right, 90 = bottom, 180 = left, 270 = top)."""
    rad = math.radians(theta_deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def _angle_for_frac(frac: float) -> float:
    """Map a 0-1 fraction of the gauge's range to its sweep angle (180 deg = min, 360 deg = max)."""
    return 180 + max(0.0, min(1.0, frac)) * 180


def _arc_path(cx: float, cy: float, r: float, frac_a: float, frac_b: float) -> str:
    """SVG path 'd' string for the arc between two fractions of the gauge range."""
    a1, a2 = _angle_for_frac(frac_a), _angle_for_frac(frac_b)
    x1, y1 = _point(cx, cy, r, a1)
    x2, y2 = _point(cx, cy, r, a2)
    large_arc = 1 if (a2 - a1) > 180 else 0
    return f"M {x1:.2f},{y1:.2f} A {r},{r} 0 {large_arc} 1 {x2:.2f},{y2:.2f}"


def speedometer_svg(
    value: float,
    min_val: float,
    max_val: float,
    label: str,
    unit: str,
    value_fmt: str = "{:.1f}",
    color: str = "#f5a623",
    track_color: str = "#1c2540",
    size: int = 240,
) -> str:
    """
    Build one semicircular speedometer gauge as a standalone SVG string.

    value, min_val, max_val : gauge reading and its scale bounds.
    label : short caption shown under the value (e.g. "DISCHARGE").
    unit : unit string shown next to the value (e.g. "m3/s").
    value_fmt : format spec applied to `value` before display.
    color : accent color of the value arc + needle hub (amber for discharge/power,
            cyan for capacity factor, by convention used when calling this function).
    track_color : color of the unfilled background arc.
    size : SVG viewBox width in px; height is size * 0.72.
    """
    width = size
    height = int(size * 0.82)
    cx, cy, r = width / 2, height * 0.80, width * 0.38

    frac = 0.0 if max_val == min_val else (value - min_val) / (max_val - min_val)
    frac = max(0.0, min(1.0, frac))

    track_d = _arc_path(cx, cy, r, 0, 1)
    value_d = _arc_path(cx, cy, r, 0, frac)

    needle_len = r * 0.72
    needle_x, needle_y = _point(cx, cy, needle_len, _angle_for_frac(frac))

    # small tick marks at 0%, 25%, 50%, 75%, 100% of the scale, for a reference rail
    ticks = ""
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        ang = _angle_for_frac(t)
        x_out, y_out = _point(cx, cy, r + 9, ang)
        x_in, y_in = _point(cx, cy, r - 5, ang)
        ticks += (
            f'<line x1="{x_in:.1f}" y1="{y_in:.1f}" x2="{x_out:.1f}" y2="{y_out:.1f}" '
            f'stroke="#4a5876" stroke-width="2"/>'
        )

    value_text = value_fmt.format(value)
    stroke_w = width * 0.06

    return f"""
<svg class="gauge-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="glow-{label}" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="4.5" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <path d="{track_d}" stroke="{track_color}" stroke-width="{stroke_w:.1f}" fill="none" stroke-linecap="round"/>
  {ticks}
  <path class="gauge-value-arc" d="{value_d}" stroke="{color}" stroke-width="{stroke_w:.1f}"
        fill="none" stroke-linecap="round" filter="url(#glow-{label})"/>
  <line x1="{cx:.1f}" y1="{cy:.1f}" x2="{needle_x:.2f}" y2="{needle_y:.2f}"
        stroke="#e8ecf3" stroke-width="3" stroke-linecap="round"/>
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{width*0.035:.1f}" fill="{color}" filter="url(#glow-{label})"/>
  <text x="{cx:.1f}" y="{cy - height*0.30:.1f}" text-anchor="middle" class="gauge-value"
        font-family="'JetBrains Mono', monospace" font-size="{width*0.135:.1f}" fill="#e8ecf3"
        font-weight="600">{value_text}</text>
  <text x="{cx:.1f}" y="{cy - height*0.14:.1f}" text-anchor="middle" class="gauge-unit"
        font-family="'JetBrains Mono', monospace" font-size="{width*0.055:.1f}" fill="#8a93a6">{unit}</text>
  <text x="{cx:.1f}" y="{height*0.98:.1f}" text-anchor="middle" class="gauge-label"
        font-family="'Space Grotesk', sans-serif" font-size="{width*0.062:.1f}" fill="#8a93a6"
        letter-spacing="1.5">{label}</text>
</svg>
""".strip()
