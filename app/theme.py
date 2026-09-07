"""
Shared CSS for the hydro app -- dark navy/graphite/amber/cyan palette, Space Grotesk for
headers, JetBrains Mono for numerals, consistent with the Wind Calculator / Solar Yield
Dashboard / Grid Explorer design language used across the rest of the portfolio.
"""

import base64

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg-navy: #0a0e17;
  --panel: #131a2b;
  --panel-border: rgba(255,255,255,0.08);
  --amber: #f5a623;
  --cyan: #22d3ee;
  --text-primary: #e8ecf3;
  --text-muted: #8a93a6;
}

.stApp {
  background: radial-gradient(circle at 15% 0%, #0e1524 0%, var(--bg-navy) 45%);
  color: var(--text-primary);
}

h1, h2, h3, h4, .hero-title {
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--text-primary) !important;
}

p, li, span, div, label {
  font-family: 'Space Grotesk', sans-serif;
}

.mono, .stat-value, code {
  font-family: 'JetBrains Mono', monospace !important;
}

/* hero header */
.hero-block {
  padding: 1.4rem 1.8rem;
  border-radius: 14px;
  background: linear-gradient(135deg, #101a2f 0%, #0d1420 100%);
  border: 1px solid var(--panel-border);
  margin-bottom: 1.4rem;
}
.hero-title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(90deg, var(--amber), var(--cyan));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: inline-block;
}
.hero-sub {
  color: var(--text-muted);
  font-size: 0.95rem;
  margin-top: 0.35rem;
}

/* generic card / panel */
.card {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 14px;
  padding: 1.1rem 1.3rem;
  margin-bottom: 1rem;
  transition: box-shadow 0.25s ease, border-color 0.25s ease, transform 0.25s ease;
}
.card:hover {
  border-color: rgba(245,166,35,0.35);
  box-shadow: 0 0 0 1px rgba(245,166,35,0.12), 0 8px 24px rgba(0,0,0,0.35);
  transform: translateY(-2px);
}
.card-title {
  font-size: 0.78rem;
  letter-spacing: 1.2px;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}

/* gauges */
.gauge-card {
  text-align: center;
  transition: filter 0.25s ease;
}
.gauge-card:hover .gauge-value-arc {
  filter: drop-shadow(0 0 6px currentColor);
}
.gauge-svg { width: 100%; height: auto; }

/* schematic */
.schematic-wrap {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 14px;
  padding: 1rem 1.4rem 0.6rem 1.4rem;
  margin-bottom: 1rem;
}
.schematic-svg { width: 100%; height: auto; }

/* tooltip (hover-glow "i" info marker) */
.tip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  border: 1px solid var(--text-muted);
  color: var(--text-muted);
  font-size: 10px;
  font-family: 'JetBrains Mono', monospace;
  margin-left: 6px;
  cursor: help;
  position: relative;
  transition: color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.tip:hover {
  color: var(--cyan);
  border-color: var(--cyan);
  box-shadow: 0 0 8px rgba(34,211,238,0.5);
}
.tip .tip-bubble {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  bottom: 130%;
  left: 50%;
  transform: translateX(-50%);
  background: #1c2540;
  border: 1px solid var(--panel-border);
  color: var(--text-primary);
  font-size: 11.5px;
  font-family: 'Space Grotesk', sans-serif;
  padding: 8px 10px;
  border-radius: 8px;
  width: 210px;
  line-height: 1.35;
  z-index: 50;
  transition: opacity 0.2s ease;
  box-shadow: 0 6px 18px rgba(0,0,0,0.45);
}
.tip:hover .tip-bubble { visibility: visible; opacity: 1; }

/* footer */
.footer-block {
  margin-top: 2rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--panel-border);
  color: var(--text-muted);
  font-size: 0.85rem;
  text-align: center;
}
.footer-block a { color: var(--cyan); text-decoration: none; }
.footer-block a:hover { text-decoration: underline; }

/* section labels */
.section-label {
  font-size: 0.75rem;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--amber);
  margin-bottom: 0.2rem;
}

/* limitation callout */
.callout {
  border-left: 3px solid var(--amber);
  background: rgba(245,166,35,0.06);
  border-radius: 0 10px 10px 0;
  padding: 0.8rem 1rem;
  color: var(--text-muted);
  font-size: 0.88rem;
  line-height: 1.5;
}
</style>
"""


def tooltip(text: str) -> str:
    """Small inline hover-glow 'i' tooltip, e.g. next to a metric label."""
    return f'<span class="tip">i<span class="tip-bubble">{text}</span></span>'


def svg_img(svg_str: str, css_class: str = "") -> str:
    """
    Wrap a raw <svg>...</svg> string as a base64 data-URI <img> tag instead of injecting
    the SVG markup directly.

    Streamlit's st.html() (and st.markdown with unsafe_allow_html) sanitize HTML with
    DOMPurify, which by default does NOT allow raw <svg> elements through unless the
    SVG profile is explicitly enabled -- in practice this silently strips the whole
    <svg> subtree, leaving only its parent <div> (still styled, so it shows up as an
    empty card instead of an error). Encoding the SVG as an <img src="data:image/svg+xml;
    base64,...">  sidesteps that entirely: DOMPurify sees a plain <img> tag with a data
    URI, which is standard and always allowed, and the browser decodes/renders the SVG
    (including embedded SMIL animations and CSS @keyframes) as an image regardless of
    the sanitizer's SVG element policy.
    """
    b64 = base64.b64encode(svg_str.encode("utf-8")).decode("ascii")
    cls = f' class="{css_class}"' if css_class else ""
    return f'<img{cls} src="data:image/svg+xml;base64,{b64}" style="width:100%;height:auto;display:block;">'