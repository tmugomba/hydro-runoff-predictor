"""
Animated run-of-river schematic: reservoir -> canal/penstock -> powerhouse -> grid.

Two animation techniques, both native SVG/CSS (no JS needed, so this renders fine inside
Streamlit's st.markdown(unsafe_allow_html=True)):
  - Flow particles: SVG SMIL <animateMotion> tracing a <path>, staggered begin offsets so
    several dots appear strung along the water route at once.
  - Turbine glow pulse: a CSS @keyframes opacity/filter pulse on the powerhouse turbine
    circle, suggesting continuous generation.
"""

WATER_PATH_D = "M 40,70 C 140,70 160,150 260,150 C 340,150 360,110 430,110"


def schematic_svg(active_power_mw: float, rated_capacity_mw: float) -> str:
    """
    Build the schematic. `active_power_mw` vs `rated_capacity_mw` sets how many flow
    particles are shown and how fast they move -- more/faster particles at higher output,
    giving the diagram a rough "load" feel without needing a physics engine.
    """
    load_frac = 0.0 if rated_capacity_mw == 0 else max(0.05, min(1.0, active_power_mw / rated_capacity_mw))
    n_particles = 3 + round(load_frac * 4)  # 3 to 7 particles
    duration = 3.6 - load_frac * 1.8  # faster flow (shorter duration) at higher load

    particles = ""
    for i in range(n_particles):
        delay = (duration / n_particles) * i
        particles += f"""
    <circle r="4" fill="#22d3ee" filter="url(#particle-glow)">
      <animateMotion dur="{duration:.2f}s" begin="{delay:.2f}s" repeatCount="indefinite"
                      path="{WATER_PATH_D}" rotate="auto"/>
    </circle>"""

    return f"""
<div class="schematic-wrap">
<style>
@keyframes turbine-pulse {{
  0%   {{ filter: drop-shadow(0 0 2px #f5a623); opacity: 0.85; }}
  50%  {{ filter: drop-shadow(0 0 10px #f5a623); opacity: 1; }}
  100% {{ filter: drop-shadow(0 0 2px #f5a623); opacity: 0.85; }}
}}
.turbine-core {{ animation: turbine-pulse 1.8s ease-in-out infinite; transform-origin: 260px 150px; }}
</style>
<svg viewBox="0 0 490 220" xmlns="http://www.w3.org/2000/svg" class="schematic-svg">
  <defs>
    <filter id="particle-glow" x="-200%" y="-200%" width="500%" height="500%">
      <feGaussianBlur stdDeviation="2.2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="river-grad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1c4a63"/>
      <stop offset="100%" stop-color="#144056"/>
    </linearGradient>
  </defs>

  <!-- water route (canal / penstock) -->
  <path d="{WATER_PATH_D}" stroke="url(#river-grad)" stroke-width="10" fill="none" stroke-linecap="round"/>
  <path d="{WATER_PATH_D}" stroke="#0e2635" stroke-width="10" fill="none" stroke-linecap="round"
        stroke-dasharray="1 0" opacity="0.35"/>
  {particles}

  <!-- reservoir / dam icon -->
  <g>
    <rect x="10" y="35" width="14" height="55" fill="#2a3550" stroke="#4a5876" stroke-width="1.5"/>
    <path d="M 24,45 L 55,45 L 55,60 L 24,60 Z" fill="#1c4a63"/>
    <path d="M 24,60 L 55,60 L 55,75 L 24,75 Z" fill="#144056"/>
    <text x="17" y="105" font-family="'Space Grotesk', sans-serif" font-size="10.5"
          fill="#8a93a6" text-anchor="start" letter-spacing="0.5">RESERVOIR</text>
    <text x="17" y="117" font-family="'Space Grotesk', sans-serif" font-size="8.5"
          fill="#5a6684" text-anchor="start">Six dams, Lake Rossignol</text>
  </g>

  <!-- lumped powerhouse -->
  <g transform="translate(230,120)">
    <rect x="0" y="0" width="60" height="42" rx="3" fill="#1a2036" stroke="#4a5876" stroke-width="1.5"/>
    <polygon points="0,0 30,-16 60,0" fill="#232b47" stroke="#4a5876" stroke-width="1.5"/>
    <circle class="turbine-core" cx="30" cy="21" r="11" fill="none" stroke="#f5a623" stroke-width="3"/>
    <circle cx="30" cy="21" r="3.5" fill="#f5a623"/>
    <text x="30" y="66" font-family="'Space Grotesk', sans-serif" font-size="10.5"
          fill="#8a93a6" text-anchor="middle" letter-spacing="0.5">POWERHOUSE</text>
    <text x="30" y="78" font-family="'Space Grotesk', sans-serif" font-size="8.5"
          fill="#5a6684" text-anchor="middle">Lumped equivalent, 42.5 MW rated</text>
  </g>

  <!-- transmission to grid -->
  <path d="M 430,110 L 460,90" stroke="#4a5876" stroke-width="2.5"/>
  <g transform="translate(460,55)">
    <line x1="0" y1="0" x2="0" y2="38" stroke="#4a5876" stroke-width="2.5"/>
    <line x1="-13" y1="8" x2="13" y2="8" stroke="#4a5876" stroke-width="2"/>
    <line x1="-9" y1="18" x2="9" y2="18" stroke="#4a5876" stroke-width="2"/>
    <circle cx="0" cy="-6" r="4" fill="#22d3ee" filter="url(#particle-glow)"/>
    <text x="0" y="55" font-family="'Space Grotesk', sans-serif" font-size="10.5"
          fill="#8a93a6" text-anchor="middle" letter-spacing="0.5">GRID</text>
    <text x="0" y="67" font-family="'Space Grotesk', sans-serif" font-size="8.5"
          fill="#5a6684" text-anchor="middle">Nova Scotia Power</text>
  </g>
</svg>
</div>
""".strip()
