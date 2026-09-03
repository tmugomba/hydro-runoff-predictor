"""
Discharge -> power conversion for the Mersey River run-of-river system.

IMPORTANT MODELING SIMPLIFICATION -- read before using this module:

The real "Mersey Hydro" asset (Nova Scotia Power) is NOT a single turbine with one clean head.
It's a system of SIX powerhouses built in 1928-1929, fed by six reservoirs, nine dams, and two
canals, running from Lake Rossignol down to the village of Milton -- the same stretch our gauge
station (01ED003, Mersey River at Milton) sits on. Combined, the system is rated up to 42.5 MW.

Since we don't have (and shouldn't fabricate) the individual head/efficiency of each of the six
powerhouses, this module treats the whole cascade as ONE lumped equivalent turbine. We calibrate
a single conversion constant so that at a chosen "design flow" (a percentile of the historical
flow-duration curve), the model's output matches the real system's 42.5 MW rated capacity. Output
is capped at that rating for any flow above design flow -- which is also physically realistic,
since real run-of-river plants spill excess water once flow exceeds what their turbines can pass.

This is a defensible approximation for a portfolio-level feasibility/education tool, but it is
NOT a substitute for an actual engineering study of the real six-powerhouse system (which would
need each powerhouse's individual head, turbine curve, and canal/penstock losses).

Source for the 42.5 MW rated capacity and six-powerhouse description: Atlantica Centre for Energy,
"Nova Scotia's Energy Resources" (nova scotia hydro asset summaries).
"""

import numpy as np
import pandas as pd

# real-world rated capacity of the full Mersey Hydroelectric System (six powerhouses combined)
MERSEY_SYSTEM_RATED_CAPACITY_MW = 42.5

# design-flow quantile used to calibrate the lumped power constant, expressed as the
# quantile of the discharge distribution (0.98 = the flow exceeded only 2% of the time,
# i.e. a near-flood-level flow).
#
# NOTE ON WHY THIS IS SO HIGH: standard small-hydro sizing conventions typically use a much
# more common design flow (Q20-Q40, i.e. exceeded 20-40% of the time). Needing a rare Q2 flow
# here to hit a realistic ~40-60% capacity factor is a signal that this single lumped-equivalent
# model is a rough approximation, not a precise one. The real Mersey system is six separate
# powerhouses drawing on combined flow and storage from six reservoirs spread across the
# watershed -- their true combined hydraulic capacity is likely higher relative to what a single
# downstream gauge (ours, at Milton) suggests, since much of that capacity is fed by water that
# never passes the Milton gauge point at all. Documented here, and in the project README, as a
# known limitation of the lumped-equivalent simplification -- not hidden or glossed over.
DESIGN_FLOW_QUANTILE = 0.98


def calibrate_power_constant(design_flow_cms: float, rated_capacity_mw: float = MERSEY_SYSTEM_RATED_CAPACITY_MW) -> float:
    """
    Solve for the single lumped conversion constant k (MW per m3/s) such that:
        power_mw = k * design_flow_cms == rated_capacity_mw

    This stands in for the combined effect of head, efficiency, and losses across the real
    six-powerhouse system, which we don't have individual specs for.

    design_flow_cms : the flow (m3/s) at which the system is assumed to hit its rated capacity.
                       A reasonable choice is a flow-duration-curve percentile like Q20-Q30
                       (exceeded 20-30% of the time) -- common small-hydro sizing convention,
                       since designing for the rare flood peak would waste turbine capacity most
                       of the year.
    rated_capacity_mw : the real system's rated output, defaults to the actual 42.5 MW.
    """
    if design_flow_cms <= 0:
        raise ValueError("design_flow_cms must be positive")
    return rated_capacity_mw / design_flow_cms


def discharge_to_power(
    discharge_cms: "float | pd.Series",
    k: float,
    rated_capacity_mw: float = MERSEY_SYSTEM_RATED_CAPACITY_MW,
) -> "float | pd.Series":
    """
    Convert discharge (m3/s) to estimated power output (MW) using the calibrated constant k,
    capped at the system's rated capacity -- modeling the real-world behaviour of excess flow
    being spilled rather than turbined once the system is at full output.

    discharge_cms : single value or pandas Series of discharge in m3/s.
    k : calibration constant from calibrate_power_constant().
    rated_capacity_mw : cap applied to the output.
    """
    raw_power_mw = discharge_cms * k
    return np.minimum(raw_power_mw, rated_capacity_mw)


def capacity_factor(power_series_mw: pd.Series, rated_capacity_mw: float = MERSEY_SYSTEM_RATED_CAPACITY_MW) -> float:
    """
    Average power output as a fraction of rated capacity over the given period -- the standard
    metric for how much of a plant's theoretical maximum output it actually achieves. Real
    run-of-river hydro typically sits in the 40-60% range; anything wildly outside that is worth
    double-checking the design_flow assumption.
    """
    return power_series_mw.mean() / rated_capacity_mw
