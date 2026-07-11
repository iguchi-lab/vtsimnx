"""Physics / perf assertion tolerances and their rationale.

Values are intentionally documented here so golden-only regressions
cannot silently lock in wrong physics without an invariant check.
"""

from __future__ import annotations

# Thermal network residual (W). Prefer simulation.tolerance.thermal_balance;
# falls back to simulation.tolerance.thermal when unset (baseline 1e-6).
# Log `maxBalance` is compared against this.
THERMAL_BALANCE_ABS_W = 1e-5
# Rationale: DirectT reports maxBalance on the order of 1e-12..1e-13 for the
# simple conductance case; allow headroom for float32 artifact reconstruction
# and multi-node RC walls (~1e-5 W is still << typical heat flows of 10..1000 W).

# Relative energy residual vs characteristic heat-flow scale on a node.
THERMAL_BALANCE_REL = 1e-3
# Rationale: float32 heat_rate bins (~1e-7 relative) plus surface RC splitting;
# 0.1% of |Q| is far tighter than engineering U-value uncertainty.

# Ventilation nodal mass imbalance (m3/s or solver flow units).
VENT_MASS_BALANCE_ABS = 1e-5
# Rationale: pressure solver tol 1e-5..1e-6; flows in baseline cases are O(0.1..10).

# Humidity ratio bounds (kg/kg').
HUMIDITY_X_MIN = 0.0
HUMIDITY_X_MAX = 0.05
# Rationale: physical absolute humidity at room conditions; values above ~0.05
# indicate unit/model blow-up rather than a valid moist-air state.

# Humidity flux nodal residual (kg/s), absolute.
HUMIDITY_FLUX_BALANCE_ABS = 1e-4
# Rationale: Phase1 linear RC; generation/advection O(1e-5..1e-3) kg/s in baselines.

# Concentration (contaminant) non-negativity and flux residual.
CONCENTRATION_MIN = 0.0
CONCENTRATION_FLUX_BALANCE_ABS = 1e-6

# Aircon OFF sensible heat (W).
AIRCON_OFF_ABS_W = 1e-3
# Rationale: numerical noise around zero when mode is OFF.

# Timestep refinement endpoint temperature difference (K).
TIMESTEP_ENDPOINT_ABS_K = 0.5
# Rationale: first-order time discretisation; 0.5 K on a multi-hour cool-down
# with dt=3600 vs 1800 is an engineering consistency bound, not a machine eps.

# Performance regression warning thresholds (not hard fails).
PERF_WARN_TIME_RATIO = 2.0
PERF_WARN_MEMORY_RATIO = 2.0
PERF_WARN_ARTIFACT_RATIO = 2.0
PERF_WARN_LU_RATIO = 3.0
PERF_WARN_AIRCON_RECOMPUTE_RATIO = 3.0
# Rationale: CI runners vary; warn on large regressions only. Strict pass/fail
# is intentionally avoided for wall-clock metrics.
