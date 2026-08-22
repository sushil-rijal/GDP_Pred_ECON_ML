# INVERSE FEASIBILITY PROTOCOL

## Purpose

The forward model asks where output is heading. The inverse exercise asks what combinations of factor accumulation, productivity, and efficiency are consistent with attaining a target with specified posterior probability.

## Parameterized conditions/levers

- `delta_iG`: constant increment to public-investment/GDP over the policy horizon. Direct policy lever.
- `delta_iP`: constant increment to private-capital-formation/GDP. Required private-sector condition, not a direct government choice.
- `delta_gH`: annual increment to human-capital growth.
- `delta_gA`: annual increment to productivity/frontier growth.
- `delta_theta`: infrastructure-efficiency improvement.

Arbitrary unconstrained year-by-year paths are not searched in the baseline. The low-dimensional parameterization makes the feasibility surface computable and interpretable.

## Probability thresholds

Report q = 0.50, 0.80, and 0.90. For each target T,

`F_T(q) = {x : P(T attained | x, data) >= q}`.

The q=0.80 frontier is the principal high-confidence display; 0.50 and 0.90 show sensitivity to the desired confidence level.

## Capital accumulation

Use the source-consistent PIM identity and propagate uncertainty in investment and depreciation. ICSD-implied depreciation schedules provide consistency checks/prior centers rather than independently identified depreciation estimates.

## Computation

1. Draw posterior/model states from the selected structural model.
2. For each low-dimensional lever/condition grid point, generate future input paths.
3. Propagate those paths through the production model.
4. Compute empirical target-attainment frequency across posterior/predictive draws (sample-average approximation).
5. Mark grid points satisfying q.
6. Recover Pareto-efficient frontier/surfaces.

## Headline displays

1. `(delta_iG, delta_iP)` frontier at baseline H/A/theta — direct probabilistic descendant of the original thesis G-K exercise.
2. Shift of that frontier under improved `theta`.
3. Shift under `delta_gH` and `delta_gA`.
4. For Nepal, separate frontiers for Sixteenth Plan, 2030 growth, 2030 GNIpc, and Vision 2100 targets.

## Feasibility versus optimization

The baseline reports the feasible set/frontier before imposing a social cost function. Optional Tier-3 Bayesian decision analysis may minimize an explicit posterior expected loss only after the feasible-set results are understood.

## Reporting rules

- Distinguish policy levers from required private/global conditions.
- Empty F(q) is reported as target infeasibility under the admissible scenario set, not as model failure.
- Current-policy inverse solves use the current-information posterior, not the 2019-origin historical-validation posterior.
- All official targets remain in their official concepts/units; conversions are documented rather than assumed.
