# Anticipation

Set `anticipation=A` when outcomes may be affected for `A` periods before actual
treatment. The default is zero. All estimators use the same timing convention:

| For a unit first treated at `g` | Interpretation |
|---|---|
| `t < g-A` | Clean pre-treatment observations |
| `g-A <= t < g` | Anticipation window |
| `t >= g` | Actual post-treatment observations |
| `g-A-1` | Last clean period and SA/universal-CS reference |

Cohort labels always describe actual treatment, and event time remains `t-g`.
For example, `anticipation=2` with `g=6` uses period 3 as the clean reference;
periods 4 and 5 can have anticipatory effects at event times -2 and -1.
Never-treated units (`g=0`) remain clean throughout the sample.

```python
from diff_stream import cs_stream
from diff_stream.simulate import simulate_conditional_panel

panel = simulate_conditional_panel(1200, 8)
fit = cs_stream(
    "y ~ x + z",
    panel,
    panel=("id", "time"),
    cohort="first_treat",
    anticipation=2,
)
print(fit.group_time())
print(fit.aggregate().tidy())
```

`anticipation` must be a nonnegative integer; booleans and floats are rejected.
It applies in balanced, complete-unit and native unbalanced panel modes.
The global integer time grid remains consecutive.

## Estimator rules

**Callaway–Sant'Anna.** With varying bases, clean pre-treatment pseudo-effects
use the preceding period. Every affected date, including an anticipatory date,
uses `g-A-1`. With universal bases, every comparison uses `g-A-1` and that
reference cell is omitted. Not-yet-treated controls must be clean at both dates:
other cohorts satisfy `h-A > max(target, base)`, in addition to never-treated
controls. The target cohort is excluded from its own control group.

Balanced conditional CS still measures covariates at the earlier comparison
date; the affected-period reference therefore moves with anticipation. Native
unbalanced CS retains the established observed-date covariate/row-functional
convention and folds scores by original unit. Outcome regression, IPW, DR,
trimming and estimated-cohort-share inference otherwise retain their contracts.

**Sun–Abraham.** The saturated specification includes all supported cohort-time
interactions except `g-A-1`. The estimated anticipation cells are included,
as are clean pre-period interactions. Balanced panels retain the exact change
shortcut; unbalanced panels retain the exact within-cohort calendar-effect
systems. Both use the shifted reference and unit-clustered covariance.

**BJS.** The first stage fits unit/time effects on clean observations only,
`t < g-A` for eventually treated units, plus all never-treated observations.
It returns effects from `g-A` onward, so the event study includes negative
anticipation event times. First-stage derivatives, smart residual averaging,
all four `avgeffectsby` groups and cluster leave-out use this clean first stage.
No anticipation-window observation is reused as untreated training data.

## Samples and identification

A cohort with `g-A <= Tmin` has no clean pre-period. It raises an explicit error;
exclude that cohort deliberately before fitting. The package does not silently
drop early cohorts. With native unbalanced BJS, each retained unit additionally
needs an observed clean pre-period. SA requires an observed cohort reference
and an identified observation graph; CS requires its comparison strata.
These checks also apply when anticipation removes previously usable pre-periods.

A future cohort is normalized to never-treated only when `g-A > Tmax`.
Future treatment whose anticipation window overlaps the sample retains its
actual cohort label and can contribute anticipatory estimates or clean
not-yet-treated controls. It has no post-actual-treatment contribution while
`g > Tmax`. At least one never-treated/reference cohort is still required.

Anticipation does not recode the user's treatment column, drop contaminated
observations from the entire data set, or change panel-completeness rules.
It changes which observations each estimator can use as clean comparisons.

## Event studies and aggregation

```python
# Require the same cohorts at all requested anticipation/post-treatment events.
window = fit.aggregate(balance_window=(-2, 1))
print(window.contributions())
print(window.composition())

# Post-actual-treatment summaries:
fit.aggregate("simple").summary()
fit.aggregate("cohort").summary()
fit.aggregate("calendar").summary()
```

Dynamic aggregation includes supported anticipation cells. Simple, cohort/group
and calendar aggregation include actual post-treatment cells (`t >= g`) only.
Their default weights therefore exclude anticipation-window cells. Restricting
them with negative-only `horizons` leaves no supported cells and raises an error.

`balance_e` still balances post-actual-treatment exposures `0..E`. An explicit
`balance_window` can include anticipation and clean pre-periods. SA and
universal-CS windows exempt the structurally omitted event `-A-1`; event `-1`
is an estimated anticipation effect when `A>0`. BJS windows start at or after
`-A`. These rules preserve actual event-time labels even for future cohorts.
Balancing fixes contributing cohorts, not individual-unit identities.

`group_time()` and `contributions()` report `period_type` as `pre_treatment`,
`anticipation` or `post_treatment`. Use clean pre-treatment cells when inspecting
placebo leads; anticipation cells can contain genuine effects. Diagnostics
retain `anticipation` and `reference_event_time`. No pre-trend test runs automatically. `pretrend_test()` explicitly selects
clean leads and respects anticipation; see [inference](inference.md).

## Validation and reference details

Fixtures cover anticipation 1 and 2, balanced/unbalanced panels, conditional and
unconditional CS, both control populations and base conventions, and all three
CS methods. They check estimates, full covariance, dynamic effects and actual
post-treatment aggregates. SA is checked against `fixest::sunab(ref.p=-A-1)`.
BJS is checked against an independent full-unit-dummy R score oracle for every
averaging/leave-out setting, plus `didimputation` with effective-onset labels and
weights selecting actual post-treatment observations. Its first-stage fixed-
effect tolerance is set to `1e-10` for these package comparisons.

The installed R `did` 2.5.1 varying-base implementation switches its base at
actual treatment rather than affected onset inside multi-period anticipation
windows. Varying-base fixtures therefore call R with effective cohort labels
`g-A` and anticipation zero, then restore original group/event labels. Universal-
base fixtures use R's actual anticipation option directly. Post-treatment
aggregation references use actual cohort labels and anticipation. This detail
is recorded in fixture version metadata; the R installation is unchanged.
[The did reference](https://bcallaway11.github.io/did/reference/att_gt.html)
describes the clean reference and anticipation convention.

Regenerate with `benchmarks/external/anticipation_reference.R`. The BJS exporter
`benchmarks/external/bjs_reference.R input.csv output-directory A` accepts an
optional anticipation count. The input fixtures are `anticipation_panel.csv`
and `anticipation_unbalanced_panel.csv` under `tests/data`. Dense Python checks
also cover future cohorts, exact anticipated effects, normalization, input
validation, partition/sample modes and zero-option equivalence.

A separate local resource check uses 100,000 units, eight periods and 750,000
observed rows, A=2, a 1 GB decimal watchdog, and native unbalanced panels.
Conditional CS, SA and BJS (all-treated smart groups with leave-out) completed,
including actual-post and balanced-window aggregation and scratch cleanup.
Reproduce with `python benchmarks/anticipation_resources.py` from the installed
benchmark environment. Raw results live under
`benchmarks/results/anticipation_resources`; generation runs separately and is
excluded from fit budgets. These are finite resource checks, not a hard-memory
guarantee at larger period/design dimensions.

For BJS clean placebos, `pretrends=L` estimates actual event times
`-anticipation-L` through `-anticipation-1` in a separate clean-sample regression.
The last clean lead is estimated, without a single omitted reference event.
See [BJS placebos](bjs_placebos.md).
