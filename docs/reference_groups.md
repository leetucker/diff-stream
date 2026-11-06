# Never-treated and last-treated references

CS, Sun–Abraham, BJS, Gardner and linear ETWFE support panels in which every
observed cohort eventually adopts treatment, using an explicit clean reference
window when needed.

```python
from diff_stream import cs_stream, sunab_stream

fit = cs_stream(
    "y ~ x + z",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    reference_group="auto",
    control_group="notyettreated",
    anticipation=1,
)
fit.group_time()
fit.aggregate().tidy()
fit.diagnostics["reference_cohort"]
fit.diagnostics["reference_cutoff"]
```

All five entry points (and `diff_stream`) accept `reference_group`. It is
independent of CS/ETWFE's `control_group`:

| Policy | Behavior |
|---|---|
| `"auto"` (default) | Use existing never-treated controls if available; otherwise use the last-treated cohort and its clean reference window. |
| `"nevertreated"` | Require clean never-treated observations; refuse a sample without them. |
| `"lasttreated"` | Select the last-treated cohort explicitly and impose its clean window. Existing never-treated units remain additional clean controls. |

With a never-treated group, the default retains the previous sample and results.
Without one, auto now issues `ReferenceWindowWarning` and uses the last cohort
as a comparison group only. To retain the earlier strict refusal behavior,
request `reference_group="nevertreated"`.

## Identification and sample selection

If the last cohort adopts at `g_last` and anticipation is `A`, estimation retains
**only dates `t < g_last-A`**. The reference cohort has no estimated treatment
coefficients. Earlier cohorts retain their actual adoption labels/event times.
With CS `control_group="notyettreated"`, earlier cohorts that are still clean
can also supply comparisons; with `"nevertreated"`, the designated reference
and any original never-treated units supply the fixed control pool.

BJS and Gardner use all clean observations in their first stage; choosing the
last reference does not turn those models into pairwise CS comparisons. ETWFE
retains its separate never/not-yet-treated regression convention on the selected
sample. The last cohort has no post-treatment effects under this policy.

This convention does not identify effects after the reference cohort becomes
affected. Two treated cohorts and at least two retained dates are required;
ordinary cohort/date support and regression identification checks still apply.
Future cohorts already normalized to never-treated by the library's existing
clean-window rule continue to act as clean controls.
Thus `nevertreated` follows the existing full-window normalization convention;
it does not require controls to remain untreated beyond the observed sample.

The original `balanced`, `complete` or `unbalanced` panel policy is applied
**before** selecting the reference cohort. Thus complete-panel dropping can
change which cohort is last. The date restriction can remove units with no
remaining observations in an unbalanced panel. Weights and unit-weight anchors
are recomputed on the retained reference-window sample. Complete-panel policy
uses completeness in the original time grid, rather than rescuing units merely
because their missing dates are outside the retained window.

Diagnostics record the selected reference, cutoff, original input counts,
excluded rows/units and retained sample limits. Cohort 0 internally pools the
reference observations; actual treatment-cell labels are unchanged. Selecting
last-treated in the presence of never-treated units includes both in this pool.
The restricted panel is restaged in bounded partitions; validated model columns
are reused, and no global table of unit IDs is introduced.

The date restriction is not sufficient for every model. BJS/Gardner still require
an identified clean first-stage design and clean observations for retained units.
Extra crossed FEs can leave unsupported subgroup/date effects; sparse unbalanced
panels can lack the required clean-date connectivity. ETWFE must retain identified
reported margins. These conditions still raise errors instead of extrapolating
effects or silently trimming additional unsupported cells.

## BJS placebos and Gardner dynamic references

BJS `pretrends=L` uses **original known adoption dates**, including those of the
last reference and future cohorts, for its separate clean-placebo regression.
Internally pooling their nuisance-reference observations into cohort 0 does not
erase those dates. `fit.placebos().support()` exposes the actual cohort/date
support, and diagnostics record `placebo_cohort_policy`. No post-treatment
effect is estimated for the reference cohort.

Gardner assigns the designated reference cohort zero second-stage design,
including its leads. Its dynamic coefficients therefore describe earlier target
cohorts. `reference_period` still chooses the omitted clean event time within
those cohorts; `reference_group` chooses the reference population/window. These
are different options. This matches an explicitly restricted/relabelled authors'
R fit, rather than an unrestricted event study pooling the last cohort's leads.

```python
from diff_stream import bjs_stream, gardner_stream, etwfe_stream

bjs = bjs_stream(
    "y ~ x",
    data,
    panel=("id", "time"),
    cohort="first_treat",
    reference_group="auto",
    pretrends=2,
)
bjs.placebos().support()
bjs.cleanup()  # releases retained aggregation scratch when applicable

gardner = gardner_stream(
    "y ~ x",
    data,
    panel=("id", "time"),
    cohort="first_treat",
    reference_group="lasttreated",
    specification="dynamic",
)
etwfe = etwfe_stream(
    "y ~ x",
    data,
    panel=("id", "time"),
    cohort="first_treat",
    reference_group="lasttreated",
    control_group="notyettreated",
)
```

`profile=True` stages/validates the same selected sample and includes reference
metadata in the profile notes. The standalone repository example
`examples/reference_populations.py` demonstrates all three new entry-point options.

## Validation

The frozen external matrix contains 96 CS and 12 SA specifications covering
balanced/unbalanced panels, anticipation 0/1, time-varying sampling weights,
both CS control/base conventions, all three CS estimation methods and richer
SA fixed effects. Coefficients and full cell covariance are checked against
actual R `did`/`fixest` commands on the explicitly restricted sample. CS dynamic
aggregation is checked throughout, and all aggregation types at anticipation 0.
Actual-adoption aggregation at positive anticipation is separately checked
against an explicitly restricted/relabelled fit.

R references shift adoption to the affected onset while fitting: otherwise an
actual adoption date outside the restricted calendar can be treated as a
never-treated sentinel despite anticipatory effects inside that calendar.
Original labels are restored in the exports.

Native R propensity fits and precision confirmations are both retained. The
confirmation changes only DRDID's in-process propensity stopping tolerance from
`1e-8` to `1e-12` and maximum iterations from 100 to 1000; installed R files are
unchanged. Tests preserve a native-output comparison and stricter precision
comparison. Source/input/output hashes and R sessions are in
`tests/data/reference_groups/PROVENANCE.json`. Run
`python benchmarks/reference_groups_validation.py` for the offline audit;
`--regenerate` reruns both R modes using the local validation installation.

References: [R did comparison populations](https://bcallaway11.github.io/did/reference/att_gt.html)
and [fixest SA reference cohorts](https://lrberge.github.io/fixest/reference/sunab.html).
The [BJS/Gardner/ETWFE reference-population review](reference_population_review.md)
records the methodological audit preceding the completed extension. Its lead
conventions are implemented as described above.

The initial CS/SA audit's largest native-R differences were `1.52e-8` in effects
and `2.56e-9` in cell covariance. Precision confirmations reduced these maxima
to `2.62e-11` and `3.95e-12`. All 149 focused reference/balance tests also pass
on Polars 1.37. The existing suite passed 1,439 tests with one kernel-only skip;
its one obsolete default-refusal assertion was updated to request the explicit
strict policy and passes separately.

Four local fresh-process probes completed under a 1.5 GB WSL/Linux cgroup v1
cap, with no estimation OOMs and clean scratch removal. The largest retained
sample had 783,653 units and 3,170,288 observed rows: peak RSS was about 540 MiB
for covariate-adjusted CS and 550 MiB for richer SA. Optional CS balance completed
78,504 retained units/317,466 rows near 482 MiB; it saved 144 reduced diagnostic
rows. These individual observations establish local feasibility rather than a
general memory bound. Raw kernel counters and source/data hashes are in
`benchmarks/results/reference_balance_resources`. Reproduce with
`python benchmarks/reference_balance_resources.py`, adding an explicit
`--cgroup-parent` for Linux kernel enforcement.

### BJS/Gardner/ETWFE extension validation

The new frozen matrix has 48 configurations: 40 main-fit comparisons (BJS dynamic
points, Gardner static/dynamic points and full covariance, and ETWFE conditional/
unconditional cell effects and full covariance), plus eight BJS clean-placebo
regressions. Those eight check native `didimputation` lead estimates/SEs and full
clean-lead covariance from `fixest`. No new native Stata inference run is claimed.
Original source pins, native outputs, inputs and hashes are in
`tests/data/reference_population_extensions/`.

Tests cover both ETWFE control conventions and cohort/unit FE, every common
aggregation, complete-panel ordering, pooled existing controls, disappearing
units, original BJS timing, strict/default policies, profiles and cleanup.
The full suite passes 1,693 tests with one kernel-only skip; all 219 focused
reference checks also pass on Polars 1.37. The shared constrained-resource
fixture passes with both its portable policy and delegated-kernel enforcement.

Run `python benchmarks/reference_population_validation.py` for the offline
audit. `--regenerate --rscript /path/to/Rscript` reruns pinned native sources
using the local R validation setup. Fresh-process scaling and kernel-counter
reports are recorded by `benchmarks/reference_population_resources.py`.

Six fresh-process extension probes completed under a verified 1.5 GB WSL/Linux
kernel cap, with no OOM kills or limit failures and clean scratch removal. Inputs
are unbalanced with date-varying weights and anticipation 1; BJS/Gardner include
region-by-date effects, BJS also computes two clean-placebo leads, and ETWFE uses
not-yet-treated controls. The driver shifts cached adoption dates by one period
to preserve their guaranteed clean observations under anticipation.

| Estimator | Retained units | Observed rows | Peak RSS MiB |
|---|---:|---:|---:|
| BJS with placebos | 7,886 | 31,791 | 507 |
| Gardner | 7,886 | 31,791 | 500 |
| ETWFE | 7,886 | 31,791 | 373 |
| BJS with placebos | 783,653 | 3,170,288 | 673 |
| Gardner | 783,653 | 3,170,288 | 663 |
| ETWFE | 783,653 | 3,170,288 | 535 |

These are finite local measurements, not a universal bound or speed comparison.
The driver explicitly requests sampled RSS; kernel charged-memory peaks are
recorded separately and need not equal RSS. Production and launcher defaults
are unchanged by the resource harness. Reproduce with
`python benchmarks/reference_population_resources.py --cgroup-parent /path/to/delegated/parent`;
omit the cgroup option for the portable watchdog. Other OS CI jobs are configured
but have not been executed on this WSL host.
