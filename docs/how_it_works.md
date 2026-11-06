# How it works

A streaming projection first writes a Parquet snapshot. Bounded Arrow batches
check values and time support. If the decoded data exceed the preprocessing
allowance, unit hashes partition the snapshot on disk. Oversized partitions are
split recursively; every unit remains in one leaf. Each bounded leaf is sorted
by unit/time, checked for unique keys, sample policy and constant cohorts, reduced
to cohort-time sums, and saved for numerical passes. No global unit-ID grouping
or global sort is required. Future cohorts are normalized only after validating
the original cohort values.

For conditional CS, the same staged leaves include compiled covariate columns.
The decoded partition budget accounts for their expanded width. Numeric terms
use the existing family expression compiler without importing pyfixest's full
model-matrix machinery; categorical designs use its established coding on a
small synthetic frame. Generic partition/design adapters live temporarily in
`diff_stream._core`.

CS estimates each outcome change against its specified comparison population.
SA exploits an exact identity for saturated, unconditional balanced panels:
each coefficient is the cohort's change from g-1 less the never-treated change.
Their inference passes accumulate bounded unit-score batches. Numba parallelizes
scores and BLAS forms cross-products.

BJS eliminates unit effects on untreated observations and solves a T-1 time
system. Its inference pass collects cohort-specific, within-unit centered
second moments, retaining G×T×T values rather than all unit scores. First-stage
estimation weights and treated weights map these moments to cluster covariance.
A positive-semidefinite root avoids cancellation when residual variation is tiny.

Covariates, additional fixed effects, or time-varying row weights route SA/BJS
through a [richer weighted within-unit system](richer_models.md). Unit intercepts
are eliminated locally, remaining nuisance terms use the bounded family design
compiler, and SA interactions enter cohort-specific active blocks. BJS retains
sorted rows for contrast-specific inference. Batch-by-time-by-design and joint
system dimensions enter admission estimates; the specialized paths above remain
available for their original simpler designs.

BJS smart residual averages are recomputed for each requested contrast, with
optional cluster leave-out adjustments. Under unit clustering and equal weights,
the default cohort-time smart mean equals the ordinary cell mean. Coarser
averaging groups can yield a more conservative variance under heterogeneous
effects, and their aggregate covariance cannot generally be obtained by simply
transforming the cell covariance. See [inference conventions](implementations.md).

CS aggregation includes estimated cohort-share influences; SA/BJS condition on
treatment timing. Memory still grows with periods, cohorts, and the joint effect
covariance (effects squared), plus a small directory of disk partitions. Source
execution and runtime reader/writer pools have separate memory use.

Conditional CS fits control OLS and logit MLE nuisance systems in shared passes,
then accumulates nuisance-adjusted unit scores. Fits are reused only for the
same comparison sample and covariate date (or proven time-invariant design).
The generic nuisance and mean-contrast machinery lives in `_core`; DiD date and
population rules live in `diff_stream.cs`.

For [native unbalanced panels](unbalanced_panels.md), sorted leaves also retain
unit codes and dates. Only numerical batches are padded, never a whole sparse
partition. SA reparameterizes the saturated observed-row regression into small
within-cohort calendar-effect systems, with joint unit-score covariance.
CS uses date-specific normalized means, cached stratified OLS and date-pair
logits, then folds row influences within unit before cross-products. BJS solves
the locally demeaned untreated time system with each unit's actual history;
its result retains sorted Parquet and recomputes smart inference in two bounded
passes per aggregation. Calling `cleanup()` releases those aggregation files.

With [sampling weights](weights.md), the snapshot retains one numeric weight
column. Whole-unit leaves validate constancy and reduce weighted sums. CS fits
weighted nuisances and centers weighted cohort indicators for share inference.
SA uses weighted cohort systems. Weighted BJS uses the observed-row disk-backed
path even on balanced input, avoiding additional cohort moment tensors for
heterogeneous weights. Neither weights nor unit scores are retained in a global
unit-sized array.

The [Goodman–Bacon diagnostic](bacon.md) reuses the panel store's cohort-time
means and counts. It computes comparison estimates via prefix sums and weights
from cohort shares and block lengths, then checks reconstruction against an
independent grouped TWFE calculation. It needs no unit-level inference pass.
Numerical/output state grows with cohort-time cells and cohort pairs; the
comparison-table guard accounts for its quadratic output. With covariates, a
bounded global treatment projection and within-timing reduction replace the
two-block formulas; full-period dyads plus the within component reconstruct the
controlled regression. See [Bacon extensions](bacon_extensions.md).

[Gardner two-stage DiD](gardner.md) shares the weighted local unit elimination
and bounded row design. It fits the untreated outcome model, then regresses its
residualized outcome on disjoint static/event-time indicators. A cross-derivative
pass and corrected unit scores account for first-stage estimation. Raw shared
`ClusterScoreAccumulator` sums preserve the package's explicit CRV0/CRV1 policy.
All temporary files are released before the direct coefficient result returns.

[Linear ETWFE](etwfe.md) builds cohort/time/cell covariate interactions in bounded
batches, solves the saturated weighted regression and averages linear treatment
contrasts within each cohort/date. Cohort intercepts enter its bounded design;
unit intercepts, when selected, are eliminated locally. Unconditional unit
scores add empirical covariate-distribution variation and save small joint
blocks with cell-mass scores for aggregation; conditional mode uses regression
scores alone. Alias checks protect reported margins, rather than requiring
individual nuisance/interacted coefficients to be unique. The expanded-design
column guard and Gram/batch admission checks precede allocation.
