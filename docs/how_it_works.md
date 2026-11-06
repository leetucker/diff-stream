# How it works

## Staging the panel

A fit reads the columns it needs once and checks them in batches: missing
values, the time grid, weights, and the levels of categorical covariates. It
then sorts the panel by unit and time and checks each unit (unique periods, a
constant cohort, complete periods if required), reducing it to cohort-period
sums on the way.

When the projected panel fits in the working memory (by default a quarter of
the memory left under `max_memory_gb`), it is sorted in one piece. When it does
not, it is split by a hash of the unit identifier into pieces on disk, each
small enough, recursively if a piece is still too large; every unit stays in
one piece, so no global sort and no table of all unit identifiers is needed.
Either way the sorted rows are written once to a temporary Parquet file, and
every later pass reads whole units from it in large batches.

## The estimators

- **Without covariates**, Callaway-Sant'Anna, Sun-Abraham and BJS reduce to
  cohort-period means: each effect is a contrast of them, computed exactly.
  A single pass then computes each unit's scores (a compiled kernel, in
  parallel over units) for the covariance.
- **Callaway-Sant'Anna with covariates** fits the outcome regressions and
  propensity logits of all comparisons together: one pass over the data for
  the regressions and one per Newton step for the logits, for every
  comparison at once. Comparisons with the same sample and covariate period
  share a model; outcome regressions on the same controls share a solve.
- **Sun-Abraham and BJS with covariates or fixed effects, two-stage and
  ETWFE** eliminate the unit effects within each batch of whole units and
  accumulate the small normal equations of the remaining design (time effects,
  covariates, extra fixed effects, treatment indicators), as matrix products.
- **BJS standard errors** need the residuals of each aggregate's averaging
  groups, so they are recomputed per aggregate: from cohort-level second
  moments in a balanced unweighted panel, by two more passes over the staged
  rows otherwise (which is why those results keep temporary files).

## The covariance

Standard errors are clustered by unit: each unit's influence-function
contributions are summed, and the covariance is the sum of their outer
products over units. No table of all units' scores is kept; the products are
accumulated batch by batch.

For Callaway-Sant'Anna, a unit's scores are zero outside the cells its cohort
enters (its own cells and those where it is a control), so the products are
accumulated cohort by cohort, each into its own block. The cohort-share
scores that aggregation adds depend on a unit only through its cohort and
weight, so their blocks need just per-cohort sums. With hundreds of
cohort-period cells this is most of the saving.

## Memory checks

Before each allocation that grows with the data or the model (the staging
buffers, each piece, the numerical batches), the fit checks that it fits in
the memory left under `max_memory_gb`, less a reserve, and stops with
`MemoryError` if not, before allocating. For Callaway-Sant'Anna, Sun-Abraham,
BJS and ETWFE, a batch size left at its default shrinks to fit instead. See
[Memory and threads](resources.md).
