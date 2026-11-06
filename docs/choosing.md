# Choosing an estimator

With staggered timing, the conventional two-way fixed effects regression of
the outcome on a treatment dummy averages comparisons that use already-treated
units as controls, so with effects that change over time it can be far from
any average treatment effect, even of the wrong sign. The six estimators
here avoid those comparisons. They differ in which untreated observations
serve as the comparison and in how they use covariates.

| | Comparison for a cohort's effect in period t | Pre-treatment periods used | Covariates |
|---|---|---|---|
| `did_callaway_santanna` | the change since one base period, against never-treated or not-yet-treated units | one base period per comparison | regression, IPW or doubly robust; parallel trends conditional on covariates |
| `did_sun_abraham` | a regression with a coefficient for each cohort and period, against never-treated units | the period before treatment is the reference | linear, plus additional fixed effects |
| `did_bjs` | the outcome predicted from unit and time effects fitted on all untreated observations | all untreated periods | linear, plus additional fixed effects |
| `did_two_stage` | as BJS: the residual from that untreated model | all untreated periods | linear, plus additional fixed effects |
| `did_extended_twfe` | a single regression with a separate effect for every cohort and period | depends on the controls | interacted with cohort and time, so effects can vary with them |
| `did_stacked` | an event study on stacked sub-experiments, one per cohort, against units untreated through its window | the window's pre-treatment periods, with the period before treatment as reference | none |

## How they relate

These hold exactly in a balanced panel without covariates (the test suite
checks them):

- **Sun-Abraham** equals Callaway-Sant'Anna with never-treated controls and
  `base_period="universal"`.
- **Two-stage** has the same event-study coefficients as BJS imputation; their
  standard errors differ (Gardner's account for the first stage as a GMM
  system, BJS's are the authors' conservative ones).
- **Extended TWFE** with not-yet-treated controls has the same cohort-period
  effects as BJS imputation (Wooldridge 2021).

So the choice is mostly between two families: comparisons with one base period
(CS, SA), and imputation from all untreated periods (BJS, two-stage, ETWFE).

## Which to use

- **Covariates that make parallel trends plausible** (trends differ by
  observed characteristics): `did_callaway_santanna`, doubly robust. It is the
  only estimator here whose parallel-trends assumption is conditional on
  covariates rather than additive in them.
- **Parallel trends credible over the whole pre-period**: `did_bjs`. Using
  every untreated period makes it more precise, and it is efficient when the
  errors are homoskedastic and serially uncorrelated (Borusyak, Jaravel and
  Spiess 2024). `did_two_stage` gives the same estimates with Gardner's
  standard errors; `did_extended_twfe` the same in a regression form whose
  effects can vary with covariates.
- **Parallel trends credible only near treatment**: `did_callaway_santanna` or
  `did_sun_abraham`. They rely on the base period and the treatment periods
  only, not the whole pre-period.
- **A familiar regression event study that only uses clean comparisons**:
  `did_stacked`, with a fixed event window per cohort.
- **A regression with extra fixed effects** (region-by-year, industry-by-year):
  `did_sun_abraham`, `did_bjs` or `did_two_stage`; with many levels
  (county-by-year), `did_sun_abraham` or `did_bjs`.

Reporting two (one from each family) is common; a large difference between
them points at a pre-trend.

## Testing for pre-trends

Each estimator has a pre-trend test, `fit.pretrend_test()`, but they test
different things:

- CS and SA estimate pre-treatment effects directly (a cohort's change
  between two pre-treatment periods, against the controls), and the test is
  that they are jointly zero.
- BJS uses all pre-periods to fit its untreated model, so its pre-treatment
  "effects" are zero by construction. Request `pretrend_leads=L`: a separate
  regression on the untreated observations adds indicators for the last L
  periods before treatment, and the test is that they are zero.
- Two-stage estimates leads in its second stage; ETWFE with never-treated
  controls estimates pre-treatment effects.

Not rejecting a pre-trend does not show that trends are parallel. For
sensitivity analysis, see the note on HonestDiD on the [front page](index.md).

## Diagnosing a two-way fixed effects estimate

When a conventional estimate is already reported, the diagnostics show what
it averages:

- `bacon_decomp` writes it as a weighted average of every two-group,
  two-period comparison, including those with already-treated controls
  (Goodman-Bacon 2021).
- `twfe_weights` gives the weight of each treated cell, some possibly
  negative, and how much effect heterogeneity would reverse its sign
  (de Chaisemartin and D'Haultfoeuille 2020).

See [Diagnostics](diagnostics.md).

## References

- Borusyak, K., Jaravel, X. and Spiess, J. (2024). Revisiting event-study
  designs: robust and efficient estimation. *Review of Economic Studies*
  91(6), 3253-3285.
- Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences with
  multiple time periods. *Journal of Econometrics* 225(2), 200-230.
- de Chaisemartin, C. and D'Haultfoeuille, X. (2020). Two-way fixed effects
  estimators with heterogeneous treatment effects. *American Economic Review*
  110(9), 2964-2996.
- Gardner, J. (2022). Two-stage differences in differences. arXiv:2207.05943.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in
  treatment timing. *Journal of Econometrics* 225(2), 254-277.
- Wing, C., Freedman, S. M. and Hollingsworth, A. (2024). Stacked
  difference-in-differences. NBER Working Paper 32054.
- Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects in event
  studies with heterogeneous treatment effects. *Journal of Econometrics*
  225(2), 175-199.
- Wooldridge, J. M. (2021). Two-way fixed effects, the two-way Mundlak
  regression, and difference-in-differences estimators. SSRN 3906345.
