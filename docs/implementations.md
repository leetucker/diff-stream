# Differences from other packages

The estimators reproduce the reference packages' estimates and standard errors
when given the same conventions (see [Validation](validation.md)). Defaults and
names differ in places; this page lists them.

## Names

| Here | R did | R fixest / didimputation / did2s / etwfe | Stata |
|---|---|---|---|
| `did_callaway_santanna` | `att_gt()` | | `csdid` |
| `did_sun_abraham` | | `feols(... sunab(cohort, period))` | `eventstudyinteract` |
| `did_bjs` | | `did_imputation()` | `did_imputation` |
| `did_two_stage` | | `did2s()` | `did2s` |
| `did_extended_twfe` | | `etwfe()` + `emfx()` | `jwdid` |
| `cohort=` (0 = never treated) | `gname` | the first argument of `sunab()`; `gname` (didimputation); `gvar` (etwfe) | `gvar` (csdid, jwdid); `Ei` (did_imputation) |
| `control_group="never_treated"`, `"not_yet_treated"` | `control_group="nevertreated"`, `"notyettreated"` | etwfe `cgroup="never"`, `"notyet"` | `notyet` (csdid) |
| `adjustment="regression"`, `"ipw"`, `"doubly_robust"` | `est_method="reg"`, `"ipw"`, `"dr"` | | `method(reg)`, `method(stdipw)`, `method(dripw)` (csdid) |
| `propensity_trim` | `trim.level` (DRDID) | | |
| `base_period="varying"`, `"universal"` | `base_period=` | | `long2` (csdid; universal) |
| `anticipation` | `anticipation` | | |
| `incomplete_units="keep"` | `allow_unbalanced_panel=TRUE` | | |
| `time=` (no `panel=`) | `panel=FALSE` | | |
| `aggregate("event_time")`, `"overall"`, `"cohort"`, `"calendar"` | `aggte(type="dynamic", "simple", "group", "calendar")` | | `estat event/simple/group/calendar` |
| `aggregate(balanced_horizon=e)` | `aggte(balance_e=e)` | | |
| `multiplier_bootstrap_reps` | `bstrap=TRUE, biters=` | | `wboot` (csdid) |
| `residual_averaging` | | | `avgeffectsby()` (did_imputation) |
| `leave_out` | | | `leaveout` (did_imputation) |
| `pretrend_leads=L` | | | `pretrends(L)` (did_imputation) |
| `event_study=False` | | did2s with a single treatment indicator in `second_stage` | |
| `treat_sample_as_fixed=True` | | emfx conditional (`vcov` of the fit) | |
| `fixed_effects="unit"` | | etwfe `ivar=` | |
| `vcov="CRV1"` | | fixest `ssc(adj=FALSE, cluster.adj=TRUE)` | |

## Conventions that differ

**Standard errors.** Every estimator clusters by unit. `vcov="CRV0"` (the
default) is the plain sum of influence-function outer products; `"CRV1"`
multiplies it by N/(N-1) only, without fixest's (N-1)/(N-K) regression
adjustment. Intervals are normal, not t. R did's default reports multiplier
bootstrap intervals; here they are opt-in (`multiplier_bootstrap_reps`), and
the analytic ones are the default.

**Callaway-Sant'Anna.** Matches R did with the same options. The doubly
robust estimator is the "traditional" one of DRDID, not the improved
locally efficient variant. Trimming drops high-score controls only, as DRDID
does. A comparison with few effective controls warns
(`LowEffectiveSampleWarning`) but is reported; R did reports it silently.

**Sun-Abraham.** Matches fixest's `sunab` with `ssc` turned off. Aggregation
weights cells by their observed treated units, as `sunab`'s aggregation does.

**BJS.** Matches the authors' Stata `did_imputation` with `minn(0)`: Stata
suppresses effects with fewer than 30 contributing units by default, this
package does not (it reports them). `leave_out` and `residual_averaging` are
the authors' `leaveout` and `avgeffectsby`. Treated observations that cannot
be imputed (a fixed-effect level with no untreated observations) are dropped
with a warning, as Stata's `autosample` option does; Stata's default is an
error, and R's didimputation drops them silently.

**Two-stage.** Matches the authors' did2s. The event study and the static ATT
are separate fits; there is no aggregation of cohort-period effects.

**Extended TWFE.** Matches etwfe and emfx. The default comparison is the
never-treated units, as for the other estimators here; etwfe's is
not-yet-treated (`cgroup = "notyet"`), so set `control_group` explicitly when
comparing. Only the linear model is implemented.

**Continuous treatment.** `did_continuous` matches the authors' R contdid 0.1.1 in
its point estimates, with these differences:

- *The dose-response curves.* contdid fits each cohort-period spline with boundary
  knots at that cell's range of doses but evaluates the curves with boundary knots at
  the range of `dvals`, so its ATT(d) and ACRT(d) are off unless every cell's doses
  span exactly the grid ([contdid issue #16](https://github.com/bcallaway11/contdid/issues/16),
  fix proposed in [PR #18](https://github.com/bcallaway11/contdid/pull/18)). Here the
  spline has its boundary at the range of all treated doses, for fitting and
  evaluating alike, and doses outside that range are an error.
- *Options contdid ignores.* contdid 0.1.1 accepts `control_group`, `base_period`
  and weights but does not apply them: it always uses not-yet-treated comparisons, a
  varying base period and no weights
  ([contdid issue #19](https://github.com/bcallaway11/contdid/issues/19)). Here both
  comparisons and both base periods are applied. Weights are not supported.
- *Standard errors.* Three problems in contdid and in ptetools, which it uses for
  inference, make its standard errors differ from these. The estimates are not
  affected.
  - Its ACRT standard errors (overall, cohort-period and event study) are too small
    by the share of treated units in each cell
    ([contdid issue #13](https://github.com/bcallaway11/contdid/issues/13), fix
    proposed in [PR #17](https://github.com/bcallaway11/contdid/pull/17)). On the
    validation panel they were about a third of these.
  - Its ATT(d) standard errors reverse the covariance between a unit's part as a
    treated unit in one cohort-period and as a comparison unit in another (with
    not-yet-treated comparisons): ptetools' `process_dose_gt` gives the comparison
    units' part the wrong sign. On the validation panel they were 1% to 4%
    smaller than these.
  - Its overall ATT and event-study standard errors use a term for the estimated
    cohort shares that differs from R did's, which these follow: ptetools'
    `pte_aggte` centers each cohort indicator at the cohort's share of the treated
    units rather than of all units. The differences were under 0.2%.
- *The overall ATT* is a cohort-weighted average of post-treatment cells, R did's
  `aggte(type = "group")`. `aggregate("overall")` weights cells by their units
  instead, as for Callaway-Sant'Anna.
- *The level event study.* contdid 0.1.1 computes it by passing the dose to
  `ptetools::did_attgt`, which DRDID 1.4.0 rejects; it is Callaway-Sant'Anna on being
  treated, and is checked against R did.

**Switching treatment.** `did_dcdh` matches the authors' R DIDmultiplegtDYN 2.4.0
(`did_multiplegt_dyn`) in its estimates, standard errors and joint tests, with these
differences:

- *The covariance matrix.* `did_multiplegt_dyn` returns a covariance matrix
  (`coef$vcov`) whose off-diagonal entries use the first group's value for every
  group, so they are not covariances: correlations can exceed one, and the matrix can
  have negative eigenvalues. Its own joint tests (`p_jointplacebo`,
  `p_equality_effects`) are computed correctly, and agree with `pretrend_test()` and
  `effects_equal_test()` here. `fit.vcov` here is the covariance those tests use. Pass
  it, not `coef$vcov`, to sensitivity analyses such as HonestDiD.
- *Missing treatments.* `did_multiplegt_dyn` accepts rows with an outcome but no
  treatment and fills the treatment in by its own rules. Here such rows are an error;
  missing periods (no row) are supported, with `incomplete_units="keep"`, as
  `did_multiplegt_dyn` treats them.
- *Linear trends with missing periods.* With `trends_lin`, `did_multiplegt_dyn` takes
  each row's first difference from the group's previous row, so across a missing
  period it uses a two-period difference as if it were one. `linear_trends=True` here
  uses a first difference only where both of its periods are observed. In balanced
  panels the two agree.
- *Switcher counts with linear trends.* With `trends_lin`, the number of switchers
  `did_multiplegt_dyn` reports for an effect ("Switchers") is that of the last
  estimate it computed for the same horizon (with 4 effects, every effect shows the
  switchers of effect 4). Its weighted count ("Switchers.w") and its estimates use the
  right switchers, and `switchers()` here reports them.
- *Names.* `trends_nonparam` is `trends_by`, `trends_lin` is `linear_trends`, and
  `weight` is `weights` (with `allow_time_varying_weights=True` for weights that
  change over time).
- *Options not implemented:* `continuous`,
  `cluster` (standard errors are clustered by group), `same_switchers`,
  `same_switchers_pl`, `less_conservative_se`, `more_granular_demeaning`,
  `dont_drop_larger_lower`, `drop_if_d_miss_before_first_switch`, `date_first_switch`,
  `by`, `by_path`, `predict_het` and `bootstrap`.

**Aggregation.** `horizons` filters event times for every kind of
aggregation, including calendar, where R did ignores its event-time limits.

**Bootstrap.** `refit_bootstrap` refits the whole estimator on unit
resamples; percentile intervals use R's `quantile(type = 7)` and the normal
interval has no bias correction, so endpoints can differ from R's
`boot.ci`. Simultaneous bands from refits are not implemented (the multiplier
bootstrap gives them for Callaway-Sant'Anna).

**Diagnostics.** `bacon_decomp` matches R's bacondecomp; `twfe_weights`
matches the `feTR` type of the authors' TwoWayFEWeights. The other types
(`feS`, first-difference) and continuous treatments are not implemented.

## Not implemented

- For a switching treatment (`did_dcdh`): a continuous first-period treatment and
  clustering above the group; de Chaisemartin and D'Haultfoeuille's
  `did_multiplegt_dyn` (R, Stata) has them.
- Nonlinear ETWFE (logit, Poisson): R's etwfe, Stata's `jwdid`.
- Fixed effects interacted with the unit (unit-specific trends).
- Sensitivity analysis for parallel trends: pass `fit.aggregate().beta` and
  `.vcov` to R's HonestDiD.
