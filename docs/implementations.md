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

- Treatment that switches off, or continuous treatment, in the estimators:
  de Chaisemartin and D'Haultfoeuille's `did_multiplegt_dyn` covers them.
- Nonlinear ETWFE (logit, Poisson): R's etwfe, Stata's `jwdid`.
- Fixed effects interacted with the unit (unit-specific trends).
- More than 256 added design columns in two-stage. BJS has the same effect
  estimates and absorbs fixed effects with any number of levels (as does
  Sun-Abraham), with BJS's standard errors rather than Gardner's.
- Sensitivity analysis for parallel trends: pass `fit.aggregate().beta` and
  `.vcov` to R's HonestDiD.
