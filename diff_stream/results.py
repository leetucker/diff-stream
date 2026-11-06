"""Family-compatible result methods and explicit aggregation maps."""

from copy import copy
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from hdfe_stream.core.results_base import RegressionResult, _select_coefs

from ._core.multiplier import multiplier_band
from .aggregation import period_type, plan
from .bjs import BJSInference
from .bjs_placebo import BJSPlaceboResult


@dataclass(kw_only=True)
class DIDResult(RegressionResult):
    """Group-time estimates and their joint covariance.

    Inherits tidy, confint, wald_test, summary, summary_json and etable support.
    ``aggregate`` returns event-time, cohort, calendar-time or simple effects.
    CS additionally retains covariance blocks accounting for estimation of
    cohort shares; no full unit-by-effect array is retained.
    """

    estimator: str
    cells: list
    counts: np.ndarray
    n_units: int
    share_cross: np.ndarray | None = field(default=None, repr=False)
    share_cov: np.ndarray | None = field(default=None, repr=False)
    bjs_inference: BJSInference | None = field(default=None, repr=False)
    observed_counts: np.ndarray | None = field(default=None, repr=False)
    cohort_counts: np.ndarray | None = field(default=None, repr=False)
    _contributor_table: pl.DataFrame | None = field(default=None, repr=False)
    _composition_table: pl.DataFrame | None = field(default=None, repr=False)
    _multiplier_draws: np.ndarray | None = field(default=None, repr=False)
    _multiplier_shares: np.ndarray | None = field(default=None, repr=False)
    _bjs_placebos: BJSPlaceboResult | None = field(default=None, repr=False)
    _balance_table: pl.DataFrame | None = field(default=None, repr=False)
    _ess_table: pl.DataFrame | None = field(default=None, repr=False)
    _suppression_mask: np.ndarray | None = field(default=None, repr=False)
    _suppression_source: object = field(default=None, repr=False)
    _normal_inference = True

    def covariate_balance(self, *, cohort=None, time=None):
        """Copy saved CS before/after propensity-weight balance diagnostics.

        Request balance=True at fitting. Regression-only fits report before
        weighting; no artificial regression-adjusted covariate sample is defined.
        Native unbalanced fits report pre/post date strata separately. Baseline
        pooled SD scales are held fixed after weighting. These are descriptive
        diagnostics, not tests of parallel trends or a DR identification check.
        """
        if self._balance_table is None:
            raise ValueError("fit CS with balance=True to save covariate balance diagnostics")
        table = self._balance_table
        if cohort is not None:
            table = table.filter(pl.col("cohort") == cohort)
        if time is not None:
            table = table.filter(pl.col("time") == time)
        return table.clone()

    def effective_sample_size(self):
        """CS Kish ESS by required comparison stratum, including aggregate contributors.

        Sampling weights define treated ESS and regression controls; IPW/DR
        controls include propensity odds and trimming. Native unbalanced panels
        report pre/post strata separately. This is not regression-leverage or
        influence-score ESS. Aggregates retain component ESS, never a pooled ESS.
        """
        if self._ess_table is None:
            raise ValueError("effective_sample_size is currently available for CS")
        return self._ess_table.clone()

    def _minimum_ess(self):
        table = (
            self.effective_sample_size().group_by("coefficient").agg(pl.col("effective_n").min())
        )
        values = dict(table.iter_rows())
        return np.array([values[name] for name in self.coefnames])

    def suppress(self, min_effective_n=30):
        """Copy with low-support CS coefficients and covariance rows/columns as NaN.

        Uses the minimum Kish ESS among each coefficient's required component
        strata. Existing suppression is retained. No refitting, dropping cells,
        renormalizing aggregation weights or changing the original fit occurs.
        Aggregates depending on a suppressed cell are also suppressed. Joint
        inference must explicitly select unsuppressed coefficients.
        """
        from .effective_sample import threshold

        limit = threshold(min_effective_n, "min_effective_n", optional=False)
        return self._apply_suppression(self._minimum_ess() < limit, limit)

    def _apply_suppression(self, mask, limit):
        out = copy(self)
        mask = np.array(mask, dtype=bool, copy=True)
        if self._suppression_mask is not None:
            mask |= self._suppression_mask
        out._suppression_source = self._suppression_source or self
        out._suppression_mask = mask
        out.beta, out.vcov = self.beta.copy(), self.vcov.copy()
        out.beta[mask] = np.nan
        out.vcov[mask, :] = np.nan
        out.vcov[:, mask] = np.nan
        out.all_vcovs = {out.vcov_type: (out.vcov, out.df_t)}
        if self._multiplier_draws is not None:
            out._multiplier_draws = self._multiplier_draws.copy()
            out._multiplier_draws[:, mask] = np.nan
        names = [name for name, bad in zip(self.coefnames, mask, strict=True) if bad]
        out.diagnostics = {
            **self.diagnostics,
            "suppressed_coefficients": names,
            "suppression_threshold": max(limit, self.diagnostics.get("suppression_threshold", 0)),
            "suppression_rule": "minimum component-stratum ESS; aggregates retain original weights",
        }
        if self._contributor_table is not None:
            out._contributor_table = self._contributor_table.with_columns(
                pl.when(pl.col("aggregate").is_in(names))
                .then(float("nan"))
                .otherwise(pl.col(column))
                .alias(column)
                for column in ("estimate", "weighted_estimate")
            )
        return out

    def confint(
        self,
        alpha=0.05,
        keep=None,
        drop=None,
        exact_match=False,
        joint=False,
        reps=10000,
        seed=None,
    ):
        if joint and self._suppression_mask is not None:
            ix = _select_coefs(self.coefnames, keep, drop, exact_match)
            if self._suppression_mask[ix].any():
                raise ValueError(
                    "joint inference includes suppressed coefficients; select unsuppressed coefficients explicitly"
                )
        return super().confint(
            alpha=alpha,
            keep=keep,
            drop=drop,
            exact_match=exact_match,
            joint=joint,
            reps=reps,
            seed=seed,
        )

    def wald_test(self, R=None, q=None, distribution=None, keep=None, drop=None, exact_match=False):
        if self._suppression_mask is not None and self._suppression_mask.any():
            if R is None:
                ix = _select_coefs(self.coefnames, keep, drop, exact_match)
            else:
                restriction = np.atleast_2d(np.asarray(R, dtype=float))
                if restriction.shape[1] != len(self.coefnames):
                    raise ValueError("R must have one column per coefficient")
                ix = np.flatnonzero(np.any(restriction != 0, axis=0))
            if self._suppression_mask[ix].any():
                raise ValueError(
                    "Wald test includes suppressed coefficients; select unsuppressed coefficients explicitly"
                )
            source = self._suppression_source
            if source.vcov_type != self.vcov_type:
                source = source.with_vcov(self.vcov_type)
            return source.wald_test(
                R=R, q=q, distribution=distribution, keep=keep, drop=drop, exact_match=exact_match
            )
        return super().wald_test(
            R=R, q=q, distribution=distribution, keep=keep, drop=drop, exact_match=exact_match
        )

    def pretrend_test(self, *, level="auto", horizons=None):
        """Joint chi-squared Wald test of clean pre-treatment effects.

        By default test group-time cells on an original CS/SA fit, or horizons
        on a dynamic aggregate. level='dynamic' aggregates an original fit
        first. Anticipation periods and omitted references are excluded.
        Singular covariance raises. BJS tests its separate clean-lead fit,
        requested at fit time with pretrends=L; only pooled dynamic tests apply.
        The returned dictionary lists the exact coefficients, cells and horizons.
        """
        from .pretrend import pretrend_test

        return pretrend_test(self, level=level, horizons=horizons)

    def placebos(self):
        """Separate BJS clean-lead regression result, fitted with pretrends=L.

        Available after cleanup and on treatment-effect aggregates. Lead
        coefficients are pooled WLS regression effects, not group-time ATTs.
        No joint covariance with the main treatment estimates is estimated.
        """
        if self.estimator != "bjs":
            raise ValueError("placebos() is available only for BJS")
        if self._bjs_placebos is None:
            raise ValueError("fit BJS with pretrends=L to estimate clean placebo leads")
        return self._bjs_placebos

    def multiplier_draws(self, *, keep=None, drop=None, exact_match=False):
        """Copy saved CS unit-multiplier perturbations (repetitions x effects).

        Fit with bootstrap_reps>=2. Draws include share uncertainty after
        aggregation and the selected CRV0/CRV1 scaling; no data are reread.
        """
        if self._multiplier_draws is None:
            raise ValueError("fit CS with bootstrap_reps>=2 to retain unit multiplier draws")
        ix = _select_coefs(self.coefnames, keep, drop, exact_match)
        factor = np.sqrt(self.n_units / (self.n_units - 1)) if self.vcov_type == "CRV1" else 1.0
        return self._multiplier_draws[:, ix] * factor

    def multiplier_confint(
        self,
        alpha=0.05,
        *,
        keep=None,
        drop=None,
        exact_match=False,
        joint=True,
        studentization="bootstrap",
    ):
        """CS unit-multiplier intervals, simultaneous by default.

        Bootstrap studentization uses the draw IQR and empirical type-1
        quantiles, as R did. 'analytic' uses the stored analytic standard
        errors instead. The result's vcov/se and inherited confint stay analytic.
        Exactly the keep/drop selection defines the simultaneous family.
        """
        if studentization not in ("bootstrap", "analytic"):
            raise ValueError("studentization must be 'bootstrap' or 'analytic'")
        ix = _select_coefs(self.coefnames, keep, drop, exact_match)
        if self._suppression_mask is not None and self._suppression_mask[ix].any():
            raise ValueError(
                "multiplier inference includes suppressed coefficients; select unsuppressed coefficients explicitly"
            )
        draws = self.multiplier_draws(keep=keep, drop=drop, exact_match=exact_match)
        se, critical = multiplier_band(
            draws,
            alpha,
            joint=joint,
            analytic_se=self.se[ix] if studentization == "analytic" else None,
        )
        beta = self.beta[ix]
        return pl.DataFrame(
            {
                "Coefficient": [self.coefnames[i] for i in ix],
                "Estimate": beta,
                "Std. Error": se,
                "critical_value": np.broadcast_to(critical, len(ix)),
                f"{alpha / 2 * 100:.1f}%": beta - critical * se,
                f"{(1 - alpha / 2) * 100:.1f}%": beta + critical * se,
            }
        )

    def cleanup(self):
        """Release optional weighted/native unbalanced BJS aggregation files.

        Estimates, stored covariance and coefficient tables remain available.
        Further BJS aggregation requires refitting. Shallow copies made by
        with_vcov share ownership of these files.
        """
        if self.bjs_inference is not None and hasattr(self.bjs_inference, "cleanup"):
            self.bjs_inference.cleanup()

    def with_vcov(self, vcov):
        """Copy using CRV0 or CRV1 (the latter applies N/(N-1) only)."""
        if vcov not in ("CRV0", "CRV1"):
            raise ValueError("vcov must be 'CRV0' or 'CRV1'")
        factor = self.n_units / (self.n_units - 1)
        ratio = (factor if vcov == "CRV1" else 1) / (factor if self.vcov_type == "CRV1" else 1)
        out = copy(self)
        out.vcov = self.vcov * ratio
        out.vcov_type = vcov
        out.all_vcovs = {vcov: (out.vcov, out.df_t)}
        if self.share_cross is not None:
            out.share_cross = self.share_cross * ratio
            out.share_cov = self.share_cov * ratio
        if self._bjs_placebos is not None:
            out._bjs_placebos = self._bjs_placebos.with_vcov(vcov)
        return out

    def group_time(self):
        """Coefficient table with cohort/time and weighting/observed/cohort counts."""
        if not self.cells:
            raise ValueError("group_time is available on unaggregated results")
        units = self.cohort_counts if self.estimator == "cs" else self.observed_counts
        units = self.counts if units is None else units
        return self.tidy().with_columns(
            *(
                [
                    pl.Series("min_effective_n", self._minimum_ess()),
                    pl.Series(
                        "suppressed",
                        self._suppression_mask
                        if self._suppression_mask is not None
                        else np.zeros(len(self.beta), dtype=bool),
                    ),
                ]
                if self._ess_table is not None
                else []
            ),
            pl.Series("cohort", [c[0] for c in self.cells]),
            pl.Series("time", [c[1] for c in self.cells]),
            pl.Series("event_time", [c[1] - c[0] for c in self.cells]),
            pl.Series(
                "period_type",
                [
                    period_type(t - g, self.diagnostics.get("anticipation", 0))
                    for g, t in self.cells
                ],
            ),
            pl.Series("n_units", units.astype(int)),
            pl.Series("weight_units", units.astype(int)),
            pl.Series("weight_total", self.counts * self.diagnostics.get("weight_scale", 1.0)),
            pl.Series(
                "observed_units",
                self.observed_counts.astype(int)
                if self.observed_counts is not None
                else [None] * len(self.cells),
                dtype=pl.Int64,
            ),
            pl.Series(
                "cohort_units",
                self.cohort_counts.astype(int)
                if self.cohort_counts is not None
                else [None] * len(self.cells),
                dtype=pl.Int64,
            ),
        )

    def contributions(self, kind=None, *, horizons=None, balance_e=None, balance_window=None):
        """Cell weights, estimates and sample counts for each aggregate.

        On an original result, arguments match aggregate(). On an aggregate,
        call without arguments to inspect its saved contributors. Inspection
        does not reread BJS aggregation files and works after cleanup().
        """
        return self._composition_view(False, kind, horizons, balance_e, balance_window)

    def composition(self, kind=None, *, horizons=None, balance_e=None, balance_window=None):
        """Cohort membership, cohort weights and sample totals per aggregate.

        observed_unit_periods sums target-cell observations; it is not a count
        of unique units across dates, nor a treated/control comparison sample.
        """
        return self._composition_view(True, kind, horizons, balance_e, balance_window)

    def _composition_view(self, summary, kind, horizons, balance_e, balance_window):
        if self.cells:
            p = plan(self, "dynamic" if kind is None else kind, horizons, balance_e, balance_window)
            return p.composition if summary else p.contributors
        if any(x is not None for x in (kind, horizons, balance_e, balance_window)):
            raise ValueError("inspect an aggregated result without aggregation arguments")
        table = self._composition_table if summary else self._contributor_table
        if table is None:
            raise ValueError("no aggregation contributors are available")
        return table.clone()

    def aggregate(self, kind="dynamic", *, horizons=None, balance_e=None, balance_window=None):
        """Aggregate by event time, post-treatment cohort, calendar time or ATT.

        kind is dynamic (default), cohort (alias group), calendar, or simple.
        horizons selects event times for every kind; non-dynamic kinds use only
        post-treatment cells. CS weights by retained cohort mass; SA/BJS by
        observed cell mass (unit counts without sampling weights).
        CS propagates estimated-share uncertainty, and BJS
        recomputes contrast-specific smart residual/leave-out inference.

        Dynamic balance_e=E requires supported events 0..E and retains R did's
        event-time range [E-(Tmax-Tmin), E]. balance_window=(lo,hi) requires every
        estimated event in that window in each retained cohort; omitted reference
        events -(anticipation+1) are exempt. These restrictions balance cohorts, not the
        identities of observed units. They are applied before horizons.
        """
        if self._suppression_source is not None:
            source = self._suppression_source
            if source.vcov_type != self.vcov_type:
                source = source.with_vcov(self.vcov_type)
            p = plan(source, kind, horizons, balance_e, balance_window)
            out = source.aggregate(
                kind, horizons=horizons, balance_e=balance_e, balance_window=balance_window
            )
            mask = np.any((p.weights != 0) & self._suppression_mask[None, :], axis=1)
            return out._apply_suppression(mask, self.diagnostics["suppression_threshold"])
        p = plan(self, kind, horizons, balance_e, balance_window)
        A = p.weights
        B = np.zeros_like(A)
        for j in range(len(p.names)):
            ix = A[j] != 0
            B[j, ix] = (self.beta[ix] - A[j] @ self.beta) / self.counts[ix].sum()
        if self.bjs_inference is not None:
            V = self.bjs_inference.covariance(A)
            if self.vcov_type == "CRV1":
                V *= self.n_units / (self.n_units - 1)
        else:
            V = A @ self.vcov @ A.T
        if self.share_cross is not None:
            cross = A @ self.share_cross @ B.T
            V += cross + cross.T + B @ self.share_cov @ B.T
        draws = None
        if self._multiplier_draws is not None:
            draws = self._multiplier_draws @ A.T
            if self._multiplier_shares is not None:
                draws += self._multiplier_shares @ B.T
        return DIDResult(
            fml=self.fml,
            depvar=self.depvar,
            coefnames=p.names,
            beta=A @ self.beta,
            vcov=V,
            vcov_type=self.vcov_type,
            df_t=self.df_t,
            n_obs=self.n_obs,
            n_clusters=self.n_clusters,
            estimator=self.estimator,
            cells=[],
            counts=np.array([]),
            n_units=self.n_units,
            _balance_table=(
                self._balance_table.join(
                    pl.DataFrame(
                        {
                            "cohort": [g for j, (g, t) in enumerate(self.cells) if np.any(A[:, j])],
                            "time": [t for j, (g, t) in enumerate(self.cells) if np.any(A[:, j])],
                        }
                    ),
                    on=["cohort", "time"],
                    how="inner",
                )
                if self._balance_table is not None
                else None
            ),
            _multiplier_draws=draws,
            _bjs_placebos=self._bjs_placebos,
            _ess_table=(
                p.contributors.select(pl.col("aggregate").alias("coefficient"), "cell").join(
                    self._ess_table.drop("coefficient"), on="cell", how="inner"
                )
                if self._ess_table is not None
                else None
            ),
            diagnostics={
                **self.diagnostics,
                **p.diagnostics,
                "warnings": list(self.diagnostics.get("warnings", []))
                + (
                    [
                        "Cohort contributors change across event times; inspect composition() or select a balanced window."
                    ]
                    if p.kind == "dynamic" and p.diagnostics["cohort_composition_changes"]
                    else ["Cohort weights vary across event times; inspect composition()."]
                    if p.kind == "dynamic" and p.diagnostics["cohort_weights_change"]
                    else []
                ),
            },
            _contributor_table=p.contributors,
            _composition_table=p.composition,
        )

    def _summary_fields(self):
        return {
            **super()._summary_fields(),
            "estimator": self.estimator,
            "n_units": self.n_units,
            "diagnostics": self.diagnostics,
        }

    def _summary_lines(self):
        return [
            f"estimator: {self.estimator}; units: {self.n_units:,}",
            *super()._summary_lines(),
            self.diagnostics.get("inference", ""),
            *(
                [
                    f"suppressed coefficients: {len(self.diagnostics['suppressed_coefficients'])}; original aggregation weights retained"
                ]
                if self.diagnostics.get("suppressed_coefficients")
                else []
            ),
            *self.diagnostics.get("warnings", []),
        ]
