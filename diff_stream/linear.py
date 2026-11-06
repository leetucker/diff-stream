"""Covariate/extra-FE SA and BJS with local weighted unit elimination.

Unit intercepts never enter a global table. Other terms use the bounded family
model-matrix compiler; numerical state depends on columns and effect cells.
"""

import numpy as np

from ._core.nuisance import solve_checked
from ._core.within import center_groups, independent_columns, moments, row_sums, weighted_squares


def fixed_effect_terms(rhs, panel):
    """(covariates, additional fixed-effect terms as lists of column names)."""
    pieces = rhs.split("|")
    if len(pieces) != 2:
        raise ValueError("use one fixed-effect section; IV formulas are unsupported")
    covariates, effects = (p.strip() for p in pieces)
    if not effects:
        raise ValueError("the fixed-effect section cannot be empty")
    terms = []
    for term in effects.split("+"):
        names = [s.strip() for s in term.strip().split("^")]
        if any(not s.isidentifier() for s in names):
            raise ValueError(
                "additional fixed effects must be column names or name^name interactions"
            )
        if len(names) == 1 and names[0] in panel:
            continue  # Unit/time intercepts are always included.
        if panel[0] in names:
            raise ValueError("unit-interacted fixed effects are not supported")
        terms.append(names)
    return covariates or "1", terms


def regression_formula(rhs, panel):
    pieces = rhs.split("|")
    if len(pieces) != 2:
        raise ValueError("use one fixed-effect section; IV formulas are unsupported")
    covariates, effects = (p.strip() for p in pieces)
    if not effects:
        raise ValueError("the fixed-effect section cannot be empty")
    terms = []
    for term in effects.split("+"):
        names = [s.strip() for s in term.strip().split("^")]
        if any(not s.isidentifier() for s in names):
            raise ValueError(
                "additional fixed effects must be column names or name^name interactions"
            )
        if len(names) == 1 and names[0] in panel:
            continue  # Unit/time intercepts are always included.
        if panel[0] in names:
            raise ValueError("unit-interacted fixed effects are not supported")
        terms.append(":".join(f"C({name})" for name in names))
    return " + ".join([covariates, *terms])


class RowDesign:
    def __init__(self, store):
        self.store = store
        self.time = np.eye(store.periods)[:, 1:]
        names = store.design.names if store.design is not None else []
        self.names = [f"time::{t}" for t in store.times[1:]] + list(names)
        self.keep = np.arange(len(self.names))
        self.dropped = []

    def batches(self, *, actual_cohort=False):
        for g, y, x, w in self.store.batches(
            include_x=True, include_weights=True, row_weights=True, actual_cohort=actual_cohort
        ):
            x = x if self.store.design is not None else x[:, :, :0]
            X = np.concatenate(
                [np.broadcast_to(self.time, (*y.shape, self.time.shape[1])), x], axis=2
            )
            yield g, y, np.nan_to_num(X, nan=0.0)[:, :, self.keep], w

    def reduce(self, clean=None):
        """Select identified columns; optionally also the clean-row moments.

        ``clean(g)`` gives the rows (units by periods) of a second, clean-sample
        regression. Its within-unit moments are accumulated in the same pass
        over all columns and returned restricted to the identified ones:
        centering acts column by column, so restricting afterwards is exact.
        """
        P = len(self.names)
        Q = np.zeros((P, P))
        norms = np.zeros(P)
        Qc, qc = (np.zeros((P, P)), np.zeros(P)) if clean is not None else (None, None)
        for g, y, X, w in self.batches():
            y = np.nan_to_num(y, nan=0.0)
            q, _ = moments(X, y, w)
            Q += q
            norms += weighted_squares(X, w)
            if clean is not None:
                a, b = moments(X, y, w * clean(g))
                Qc += a
                qc += b
        self.keep = independent_columns(Q, original_norm=norms)
        self.dropped = [n for j, n in enumerate(self.names) if j not in self.keep]
        if not len(self.keep):
            raise ValueError("the within-unit regression has no identified nuisance columns")
        if clean is not None:
            return Qc[np.ix_(self.keep, self.keep)], qc[self.keep]

    def residuals(self, g, y, X, w, theta, anticipation, *, label="BJS"):
        observed = np.isfinite(y)
        use = observed & ((g[:, None] == 0) | (self.store.times < g[:, None] - anticipation))
        clean_weight = w * use
        raw = np.where(observed, y - X @ theta, 0)
        mass = clean_weight.sum(axis=1)
        if np.any(mass <= 0):
            raise ValueError(f"{label} requires at least one untreated observation per unit")
        xm = row_sums(clean_weight, X) / mass[:, None]
        alpha = (clean_weight * raw).sum(axis=1) / mass
        residual = np.where(observed, raw - alpha[:, None], 0)
        first_score = row_sums(clean_weight * residual, X - xm[:, None])
        return observed, use, xm, residual, first_score


def fit_bjs(store, cells, avgeffectsby, leaveout, workdir, anticipation=0):
    from .unbalanced import ObservedBJSInference

    model = RowDesign(store)
    Q, q = model.reduce(
        clean=lambda g: (g[:, None] == 0) | (store.times < g[:, None] - anticipation)
    )
    P, M = len(model.keep), len(cells)
    theta = solve_checked(Q, q, "BJS untreated covariate/fixed-effect design is unidentified")
    counts = np.array([store.cell_weights[g][t - store.times[0]] for g, t in cells])
    beta, D = np.zeros(M), np.zeros((M, P))
    for g, y, X, w in model.batches():
        obs, _, xm, resid, _ = model.residuals(g, y, X, w, theta, anticipation)
        for j, (cg, t) in enumerate(cells):
            ti = t - store.times[0]
            mask = (g == cg) & obs[:, ti]
            sw = w[mask, ti] / counts[j]
            beta[j] += sw @ resid[mask, ti]
            D[j] += sw @ (X[mask, ti] - xm[mask])
    H = solve_checked(Q, D.T, "BJS untreated covariate/fixed-effect design").T
    context = ObservedBJSInference(
        store,
        cells,
        counts,
        None,
        theta,
        H,
        avgeffectsby,
        leaveout,
        workdir,
        anticipation,
        model=model,
    )
    try:
        V = context.covariance(np.eye(M), allow_unidentified=avgeffectsby != "cohort_time")
    except BaseException:
        context.cleanup()
        raise
    return beta, counts, V, context, model


def fit_sunab(store, cells, anticipation=0):
    model = RowDesign(store)
    model.reduce()
    K, M = len(model.keep), len(cells)
    P = K + M
    Q, q = np.zeros((P, P)), np.zeros(P)

    def batches():
        for g, y, X, w in model.batches():
            for cg in np.unique(g):
                mask = g == cg
                js = [j for j, (cohort, _) in enumerate(cells) if cohort == cg]
                interaction = (
                    np.column_stack([store.times == cells[j][1] for j in js])
                    if js
                    else np.empty((store.periods, 0))
                )
                active = np.r_[np.arange(K), K + np.array(js, dtype=int)]
                design = np.concatenate(
                    [X[mask], np.broadcast_to(interaction, (int(mask.sum()), *interaction.shape))],
                    axis=2,
                )
                yield mask, active, design, y[mask], w[mask]

    for _, active, X, y, w in batches():
        a, b = moments(X, np.nan_to_num(y, nan=0.0), w)
        Q[np.ix_(active, active)] += a
        q[active] += b
    coefficients = solve_checked(
        Q, q, "Sun–Abraham covariate/fixed-effect interactions are unidentified"
    )
    H = solve_checked(Q, np.eye(P)[:, K:], "Sun–Abraham joint design")
    V = np.zeros((M, M))
    for _, active, X, y, w in batches():
        xc, yc, _, _ = center_groups(X, np.nan_to_num(y, nan=0.0), w)
        residual = yc - xc @ coefficients[active]
        score = row_sums(w * residual, xc) @ H[active]
        V += score.T @ score
    counts = np.array([store.cell_weights[g][t - store.times[0]] for g, t in cells])
    return coefficients[K:], counts, (V + V.T) / 2, model


def fit_placebos(store, pretrends, anticipation, *, fml, depvar, panel, vcov):
    """Separate clean lead regression with the same richer nuisance design."""
    import polars as pl

    from .bjs_placebo import BJSPlaceboResult

    model = RowDesign(store)
    model.reduce()
    L, K = pretrends, len(model.keep)
    horizons = np.arange(-anticipation - L, -anticipation, dtype=np.int64)
    support, rows = np.zeros(L, dtype=int), []
    for g, counts in sorted(store.actual_cell_counts.items()):
        if not g:
            continue
        for ti, t in enumerate(store.times):
            j = int(t) - g + anticipation + L
            if 0 <= j < L and counts[ti]:
                support[j] += counts[ti]
                rows.append(
                    dict(
                        aggregate=f"event::{horizons[j]}",
                        cohort=int(g),
                        time=int(t),
                        event_time=int(horizons[j]),
                        observed_units=int(counts[ti]),
                        weight_total=float(
                            store.actual_cell_weights[g][ti]
                            * store.weight_scale
                            / store.weight_normalizer
                        ),
                    )
                )
    if np.any(support == 0):
        raise ValueError(
            "BJS requested clean placebo horizons are not observed; reduce pretrend_leads"
        )

    def batches():
        for g, y, X, w in model.batches(actual_cohort=True):
            lead = ((store.times[None, :, None] - g[:, None, None]) == horizons) & (
                g[:, None, None] != 0
            )
            use = np.isfinite(y) & ((g[:, None] == 0) | (store.times < g[:, None] - anticipation))
            yield np.concatenate([X, lead], axis=2), np.nan_to_num(y, nan=0.0), w * use, use

    P = K + L
    Q, q = np.zeros((P, P)), np.zeros(P)
    n_obs = 0
    for X, y, w, use in batches():
        a, b = moments(X, y, w)
        Q += a
        q += b
        n_obs += int(use.sum())
    keep = independent_columns(Q)
    if not np.isin(np.arange(K, P), keep).all():
        raise ValueError("BJS clean placebo design is unidentified; reduce pretrend_leads")
    Q = Q[np.ix_(keep, keep)]
    context = "BJS clean placebo covariate/fixed-effect design"
    coefficient = solve_checked(Q, q[keep], context)
    H = solve_checked(Q, np.eye(len(keep))[:, -L:], context)
    V = np.zeros((L, L))
    for X, y, w, _ in batches():
        xc, yc, _, _ = center_groups(X[:, :, keep], y, w)
        score = row_sums(w * (yc - xc @ coefficient), xc) @ H
        V += score.T @ score
    V *= store.n_units / (store.n_units - 1) if vcov == "CRV1" else 1.0
    all_names = [*(model.names[j] for j in model.keep), *(f"event::{h}" for h in horizons)]
    return BJSPlaceboResult(
        fml=fml,
        depvar=depvar,
        coefnames=[f"event::{h}" for h in horizons],
        beta=coefficient[-L:],
        vcov=(V + V.T) / 2,
        vcov_type=vcov,
        df_t=store.n_units - 1,
        all_vcovs={vcov: (V, store.n_units - 1)},
        n_obs=n_obs,
        n_clusters={panel[0]: store.n_units},
        n_units=store.n_units,
        weights=None,
        _support=pl.DataFrame(rows),
        diagnostics=dict(
            aggregation="event_time",
            anticipation=anticipation,
            reference_event_time=None,
            fitted_horizons=horizons.tolist(),
            pretrends=L,
            base_period="all other clean observations; no fixed event-time reference",
            inference="unit clustered; clean-sample covariate/fixed-effect WLS regression",
            regression_columns=[all_names[j] for j in keep],
            absorbed_or_redundant_columns=model.dropped
            + [n for j, n in enumerate(all_names) if j not in keep],
            warnings=[],
        ),
    )
