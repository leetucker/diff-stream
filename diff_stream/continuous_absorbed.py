"""did_continuous with several fixed-effect terms: the comparison regressions by hdfe-stream.

With one term, ``continuous_general`` solves each cohort-period's comparison regression
exactly from sums by level. With several (``| county + industry``), each is fitted by
hdfe-stream, and this module fills the same per-cell tables that the general fit's
passes use: the covariates' coefficients, each term's level effects (alpha), which
levels have comparison units, and, after the treated units' sums, how each comparison
unit's residual moves the cell's estimates.

Cells with the same design (comparison groups, base period and covariate period, and
the period compared when the panel is unbalanced or the weights change over time) share
one hdfe-stream fit, with an outcome change for each cell. The fits read a wide file of
the units written once, and are kept until the influence is solved.

Identification. A treated unit's adjusted change uses the fitted effects of its levels,
which are identified only when its combination of levels is estimable from the
comparison units (with two terms, when its levels are in one connected component of the
comparison units' levels). For each design a probe finds the units for which it is not:
with random level values r, u = S^- S r (S the comparison regression's normal-equations
matrix, solved by hdfe-stream from the rows), so delta = r - u is a generic vector of the
null space of the comparison design, and a design z is estimable when z'delta = 0.

Influence. A comparison unit i with residual e_i and weight w_i moves cell j's dose
coefficients by -A_j^-1 U_j' z_i w_i e_i, with U_j = S^- R_j and R_j the treated units'
sums of w psi z (by term and level, and with the covariates) from the general fit's
pass B; the binarized ATT the same with sums of w z over the cell's treated weight.
hdfe-stream solves the normal equations for right-hand sides built from rows, so R_j is
passed as a few rows whose sums equal it (see ``_rows``).
"""

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .absorbed import hdfe_options, solve_block

# |z'delta| above this marks a combination of levels that cannot be estimated. The
# probe's level values are standard normal: for an estimable design z'delta is zero up
# to the solver's precision (about 1e-12 on the test panels); for one that is not it is
# of the order of the probe's values on the smaller of the components it joins.
TOLERANCE = 1e-6
PROBE_SEED = 20261120


class Absorbed:
    """hdfe-stream fits of the comparison regressions, one per design; a context manager
    that removes them and their files."""

    def __init__(self, store, threads):
        self.store, self.threads = store, threads
        self.directory = Path(tempfile.mkdtemp(prefix="continuous-absorbed-", dir=store.root))
        self.designs = []  # (cells, fit, model)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False

    def cleanup(self):
        for _, fit, _ in self.designs:
            fit.cleanup()
        self.designs = []
        shutil.rmtree(self.directory, ignore_errors=True)

    # -- the wide file and the designs -------------------------------------------------

    def _write_units(self, batches, ti, bases, period):
        """One row per unit: its group, and its outcome, weight, covariates and levels in
        the periods the cells use."""
        compared = sorted(set(ti.tolist()) | set(bases.tolist()))
        covariate_periods = sorted(set(period.tolist()))
        path = self.directory / "units.parquet"
        writer = None
        try:
            for k, y, X, lv, _, w, _, _ in batches():
                self.F, self.P = lv.shape[2], X.shape[2]
                columns = {"k": k.astype(np.int64)}
                for s in compared:
                    columns[f"y_{s}"] = y[:, s]
                    columns[f"w_{s}"] = w[:, s]
                for s in covariate_periods:
                    for c in range(self.P):
                        columns[f"x{c}_{s}"] = X[:, s, c]
                    for f in range(self.F):
                        columns[f"fe{f}_{s}"] = lv[:, s, f].astype(np.int64)
                table = pa.table(columns)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        return path

    def _designs(self, cells, ti, bases, period, controls):
        """The cells grouped by design, in the order of their first cell."""
        store = self.store
        by_period = not store.is_balanced or not store.weights_time_invariant
        designs = {}
        for j in range(len(cells)):
            key = (
                controls[j].tobytes(),
                int(bases[j]),
                int(period[j]),
                int(ti[j]) if by_period else -1,
            )
            designs.setdefault(key, []).append(j)
        return list(designs.values())

    def _comparison_rows(self, path, members, controls, ti, bases, period):
        """The design's comparison units, with an outcome change for each of its cells."""
        j0 = members[0]
        groups = np.flatnonzero(controls[j0]).tolist()
        b, p = int(bases[j0]), int(period[j0])
        observed = [pl.col(f"y_{int(ti[j])}").is_not_nan() for j in members]
        return (
            pl.scan_parquet(path)
            .filter(pl.col("k").is_in(groups), pl.col(f"y_{b}").is_not_nan(), *observed)
            .select(
                *[
                    (pl.col(f"y_{int(ti[j])}") - pl.col(f"y_{b}")).alias(f"dy{i}")
                    for i, j in enumerate(members)
                ],
                *[pl.col(f"x{c}_{p}").alias(f"x{c}") for c in range(self.P)],
                *[pl.col(f"fe{f}_{p}").alias(f"fe{f}") for f in range(self.F)],
                pl.col(f"w_{int(ti[j0])}").alias("w"),
            )
        )

    # -- pass A ------------------------------------------------------------------------

    def fit(
        self, batches, cells, gi, ti, bases, period, controls, gamma, alpha, present, probe,
        probe_x,
    ):  # fmt: skip
        """Fit each design's comparison regressions and probe their identification,
        filling ``gamma``, ``alpha``, ``present``, ``probe`` and ``probe_x`` in place."""
        from hdfe_stream import feols_stream

        path = self._write_units(batches, ti, bases, period)
        xs = [f"x{c}" for c in range(self.P)]
        terms = [f"fe{f}" for f in range(self.F)]
        rng = np.random.default_rng(PROBE_SEED)
        for members in self._designs(cells, ti, bases, period, controls):
            rows = self._comparison_rows(path, members, controls, ti, bases, period)
            if rows.select(pl.len()).collect().item() == 0:
                g, t = cells[members[0]]
                raise ValueError(f"cohort {g} has no comparison units observed in period {t}")
            outcomes = ", ".join(f"dy{i}" for i in range(len(members)))
            outcomes = f"sw({outcomes})" if len(members) > 1 else outcomes
            fml = f"{outcomes} ~ {' + '.join(xs) or '1'} | {' + '.join(terms)}"
            options = hdfe_options(
                self.store, self.threads, self.P, vcov="iid", keep_intermediates=True
            )
            options["weights"] = "w"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*variables dropped due to multicollinearity")
                fit = feols_stream(fml, rows, workdir=self.directory, **options)
            models = list(fit) if hasattr(fit, "fetch_model") else [fit]
            self.designs.append((members, fit, models[0]))
            for j, model in zip(members, models, strict=True):
                if list(model.coefnames) != xs:
                    g, t = cells[j]
                    raise ValueError(
                        f"the comparison regression of cohort {g}, period {t}: covariates "
                        "are collinear with the fixed effects"
                    )
                gamma[j] = np.asarray(model.beta, dtype=float)
                for f, term in enumerate(terms):
                    codes, values = _effects(model.fixef(term).collect(), term)
                    alpha[j, f, codes] = values
                    present[j, f, codes] = True
            self._probe(models[0], rows, members, terms, present, probe, probe_x, rng)

    def _probe(self, model, rows, members, terms, present, probe, probe_x, rng):
        """delta = r - S^- S r for random level values r: a generic null vector."""
        j0 = members[0]
        values, frame = [], rows
        for f, term in enumerate(terms):
            codes = np.flatnonzero(present[j0, f])
            r = rng.standard_normal(len(codes))
            values.append((codes, r))
            table = pl.DataFrame({term: codes.astype(np.int64), f"__r{f}": r})
            frame = frame.join(table.lazy(), on=term, how="left")
        frame = frame.with_columns(
            (pl.col("w") * pl.sum_horizontal(f"__r{f}" for f in range(len(terms)))).alias("__probe")
        )
        solution = model.solve_normal_equations(frame, ["__probe"], block=1)
        try:
            delta = np.zeros(probe.shape[1:])
            for f, (term, (codes, r)) in enumerate(zip(terms, values, strict=True)):
                solved, u = _effects(solution.fixef(term).collect(), term, "__probe")
                lookup = np.zeros(probe.shape[2])
                lookup[solved] = u
                delta[f, codes] = r - lookup[codes]
            delta_x = -solution.coef["__probe"].to_numpy()
        finally:
            solution.cleanup()
        for j in members:
            probe[j], probe_x[j] = delta, delta_x

    # -- between passes B and C -----------------------------------------------------------

    def influence(
        self, level_basis, level_count, basis_x, treated_x, ainv, size, level_weight,
        slope_weight, att_level_weight, att_slope_weight,
    ):  # fmt: skip
        """Solve each design's normal equations for its cells' treated sums and fill the
        comparison units' weights in place (see the module docstring)."""
        terms = [f"fe{f}" for f in range(self.F)]
        K = level_basis.shape[3]
        for members, _, model in self.designs:
            names, levels, covariates = [], [], []
            for j in members:
                for k in range(K):
                    names.append(f"c{j}_{k}")
                    levels.append(level_basis[j, :, :, k])
                    covariates.append(basis_x[j, k])
                names.append(f"c{j}_att")
                levels.append(level_count[j])
                covariates.append(treated_x[j])
            levels = np.stack(levels, axis=2)  # terms, levels, columns
            covariates = np.stack(covariates, axis=1) if self.P else np.zeros((0, len(names)))
            codes = [_effects(model.fixef(t).collect(), t)[0] for t in terms]
            frame = _rows(terms, codes, levels, covariates, names, self.P)
            solution = model.solve_normal_equations(
                frame, names, block=solve_block(self.store, len(names))
            )
            try:
                U = np.zeros(levels.shape)
                for f, term in enumerate(terms):
                    table = solution.fixef(term).collect()
                    U[f, table[term].to_numpy()] = table.select(names).to_numpy()
                Ux = solution.coef.select(names).to_numpy()  # covariates, columns
            finally:
                solution.cleanup()
            for i, j in enumerate(members):
                block = slice(i * (K + 1), i * (K + 1) + K)
                level_weight[j] = U[:, :, block] @ ainv[j].T
                slope_weight[j] = ainv[j] @ Ux[:, block].T
                att_level_weight[j] = U[:, :, i * (K + 1) + K] / size[j]
                att_slope_weight[j] = Ux[:, i * (K + 1) + K] / size[j]


def _effects(table, term, value=None):
    """A fixef table's level codes and values (``value`` names the value column)."""
    if value is None:
        value = next(c for c in table.columns if c not in (term, "n_obs"))
    return table[term].to_numpy().astype(np.int64), table[value].to_numpy()


def _rows(terms, codes, levels, covariates, names, P):
    """Rows whose sums Z'v equal a right-hand side given by term and level.

    ``levels`` is (terms, levels, columns) and ``covariates`` (covariates, columns). A
    right-hand side built from rows has the same total in every term, since each row
    has one level of each. A row for each level of each term carries its sums, with the
    other terms at a reference level and the covariates zero; a row for each covariate
    has that covariate one and every term at its reference level; a last row, at the
    reference levels, removes what the other rows added to them: (F - 1) times the
    total, and the covariates' sums.
    """
    F = len(terms)
    reference = [int(c[0]) for c in codes]
    total = levels[0].sum(axis=0)
    blocks = []

    def block(level_codes, x, values):
        n = len(values)
        data = {
            term: np.asarray(level_codes[f], dtype=np.int64) * np.ones(n, dtype=np.int64)
            for f, term in enumerate(terms)
        }
        data.update({f"x{c}": x[:, c] for c in range(P)})
        data.update({name: values[:, m] for m, name in enumerate(names)})
        blocks.append(pl.DataFrame(data))

    for f in range(F):
        level_codes = [np.full(len(codes[f]), reference[e]) for e in range(F)]
        level_codes[f] = codes[f]
        block(level_codes, np.zeros((len(codes[f]), P)), levels[f][codes[f]])
    if P:
        block([np.full(P, r) for r in reference], np.eye(P), covariates)
    correction = -(F - 1) * total - covariates.sum(axis=0)
    block([np.array([r]) for r in reference], np.zeros((1, P)), correction[None, :])
    return pl.concat(blocks)
