"""Linear ETWFE references: authors' R commands and independent dense WLS.

--regenerate uses pinned source and locally installed R validation dependencies;
no downloading/installation occurs here. Ordinary tests use frozen CSVs only.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks._estimators import public  # noqa: E402
from benchmarks._files import digest as file_digest  # noqa: E402
from benchmarks._files import read_csv  # noqa: E402
from diff_stream import did_extended_twfe  # noqa: E402

DATA = ROOT / "tests/data"
REF = DATA / "etwfe_reference"
COMMIT = "d36673d3f07783aa297bbec31d6d8aabfc66243b"
SOURCE = ROOT / "benchmarks/external/.r/etwfe-source" / COMMIT
EXPORTER = ROOT / "benchmarks/external/etwfe_reference.R"


def scenarios():
    for panel in ["balanced", "unbalanced"]:
        for control in ["never", "notyet"]:
            for fe in ["cohort", "unit"]:
                for case in ["plain_equal", "numeric_varying"]:
                    yield panel, control, fe, case, 0
            for case in ["numeric_equal", "numeric_unit"]:
                yield panel, control, "cohort", case, 0
            yield panel, control, "cohort", "numeric_varying", 1
    for control in ["never", "notyet"]:
        yield "mpdta", control, "cohort", "mpdta", 0


def sample(panel):
    if panel == "mpdta":
        return read_csv(DATA / "mpdta.csv").rename(
            {
                "countyreal": "id",
                "year": "time",
                "first.treat": "first_treat",
                "lemp": "y",
                "lpop": "x",
            }
        )
    return read_csv(
        DATA / ("richer_panel.csv" if panel == "balanced" else "richer_unbalanced_panel.csv")
    )


def configuration(case):
    formula = "y ~ 1" if case == "plain_equal" else "y ~ x" if case == "mpdta" else "y ~ x+z"
    weight = (
        None
        if case in ("plain_equal", "numeric_equal", "mpdta")
        else "unit_w"
        if case == "numeric_unit"
        else "w"
    )
    return formula, weight


def dense_oracle(d, case, control, fe, A, inference):
    """All-unit dummies, SVD and row influences; independent of stream kernels."""
    _, weight = configuration(case)
    y, ids, time, g = [d[c].to_numpy() for c in ["y", "id", "time", "first_treat"]]
    w = np.ones(len(d)) if weight is None else d[weight].to_numpy()
    x = (
        np.empty((len(d), 0))
        if case == "plain_equal"
        else d.select("x" if case == "mpdta" else ["x", "z"]).to_numpy()
    )
    groups, dates = np.unique(g), np.unique(time)
    cells = [
        (int(cg), int(t))
        for cg in groups
        if cg
        for t in dates
        if np.any((g == cg) & (time == t))
        and (t >= cg - A if control == "notyet" else t != cg - A - 1)
    ]
    columns = (
        [(ids == u).astype(float) for u in np.unique(ids)]
        if fe == "unit"
        else [(g == cg).astype(float) for cg in groups]
    )
    columns += [(time == t).astype(float) for t in dates[1:]]
    columns += [x[:, k] * (g == cg) for cg in groups for k in range(x.shape[1])]
    columns += [x[:, k] * (time == t) for t in dates[1:] for k in range(x.shape[1])]
    offset = len(columns)
    one_x = np.column_stack([np.ones(len(d)), x])
    columns += [
        one_x[:, k] * ((g == cg) & (time == t)) for cg, t in cells for k in range(one_x.shape[1])
    ]
    F = np.column_stack(columns)
    root = np.sqrt(w)
    pinv = np.linalg.pinv(F * root[:, None], rcond=1e-12)
    coef = pinv @ (y * root)
    Qinv = pinv @ pinv.T
    M, P = len(cells), F.shape[1]
    L = np.zeros((M, P))
    mass = np.zeros(M)
    masks = []
    for j, (cg, t) in enumerate(cells):
        mask = (g == cg) & (time == t)
        masks.append(mask)
        mass[j] = w[mask].sum()
        L[j, offset + j * one_x.shape[1] : offset + (j + 1) * one_x.shape[1]] = (
            w[mask] @ one_x[mask] / mass[j]
        )
    beta = L @ coef
    score = (F * (w * (y - F @ coef))[:, None]) @ Qinv @ L.T
    share = np.column_stack([w * mask for mask in masks])
    if inference == "unconditional":
        for j, mask in enumerate(masks):
            row = (
                one_x[mask] @ coef[offset + j * one_x.shape[1] : offset + (j + 1) * one_x.shape[1]]
            )
            score[mask, j] += w[mask] * (row - beta[j]) / mass[j]
    units = np.unique(ids)
    cluster = np.vstack([score[ids == u].sum(axis=0) for u in units])
    sizes = np.vstack([share[ids == u].sum(axis=0) for u in units]) - mass / len(units)
    V = cluster.T @ cluster
    aggregates = {}
    for kind in ["dynamic", "simple", "cohort", "calendar"]:
        c = np.array(cells)
        event = c[:, 1] - c[:, 0]
        keep = np.ones(M, dtype=bool) if kind == "dynamic" else event >= 0
        values = {"dynamic": event, "cohort": c[:, 0], "calendar": c[:, 1]}.get(
            kind, np.zeros(M, dtype=int)
        )
        levels = sorted(set(values[keep]))
        W = np.zeros((len(levels), M))
        for j, v in enumerate(levels):
            mask = keep & (values == v)
            W[j, mask] = mass[mask] / mass[mask].sum()
        effect = W @ beta
        psi = cluster @ W.T
        if inference == "unconditional":
            for j in range(len(levels)):
                ix = W[j] != 0
                derivative = np.where(ix, (beta - effect[j]) / mass[ix].sum(), 0)
                psi[:, j] += sizes @ derivative
        aggregates[kind] = (effect, psi.T @ psi)
    return cells, beta, V, aggregates


def digest(path):
    return file_digest(path)


def verify_provenance():
    manifest = json.loads((REF / "PROVENANCE.json").read_text())
    assert manifest["source"]["commit"] == COMMIT
    assert digest(EXPORTER) == manifest["exporter_sha256"]
    for name, value in manifest["inputs"].items():
        assert digest(DATA / name) == value, name
    for name, value in manifest["outputs"].items():
        assert digest(REF / name) == value, name
    return manifest


def compare_reference(fit, prefix):
    ref = read_csv(REF / (prefix + "_cells.csv"))
    # R was fit on shifted adoption dates for anticipation; retain actual labels.
    A = fit.diagnostics["anticipation"]
    keys = list(zip(ref["first_treat"].to_numpy() + A, ref["time"], strict=True))
    ix = [keys.index(c) for c in fit.cells]
    V = read_csv(REF / (prefix + "_cells_vcov.csv"), has_header=False).to_numpy()[np.ix_(ix, ix)]
    factor = fit.n_units / (fit.n_units - 1) if not fit.diagnostics["treat_sample_as_fixed"] else 1
    beta = ref["estimate"].to_numpy()[ix]
    assert_allclose(fit.beta, beta, atol=2e-9, rtol=0)
    assert_allclose(fit.vcov * factor, V, atol=2e-9, rtol=0)
    errors = dict(
        beta_max_abs=float(np.max(abs(fit.beta - beta))),
        vcov_max_abs=float(np.max(abs(fit.vcov * factor - V))),
    )
    for kind, rkind, key in [
        ("dynamic", "event", "event"),
        ("simple", "simple", None),
        ("cohort", "group", "first_treat"),
        ("calendar", "calendar", "time"),
    ]:
        agg = fit.aggregate(public("kind", kind))
        ref = read_csv(REF / (prefix + "_" + rkind + ".csv"))
        if key:
            levels = ref[key].to_numpy() + (
                A if key == "first_treat" else -A if key == "event" else 0
            )
            ix = [list(levels).index(int(n.split("::")[-1])) for n in agg.coefnames]
        else:
            ix = [0]
        rb = ref["estimate"].to_numpy()[ix]
        rv = read_csv(REF / (prefix + "_" + rkind + "_vcov.csv"), has_header=False).to_numpy()[
            np.ix_(ix, ix)
        ]
        assert_allclose(agg.beta, rb, atol=2e-9, rtol=0)
        assert_allclose(agg.vcov * factor, rv, atol=2e-9, rtol=0)
        errors[kind + "_beta_max_abs"] = float(np.max(abs(agg.beta - rb)))
        errors[kind + "_vcov_max_abs"] = float(np.max(abs(agg.vcov * factor - rv)))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    parser.add_argument(
        "--anticipation-only",
        action="store_true",
        help="Regenerate only anticipation references; requires --regenerate",
    )
    args = parser.parse_args()
    if args.anticipation_only and not args.regenerate:
        parser.error("--anticipation-only requires --regenerate")
    if args.regenerate:
        source = json.loads((SOURCE / "source.json").read_text())
        for name, entry in source["files"].items():
            assert digest(SOURCE / name) == entry["sha256"]
        env = dict(
            os.environ,
            R_LIBS_USER=str(ROOT / "benchmarks/external/.r/library"),
            TZ="UTC",
            OPENBLAS_NUM_THREADS="2",
            OMP_NUM_THREADS="2",
        )
        subprocess.run(
            [args.rscript, str(EXPORTER), str(ROOT), str(REF), str(SOURCE / "R")]
            + (["anticipation"] if args.anticipation_only else []),
            env=env,
            check=True,
        )
        manifest = dict(
            source=source,
            marginaleffects_source=json.loads(
                (
                    ROOT
                    / "benchmarks/external/.r/marginaleffects-source"
                    / "a4ac200424ec397862b434b67c8cd8fff131e972"
                    / "source.json"
                ).read_text()
            ),
            execution="Pinned unmodified authors etwfe/emfx functions sourced directly, linear only; explicit feo/none coding; conditional vcov=TRUE; unconditional HC0 unit clustering.",
            exporter_sha256=digest(EXPORTER),
            inputs={
                name: digest(DATA / name)
                for name in ["richer_panel.csv", "richer_unbalanced_panel.csv", "mpdta.csv"]
            },
            outputs={
                p.name: digest(p)
                for p in sorted(REF.iterdir())
                if p.is_file() and p.name != "PROVENANCE.json"
            },
        )
        (REF / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_provenance()
    rows = []
    for panel, control, fe, case, A in scenarios():
        d = sample(panel)
        formula, weight = configuration(case)
        for inference in ["conditional", "unconditional"]:
            prefix = f"{panel}_{control}_{fe}_{case}_{A}_{inference}"
            fit = did_extended_twfe(
                formula,
                d,
                panel=("id", "time"),
                cohort="first_treat",
                control_group=public(
                    "control_group", "nevertreated" if control == "never" else "notyettreated"
                ),
                fixed_effects=fe,
                treat_sample_as_fixed=public("treat_sample_as_fixed", inference),
                weights=weight,
                allow_time_varying_weights=True,
                incomplete_units=public(
                    "incomplete_units", "unbalanced" if panel == "unbalanced" else "balanced"
                ),
                anticipation=A,
                n_threads=2,
                batch_units=37,
                preprocess_mb=4,
            )
            rows.append(dict(case=prefix, **compare_reference(fit, prefix)))
    out = ROOT / "benchmarks/results/etwfe_validation"
    out.mkdir(parents=True, exist_ok=True)
    save = dict(cases=len(rows), passed=True, comparisons=rows)
    (out / "report.json").write_text(json.dumps(save, indent=2) + "\n")
    print(json.dumps({k: v for k, v in save.items() if k != "comparisons"}))


if __name__ == "__main__":
    main()
