"""Bounded formula-design discovery using hdfe-stream's existing compiler.

Generic staging adapter; intended for eventual hdfe_stream.core.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from hdfe_stream.core.formula import categorical_vars, compile_design, numeric_expr


@dataclass
class CovariatePlan:
    outcome: str
    rhs: str
    schema: object
    required: list
    categorical: set
    levels: dict = field(default_factory=dict)
    names: list = field(default_factory=list)
    expressions: list = field(default_factory=list)
    max_columns: int = 256

    @classmethod
    def parse(cls, outcome, rhs, schema):
        import formulaic

        formula = formulaic.Formula(rhs)
        variables = set(formula.required_variables)
        # formulaic does not know pyfixest's i() function during discovery.
        if not any(f.expr == "i" for term in formula for f in term.factors):
            variables.discard("i")
        functions = {"abs", "log", "log1p", "log10", "exp", "sqrt", "np", "I", "C"}
        scalar_names = {f.expr for term in formula for f in term.factors}
        variables -= functions - scalar_names
        required = sorted(variables)
        missing = set(required) - set(schema)
        if missing:
            raise ValueError(f"unknown variables in formula: {sorted(missing)}")
        cats = categorical_vars(rhs, schema)
        return cls(outcome, rhs, schema, required, cats, {c: set() for c in cats})

    @property
    def raw_columns(self):
        return [f"__raw{j}" for j in range(len(self.required))]

    def discover(self, frame):
        for j, c in enumerate(self.required):
            if c in self.categorical:
                self.levels[c].update(frame[f"__raw{j}"].unique().to_list())
                if len(self.levels[c]) > self.max_columns:
                    raise ValueError(
                        "categorical design exceeds the current 256-column design limit"
                    )

    def compile(self):
        import formulaic

        self.levels = {c: sorted(v) for c, v in self.levels.items()}
        # Guard expansion before constructing the compiler's small synthetic frame.
        upper = 0
        for term in formulaic.Formula(self.rhs):
            cats = (
                set().union(*(factor.required_variables for factor in term.factors))
                & self.categorical
            )
            upper += int(np.prod([len(self.levels[c]) for c in cats])) if cats else 1
        if upper > self.max_columns * 4:
            raise ValueError("formula expansion exceeds the current 256-column design limit")
        if not self.categorical:
            # Numeric terms need no synthetic model matrix or pyfixest import.
            # Reuse the same expression compiler as the family frontend.
            design = []
            for term in formulaic.Formula(self.rhs):
                if str(term) == "1":
                    design.append(("Intercept", pl.lit(1.0)))
                else:
                    expr = pl.lit(1.0)
                    for factor in term.factors:
                        expr = expr * numeric_expr(factor.expr)
                    design.append((str(term), expr))
        else:
            _, design = compile_design(
                self.outcome, self.rhs, self.schema, self.levels, intercept=True
            )
        if not design or len(design) > self.max_columns:
            raise ValueError("the formula needs between 1 and 256 design columns")
        self.names = [n for n, e in design]
        self.expressions = [e.cast(pl.Float64).alias(f"__x{j}") for j, (n, e) in enumerate(design)]

    def evaluate(self, frame):
        anchor = "__diff_anchor"
        while anchor in self.required:
            anchor += "_"
        work = frame.select(
            pl.col("y").alias(anchor),
            *[pl.col(f"__raw{j}").alias(c) for j, c in enumerate(self.required)],
        )
        result = work.with_columns(self.expressions).select(
            [f"__x{j}" for j in range(len(self.names))]
        )
        if sum(result.null_count().row(0)) or not np.isfinite(result.to_numpy()).all():
            raise ValueError("covariate design has missing or non-finite values")
        return result
