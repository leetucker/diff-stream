"""Period labels, and cohorts from a treatment column.

Panels with evenly spaced periods (every year, every second year, ...) are
numbered consecutively inside the store, starting at the first period, so the
estimators count periods: event time 1 is the next period, whatever its
spacing, and anticipation and pre-trend leads are numbers of periods. Results
show the original labels of cohorts and periods.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodLabels:
    """Original period = origin + (internal period - origin) * step."""

    origin: int
    step: int = 1

    def internal(self, t):
        """Internal periods of on-grid original periods."""
        return self.origin + (np.asarray(t) - self.origin) // self.step

    def internal_cohorts(self, g, last_internal):
        """Internal cohorts; 0 stays 0. Cohorts within the sample must be periods of it."""
        g = np.asarray(g).copy()
        nonzero = g != 0
        offset = g[nonzero] - self.origin
        inside = (offset > 0) & (g[nonzero] <= self.label(last_internal))
        off_grid = inside & (offset % self.step != 0)
        if off_grid.any():
            raise ValueError(
                f"cohort {int(g[nonzero][off_grid][0])} is not one of the panel's periods "
                f"({self.origin}, {self.origin + self.step}, ...): give the first period in "
                "which the unit is observed treated, or use treatment="
            )
        # Cohorts after the sample stay after it (they are never treated within it).
        g[nonzero] = self.origin - (-offset // self.step)
        return g

    def label(self, t):
        """Original labels of internal periods (scalars or arrays)."""
        if self.step == 1:
            return t
        return self.origin + (np.asarray(t) - self.origin) * self.step

    def label_int(self, t):
        return int(self.label(int(t)))

    def label_cohort(self, g):
        """Original cohort label; 0 (never treated) stays 0."""
        return 0 if int(g) == 0 else self.label_int(g)


def cohorts_from_treatment(treatment, t, ids, starts, unit):
    """Each row's cohort: its unit's first period with treatment 1, else 0.

    Rows are sorted by unit and period. Treatment must be 0/1 and absorbing.
    """
    if np.any((treatment != 0) & (treatment != 1)):
        raise ValueError("treatment must be 0 or 1")
    same_unit = ids[1:] == ids[:-1]
    if np.any(same_unit & (treatment[1:] < treatment[:-1])):
        raise ValueError(
            "treatment must be absorbing: once 1, a unit stays treated in its later periods"
        )
    never = np.iinfo(np.int64).max
    first = np.minimum.reduceat(np.where(treatment == 1, t, never), starts)
    first[first == never] = 0
    return first[unit]
