"""BJS smart residual averaging and cluster leave-out adjustments.

The implementation uses cohort-sized first/second moments. It implements the
published score algebra independently; no author implementation is vendored.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BJSInference:
    groups: np.ndarray
    sizes: np.ndarray
    times: np.ndarray
    cells: list
    means: np.ndarray
    Z: np.ndarray
    theta: np.ndarray
    H: np.ndarray
    scatter: np.ndarray
    avgeffectsby: str
    leaveout: bool
    anticipation: int = 0

    def covariance(self, contrasts, *, allow_unidentified=False):
        """Covariance for contrasts of the original cell estimates.

        Averaging is contrast-specific. Coarser residual groups therefore need
        fresh smart averages, rather than a linear transform of the cell vcov.
        """
        A = np.atleast_2d(contrasts)
        P, M = A.shape
        T, G = len(self.times), len(self.groups)
        weights = np.zeros((G, P, T))
        labels = {}
        cell_groups = []
        for j, (g, t) in enumerate(self.cells):
            k = int(np.searchsorted(self.groups, g))
            weights[k, :, t - self.times[0]] = A[:, j] / self.sizes[k]
            label = {"cohort_time": (g, t), "event_time": t - g, "cohort": g, "all": 0}[
                self.avgeffectsby
            ]
            cell_groups.append(labels.setdefault(label, len(labels)))
        R = len(labels)
        residual_means = np.zeros((G, T))
        denom, numer = np.zeros((P, R)), np.zeros((P, R))

        def group_totals(k):
            totals, effects = np.zeros((P, R)), np.zeros((P, R))
            for j, (g, t) in enumerate(self.cells):
                if g == self.groups[k]:
                    ti, r = t - self.times[0], cell_groups[j]
                    totals[:, r] += weights[k, :, ti]
                    effects[:, r] += weights[k, :, ti] * residual_means[k, ti]
            return totals, effects

        for k, g in enumerate(self.groups):
            use = self.times < g - self.anticipation if g else np.ones(T, dtype=bool)
            mu = self.means[k] - self.Z @ self.theta
            residual_means[k] = mu - mu[use].mean()
            totals, effects = group_totals(k)
            denom += self.sizes[k] * totals**2
            numer += self.sizes[k] * totals * effects
        averages = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
        target_derivative = A @ self.H
        V = np.zeros((P, P))
        unidentified = np.zeros(P, dtype=bool)
        for k, g in enumerate(self.groups):
            use = self.times < g - self.anticipation if g else np.ones(T, dtype=bool)
            totals, _ = group_totals(k)
            leverage = np.divide(totals**2, denom, out=np.zeros_like(totals), where=denom > 0)
            raw = weights[k].copy()
            raw[:, use] = -target_derivative @ self.Z[use].T
            center = np.zeros(P)
            for j, (cg, t) in enumerate(self.cells):
                if cg != g:
                    continue
                ti, r = t - self.times[0], cell_groups[j]
                w = weights[k, :, ti]
                lev = leverage[:, r]
                if self.leaveout:
                    invalid = (w != 0) & (lev >= 1 - 1e-12)
                    if np.any(invalid) and not allow_unidentified:
                        raise ValueError(
                            "BJS leaveout requires more than one contributing unit in each residual group; use broader avgeffectsby groups"
                        )
                    unidentified |= invalid
                    raw[:, ti] = np.divide(w, 1 - lev, out=np.zeros_like(w), where=lev < 1 - 1e-12)
                center += raw[:, ti] * averages[:, r]
            # Scatter was formed after local elimination of unit intercepts.
            # A PSD root avoids cancellation when residual variation is tiny.
            values, vectors = np.linalg.eigh(self.scatter[k])
            linear = raw @ (vectors * np.sqrt(np.maximum(values, 0)))
            mean_score = raw @ residual_means[k] - center
            V += linear @ linear.T + self.sizes[k] * np.outer(mean_score, mean_score)
        V = (V + V.T) / 2
        V[unidentified, :] = np.nan
        V[:, unidentified] = np.nan
        return V
