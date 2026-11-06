"""Joint covariance of cell scores and cohort-share scores, by cohort.

The CS covariance sums, over units i, the outer products of [scores_i, share_i]
where share_i = anchor_i * (e_{g_i} - p): e_g marks the cells of cohort g and p
is a constant vector. Two facts make this cheaper than one dense product per
batch (ClusterScoreAccumulator), with the same sums:

- A unit's scores are zero outside the cells its cohort enters (its own cells,
  and those where it is a control), so each cohort's rows add a dense product
  only into those columns. Only never-treated units touch every column.
- The share terms depend on a unit only through its cohort and anchor, so the
  share and score-share blocks need just the per-cohort sums of anchor**2 and
  of anchor * scores.

With many cohorts and periods (hundreds of cells) this removes most of the
work; with few cells it costs about the same as the dense product.
"""

import numpy as np


class CohortScoreCovariance:
    """Accumulate V = sum s s', cross = sum s h', shares = sum h h' over units.

    ``cell_cohorts`` holds each cell's cohort; ``center`` is p (zeros for an
    uncentered indicator share).
    """

    def __init__(self, cell_cohorts, center=None):
        self.cell_cohorts = np.asarray(cell_cohorts)
        M = len(self.cell_cohorts)
        self.center = np.zeros(M) if center is None else np.asarray(center, dtype=float)
        self.V = np.zeros((M, M))
        self._weighted = {}  # cohort -> sum of anchor * scores (M,)
        self._square = {}  # cohort -> sum of anchor**2
        self.n_clusters = 0

    def add(self, g, scores, anchor=None):
        """Add one row per unit: cohort labels, cell scores and share anchors."""
        anchor = np.ones(len(g)) if anchor is None else anchor
        order = np.argsort(g, kind="stable")
        labels = g[order]
        starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
        M = self.V.shape[0]
        for a, b in zip(starts[:-1], starts[1:], strict=True):
            rows = order[a:b]
            k = labels[a].item()
            S = scores[rows]
            used = np.flatnonzero(np.any(S != 0, axis=0))
            if len(used) == M:
                self.V += S.T @ S
            elif len(used):
                block = S[:, used]
                self.V[np.ix_(used, used)] += block.T @ block
            w = anchor[rows]
            self._weighted[k] = self._weighted.get(k, 0.0) + w @ S
            self._square[k] = self._square.get(k, 0.0) + float(w @ w)
        self.n_clusters += len(g)

    def blocks(self):
        """(V, cross, shares): the blocks of the joint [scores, share] sums."""
        M = len(self.cell_cohorts)
        cross, shares = np.zeros((M, M)), np.zeros((M, M))
        for k, weighted in self._weighted.items():
            h = (self.cell_cohorts == k) - self.center
            cross += np.outer(weighted, h)
            shares += self._square[k] * np.outer(h, h)
        return self.V, cross, shares
