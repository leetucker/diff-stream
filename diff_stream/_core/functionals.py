"""Streaming normalized means and their smooth-functional derivatives.

These primitives have no panel, date, treatment, or cohort conventions.
"""

import numpy as np


class NormalizedMeans:
    def __init__(self, terms, width):
        self.weight = np.zeros(terms)
        self.value = np.zeros(terms)
        self.x = np.zeros((terms, width))
        self.xvalue = np.zeros((terms, width))

    def update(self, weights, values, X):
        self.weight += weights.sum(axis=0)
        self.value += (weights * values).sum(axis=0)
        self.x += weights.T @ X
        self.xvalue += (weights * values).T @ X

    def finish(self):
        if np.any(self.weight <= 0):
            raise ValueError("a normalized mean has no contributing observations after trimming")
        self.mean = self.value / self.weight
        self.mean_x = self.x / self.weight[:, None]
        self.centered_x = self.xvalue / self.weight[:, None] - self.mean[:, None] * self.mean_x

    def scores(self, weights, values, signs):
        return (weights * (values - self.mean) / self.weight) @ signs
