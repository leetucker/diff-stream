"""Descriptive TWFE decomposition and already-treated-control contributions."""

from diff_stream import bacon_decomp
from diff_stream.simulate import simulate_panel

data = simulate_panel(2000, 8, seed=72)
bacon = bacon_decomp(
    "y ~ 1",
    data,
    panel=("id", "time"),
    cohort="first_treat",
    threads=2,
)
print("TWFE:", bacon.twfe_estimate)
print("Reconstructed:", bacon.decomposed_estimate)
print(bacon.summary())
print(bacon.comparisons())
