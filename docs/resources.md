# Resources

```python
from diff_stream import cs_stream

p = cs_stream(
    "y ~ 1",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    preprocess_mb=16,
    batch_units=4096,
    memory_gb=1.0,
    profile=True,
)
print(p)
print(p.recommend())
```

Profiling now stages a snapshot, validates through bounded disk partitions, and
returns the family Profile without fitting. It needs scratch disk. The profile
records the partition count and largest decoded leaf, numerical buffers, runtime
allowances and expected disk. Estimates remain heuristic.

`preprocess_mb` is a decimal MB working allowance for decoded partition data and
sort/validation copies. Oversized partitions are recursively split using unit
hashes before sorting. Validation holds one partition's units at a time, with
no global unit-ID table. Smaller allowances use more files and I/O. A single
unit that exceeds the allowance cannot be split; invalid extreme unit skew is
rejected, and a valid oversized unit requires more preprocessing memory.

`memory_gb` sets the total process budget through stream-limits, including
current RSS and a reserve. Guards run before snapshot execution, while reading
and partitioning, before decoding/sorting leaves, and before numerical buffers.
Unused Arrow pool pages are released once preparation readers/writers close.
Source queries, reader/writer buffers and allocator retention are separate from
the decoded partition allowance. A source LazyFrame containing its own global
operations can require additional memory. The guard does not create an OS-level
memory ceiling. Cluster job limits remain the enforcement mechanism.

Set `STREAM_LIMITS_MAX_THREADS` and `STREAM_LIMITS_MEMORY_GB` before import so
scheduler defaults also configure supported runtime pools and spill budgets.
`threads=` limits Numba/BLAS at fit time; Polars sizes its pool on import.

`workdir` is a parent for a unique run directory. The package removes its own
snapshot, hash partitions and sorted leaves on success or Python exceptions,
preserving unrelated files. An external kill can leave the run directory behind.
Weighted and native unbalanced BJS retain sorted observed-row partitions in a separate
result-owned directory for contrast-specific aggregation. Call `fit.cleanup()`
after creating the desired aggregates; garbage collection also removes them.
See [unbalanced panels](unbalanced_panels.md) for the complete lifecycle.

Native unbalanced SA uses exact within-cohort time-effect systems and bounded
score batches, avoiding a staged row-by-interaction design. Its numerical state
grows with periods/cohorts and joint covariance size, rather than retaining an
all-unit score or interaction table. Source execution, reader/writer buffers and
allocator retention still belong in the process budget.

The local eight-million-row check completed below a parent-monitored 1 GB RSS
ceiling. See the [benchmark notes](benchmarks/index.md) for measurements and
sampling limitations, and `benchmarks/limited_memory.py` to reproduce the check.

The native SA import/working-set probe compares 10,000, 100,000 and one million
units. The specialized fit reaches about 502 MiB peak RSS at one million units
(five periods, 4.08 million observed rows), versus 1,434 MiB with the former
general adapter. The numerical pass peaks at about 407 MiB; preparation sets the
overall peak. See [benchmark details](benchmarks/index.md) for the warm-up,
tiled-input convention and sampling limits. This does not establish flat RSS
or a hard memory ceiling at larger scales or more effects.

## Conditional CS and observed-panel checks

The [expanded resource report](resource_checks.md) covers up to one million
units, 33-column conditional designs, more periods/cohorts, late entry/early
exit, string identifiers, large Parquet row groups and tighter numerical buffers.
It separates preprocessing, nuisance and aggregation peaks and records both
sampled process-tree RSS and the process high-water mark. Cases are run serially
under explicit budgets and timeouts, with atomic results and resumable progress.

These checks found and fixed a large-sample logit line-search stall. The
summed objective now uses compensated accumulation and a roundoff allowance
that scales with its magnitude; the coefficient convergence tolerance and
unpenalized estimator are unchanged. The report retains the earlier timeout and
probe evidence alongside the corrected measurements and R comparisons.

The runner uses only installed local packages and data generation; it performs
no network requests or installations. See `benchmarks/resource_checks.py` and
`benchmarks/resource_report.py` in the source checkout for reproduction.

The [stronger memory review](memory_checks.md) adds 10-million-row conditional-CS
checks, long unit strings, shuffled rows, much larger source row groups and unused
wide columns. It records clean headroom refusals separately from completed fits.
The [kernel-enforced follow-up](kernel_memory_checks.md) now verifies 12 selected
cases on this WSL host, using a delegated cgroup v1 memory controller outside
the sandbox. It distinguishes kernel-charged memory from RSS and verifies the
cap through a successful control, automatic limit detection and a kernel OOM test.
