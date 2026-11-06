# When every unit is eventually treated

No unit is never treated, so the last-treated cohort serves as the comparison until it is treated (`last_treated_controls="if_needed"`, with a warning). BJS, the two-stage estimator and ETWFE on the same panel.

```{literalinclude} ../../examples/reference_populations.py
:language: python
:start-at: import
```

Output:

```{literalinclude} output/reference_populations.txt
:language: text
```
