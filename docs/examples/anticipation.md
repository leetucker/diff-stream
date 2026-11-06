# Anticipation

Units may respond two periods before treatment (`anticipation=2`): only periods before `cohort - 2` count as untreated. The cohort-period tables label those periods, and the event study reports them at event times -2 and -1.

```{literalinclude} ../../examples/anticipation.py
:language: python
:start-at: import
```

Output:

```{literalinclude} output/anticipation.txt
:language: text
```
