# BJS pre-trend test

`pretrend_leads=1` fits a separate regression on the untreated observations with an indicator for the period before treatment, and tests it. The treatment effects are unchanged.

```{literalinclude} ../../examples/bjs_placebos.py
:language: python
:start-at: import
```

Output:

```{literalinclude} output/bjs_placebos.txt
:language: text
```
