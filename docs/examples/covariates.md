# Covariates in Callaway-Sant'Anna

Outcome regression, inverse probability weighting and the doubly robust estimator, with two covariates. The diagnostics count the nuisance models and the passes over the data they took: comparisons with the same sample and covariate period share a model, and all are fitted together.

```{literalinclude} ../../examples/conditional_cs.py
:language: python
:start-at: import
```

Output:

```{literalinclude} output/conditional_cs.txt
:language: text
```
