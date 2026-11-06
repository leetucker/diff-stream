* Run from the repository root on a machine with Stata.
* Requires: ssc install did_imputation; ssc install eventstudyinteract;
*           ssc install csdid; ssc install drdid; ssc install reghdfe.
* Generate input: python benchmarks/make_reference_data.py
clear all
set more off
capture mkdir benchmarks/results/stata
log using benchmarks/results/stata/validation.log, text replace
import delimited tests/data/reference_panel.csv, clear
isid id time
xtset id time
gen never = first_treat == 0
gen Ei = first_treat
replace Ei = . if never
did_imputation y id time Ei, horizons(0/3)
matrix list e(b)
matrix list e(V)
estimates save benchmarks/results/stata/bjs, replace
gen relative = time - first_treat if !never
local interactions
forvalues e=-4/3 {
    if `e' != -1 {
        local suffix = cond(`e'<0,"m"+string(abs(`e')),"p"+string(`e'))
        gen ev_`suffix' = (relative==`e') & !never
        local interactions `interactions' ev_`suffix'
    }
}
* This panel has five periods, so all supported leads/lags lie in [-4,3].
eventstudyinteract y `interactions', cohort(first_treat) control_cohort(never) absorb(id time) vce(cluster id)
matrix list e(b_iw)
matrix list e(V_iw)
estimates save benchmarks/results/stata/sunab, replace
csdid y, ivar(id) time(time) gvar(first_treat) method(reg)
matrix list e(b)
matrix list e(V)
estat event
estat simple
estimates save benchmarks/results/stata/cs, replace
log close
