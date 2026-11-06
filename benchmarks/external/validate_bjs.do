* Run from the repository root with Stata and did_imputation/reghdfe/ftools installed.
* Independent execution target: the original authors' implementation.
clear all
set more off
capture mkdir benchmarks/results/stata_bjs
log using benchmarks/results/stata_bjs/validation.log, text replace
which did_imputation
which reghdfe
import delimited tests/data/reference_panel.csv, clear
isid id time
gen Ei = first_treat
replace Ei = . if first_treat==0
gen relative = time-Ei
gen avg_all = 1

capture program drop export_bjs
program define export_bjs
    syntax, Stem(string)
    tempname B V
    matrix `B' = e(b)
    matrix `V' = e(V)
    local names : colnames `B'
    local k = colsof(`B')
    preserve
    clear
    set obs `k'
    gen str40 term = ""
    gen double estimate = .
    gen double se = .
    forvalues j=1/`k' {
        local name : word `j' of `names'
        replace term = "`name'" in `j'
        replace estimate = el(`B',1,`j') in `j'
        replace se = sqrt(el(`V',`j',`j')) in `j'
    }
    export delimited term estimate se using "`stem'.csv", replace
    clear
    svmat double `V', names(v)
    export delimited using "`stem'_vcov.csv", replace
    restore
end

foreach avg in cohort_time event_time cohort all {
    if "`avg'"=="cohort_time" local by Ei time
    if "`avg'"=="event_time" local by relative
    if "`avg'"=="cohort" local by Ei
    if "`avg'"=="all" local by avg_all
    forvalues lo=0/1 {
        local option
        local label plugin
        if `lo' {
            local option leaveout
            local label leaveout
        }
        * minn(0) deliberately matches diff-stream's explicit no-suppression convention.
        did_imputation y id time Ei, allhorizons avgeffectsby(`by') cluster(id) minn(0) `option' tol(1e-10) maxit(10000)
        export_bjs, stem("benchmarks/results/stata_bjs/bjs_`avg'_`label'_dynamic")
        did_imputation y id time Ei, avgeffectsby(`by') cluster(id) minn(0) `option' tol(1e-10) maxit(10000)
        export_bjs, stem("benchmarks/results/stata_bjs/bjs_`avg'_`label'_simple")
    }
}
log close
