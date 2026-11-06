* Launched by RUN_STATA.do from the extracted bundle root.
clear all
set more off
set varabbrev off
capture mkdir results
log using results/validation.log, text replace
display "Stata version: " c(stata_version)
display "Operating system: " c(os)
which did_imputation
which reghdfe
which ftools
foreach command in did_imputation reghdfe ftools {
    quietly findfile `command'.ado
    copy "`r(fn)'" "results/`command'.ado", replace
}

capture program drop export_bjs
program define export_bjs
    syntax, Stem(string) [Leads(integer 0)]
    tempname B V
    matrix `B' = e(b)
    matrix `V' = e(V)
    if `leads'>0 {
        matrix `B' = `B'[1,1..`leads']
        matrix `V' = `V'[1..`leads',1..`leads']
    }
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
    format estimate se %24.17g
    export delimited term estimate se using "results/`stem'.csv", datafmt replace
    clear
    set obs `k'
    svmat double `V', names(v)
    format v* %24.17g
    export delimited using "results/`stem'_vcov.csv", datafmt replace
    restore
end

capture program drop run_case
program define run_case
    syntax, Name(string) Input(string) Anticipation(integer) Pretrends(integer) Weighted(integer)
    import delimited "`input'", asdouble clear
    isid id time
    gen double Ei = first_treat
    replace Ei = . if first_treat==0
    gen double relative = time-Ei
    gen byte avg_all = 1
    * Simple ATT includes actual post-treatment rows, excluding anticipation rows.
    gen double actual_post = first_treat>0 & time>=first_treat
    local weight
    if `weighted' local weight [aw=sampling_weight]
    local leads
    forvalues j=1/`pretrends' {
        gen byte lead`j' = Ei<. & time-Ei+`anticipation'==-`j'
        local leads `leads' lead`j'
    }
    tempfile input_data
    quietly save `input_data'
    * Same clean regression and SSC inputs as authors' pretrend implementation.
    capture noisily reghdfe y `leads' `weight' if missing(Ei) | time<Ei-`anticipation', absorb(id time) cluster(id)
    local clean_rc = _rc
    if `clean_rc'==0 {
        tempname C
        matrix `C' = (e(N),e(df_m),e(df_a),e(N_clust),e(df_r))
        matrix colnames `C' = N df_m df_a clusters df_r
        export_bjs, stem("`name'_clean") leads(`pretrends')
        preserve
        clear
        set obs 1
        svmat double `C', names(col)
        format * %24.17g
        export delimited using "results/`name'_clean_meta.csv", datafmt replace
        restore
    }
    foreach avg in cohort_time event_time cohort all {
        if "`avg'"=="cohort_time" local by Ei time
        if "`avg'"=="event_time" local by relative
        if "`avg'"=="cohort" local by Ei
        if "`avg'"=="all" local by avg_all
        foreach label in plugin leaveout {
            local option
            if "`label'"=="leaveout" local option leaveout
            foreach kind in dynamic simple {
                * Reset even after a failed fit; never reuse temporary estimator state.
                quietly use `input_data', clear
                local target allhorizons
                if "`kind'"=="simple" local target wtr(actual_post)
                local stem `name'_`avg'_`label'_`kind'
                capture noisily did_imputation y id time Ei `weight', `target' shift(`anticipation') pretrends(`pretrends') avgeffectsby(`by') cluster(id) minn(0) `option' tol(1e-10) maxit(10000)
                local rc = _rc
                post STATUS ("`stem'") (`rc') (`clean_rc')
                if `rc'==0 {
                    tempname M
                    matrix `M' = (e(N),e(pre_F),e(pre_p),e(pre_df))
                    matrix colnames `M' = N pre_F pre_p pre_df
                    export_bjs, stem("`stem'")
                    preserve
                    clear
                    set obs 1
                    svmat double `M', names(col)
                    format * %24.17g
                    export delimited using "results/`stem'_meta.csv", datafmt replace
                    restore
                }
            }
        }
    }
end

postfile STATUS str100 stem int rc clean_rc using results/status.dta, replace
do cases.do
postclose STATUS
use results/status.dta, clear
list if rc!=0 | clean_rc!=0
export delimited using results/status.csv, replace
display "BJS EXTENDED VALIDATION FINISHED"
log close
