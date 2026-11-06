# Actual did/DRDID full group-time covariance, in canonical covariate/outcome units.
args <- commandArgs(TRUE)
jobs <- jsonlite::fromJSON(args[1],simplifyVector=FALSE)
results <- lapply(jobs,function(s) {
 warnings <- character()
 tryCatch(withCallingHandlers({
  d <- read.csv(s$csv_path)
  a <- did::att_gt(yname='y',tname='time',idname='id',gname='first_treat',data=d,
    xformla=~x+z,est_method=s$method,weightsname='w',
    allow_unbalanced_panel=s$panel_mode=='unbalanced',bstrap=FALSE,cband=FALSE,
    panel=TRUE)
  # did's wrapper does not expose custom DRDID trimming. Use its default for
  # this canonical reference; nondefault trimming cases are handled below.
  if(s$trim_level!=.995) {
   estimator <- switch(s$method,
    reg=function(y1,y0,D,covariates,i.weights,...) DRDID::reg_did_panel(y1,y0,D,covariates,i.weights=i.weights,inffunc=TRUE),
    ipw=function(y1,y0,D,covariates,i.weights,...) DRDID::std_ipw_did_panel(y1,y0,D,covariates,i.weights=i.weights,inffunc=TRUE,trim.level=s$trim_level),
    dr=function(y1,y0,D,covariates,i.weights,...) DRDID::drdid_panel(y1,y0,D,covariates,i.weights=i.weights,inffunc=TRUE,trim.level=s$trim_level))
   # Unbalanced comparisons use repeated-cross-section nuisance regressions.
   if(s$panel_mode=='unbalanced') estimator <- switch(s$method,
    reg=function(y,post,D,covariates,i.weights,...) DRDID::reg_did_rc(y,post,D,covariates,i.weights=i.weights,inffunc=TRUE),
    ipw=function(y,post,D,covariates,i.weights,...) DRDID::std_ipw_did_rc(y,post,D,covariates,i.weights=i.weights,inffunc=TRUE,trim.level=s$trim_level),
    dr=function(y,post,D,covariates,i.weights,...) DRDID::drdid_rc(y,post,D,covariates,i.weights=i.weights,inffunc=TRUE,trim.level=s$trim_level))
   a <- did::att_gt(yname='y',tname='time',idname='id',gname='first_treat',data=d,
     xformla=~x+z,est_method=estimator,weightsname='w',allow_unbalanced_panel=s$panel_mode=='unbalanced',bstrap=FALSE,cband=FALSE)
  }
  b <- did::aggte(a,type='simple',bstrap=FALSE,cband=FALSE)
  list(name=s$name,status='ok',warnings=warnings,fit=list(cells=unname(as.matrix(data.frame(g=a$group,t=a$t))),
   beta=a$att,vcov=unname(as.matrix(a$V_analytical)/a$n),att=b$overall.att,se=b$overall.se))
 },warning=function(w) { warnings <<- c(warnings,conditionMessage(w));invokeRestart('muffleWarning') }),
 error=function(e) list(name=s$name,status='refused',error=conditionMessage(e),warnings=warnings))
})
writeLines(jsonlite::toJSON(results,auto_unbox=TRUE,digits=16,null='null'),args[2])
writeLines(c(paste('R',getRversion()),paste('did',packageVersion('did')),paste('DRDID',packageVersion('DRDID'))),paste0(args[2],'.versions.txt'))
