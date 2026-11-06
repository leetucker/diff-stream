# Offline resource comparison and full covariance reference; no installs/network.
s <- jsonlite::fromJSON(commandArgs(TRUE)[1])
peak <- function() {
 z <- readLines('/proc/self/status'); as.numeric(strsplit(trimws(sub('VmHWM:','',grep('^VmHWM:',z,value=TRUE))),' +')[[1]][1])/1024
}
requireNamespace('did',quietly=TRUE)
if (requireNamespace('data.table',quietly=TRUE)) data.table::setDTthreads(s$threads)
baseline <- peak(); start <- proc.time()[[3]]
result <- tryCatch({
 d <- read.csv(s$csv_path)
 # Labels only: did requires numeric unit identifiers.
 if (!is.numeric(d$id)) d$id <- as.integer(factor(d$id))
 a <- did::att_gt(yname='y',tname='time',idname='id',gname='first_treat',data=d,
      xformla=as.formula(s$r_formula),est_method=s$method,
      allow_unbalanced_panel=(s$panel_mode=='unbalanced'),bstrap=FALSE,cband=FALSE)
 fit_s <- unname(proc.time()[[3]]-start)
 b <- did::aggte(a,type='simple',bstrap=FALSE,cband=FALSE)
 list(status='ok',fit_s=fit_s,peak_rss_mib=peak(),import_baseline_mib=baseline,
      cells=unname(as.matrix(data.frame(g=a$group,t=a$t))),beta=a$att,
      vcov=unname(as.matrix(a$V_analytical)/a$n),att=b$overall.att,se=b$overall.se,
      n_obs=nrow(d),n_units=a$n,R=as.character(getRversion()),did=as.character(packageVersion('did')))
},error=function(e) list(status='fit_failed',error=conditionMessage(e),peak_rss_mib=peak()))
cat('RESULT ',jsonlite::toJSON(result,auto_unbox=TRUE,digits=16,null='null'),'\n',sep='')
