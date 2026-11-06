# Same balanced unconditional targets as the Python benchmark. CSV read is timed.
s <- jsonlite::fromJSON(commandArgs(TRUE)[1])
p <- switch(s$model, cs="did", sunab="fixest", bjs="didimputation")
if (!requireNamespace(p, quietly=TRUE)) stop(paste("Missing package:",p))
peak <- function() {
  z <- readLines("/proc/self/status")
  as.numeric(strsplit(trimws(sub("VmHWM:","",grep("^VmHWM:",z,value=TRUE)))," +")[[1]][1])/1024
}
baseline <- peak()
start <- proc.time()[[3]]
d <- read.csv(s$data)
if(s$model=="cs") {
  r <- did::att_gt(yname="y", tname="time", idname="id", gname="first_treat",
                   xformla=~1, data=d, est_method="reg", bstrap=FALSE, cband=FALSE)
  checksum <- sum(r$att)
} else if(s$model=="sunab") {
  d$gref <- ifelse(d$first_treat==0, max(d$time)+1, d$first_treat)
  r <- fixest::feols(y ~ sunab(gref,time) | id + time, data=d, vcov=~id,
                     nthreads=s$threads, ssc=fixest::ssc(K.adj=FALSE,G.adj=FALSE))
  checksum <- sum(coef(r, agg=FALSE))
} else {
  fixest::setFixest_nthreads(s$threads)
  r <- didimputation::did_imputation(data=d,yname="y",tname="time",gname="first_treat",idname="id",horizon=TRUE)
  checksum <- sum(r$estimate)
}
cat("RESULT ", jsonlite::toJSON(list(wall_s=unname(proc.time()[[3]]-start),
  peak_memory_mb=peak(),baseline_memory_mb=baseline,peak_disk_mb=0,
  n_obs=nrow(d),checksum=checksum),auto_unbox=TRUE),"\n",sep="")
