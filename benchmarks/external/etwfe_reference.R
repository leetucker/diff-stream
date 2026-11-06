# Execute the pinned authors' unmodified etwfe/emfx commands for linear models.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; out <- args[2]; source_dir <- args[3]
library(fixest); library(Formula); library(data.table); library(marginaleffects)
source(file.path(source_dir,'etwfe.R')); source(file.path(source_dir,'emfx.R'))
setFixest_nthreads(2); setDTthreads(2)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_result <- function(m, prefix, kind) {
 v <- vcov(m)
 d <- as.data.frame(m)
 # emfx can remove placeholder output groups while retaining the original
 # marginaleffects covariance. Align by original sorted group keys, never by SE.
 if(nrow(v)!=nrow(d)) {
  by <- switch(kind,cells=c('first_treat','time'),simple='.Dtreat',event='event',group='first_treat',calendar='time')
  nd <- as.data.frame(attr(m,'marginaleffects')@newdata)
  full <- unique(nd[by]); full <- full[do.call(order,full),,drop=FALSE]
  stopifnot(nrow(full)==nrow(v))
  key <- function(x) do.call(paste,c(x[by],sep='|'))
  ix <- match(key(d),key(full)); stopifnot(!anyNA(ix))
  v <- v[ix,ix,drop=FALSE]
 }
 finite <- is.finite(d$std.error)
 stopifnot(max(abs(diag(v)[finite]-d$std.error[finite]^2))<1e-8)

 write.csv(d,file.path(out,paste0(prefix,'_',kind,'.csv')),row.names=FALSE)
 write.table(v,file.path(out,paste0(prefix,'_',kind,'_vcov.csv')),sep=',',row.names=FALSE,col.names=FALSE)
}
run_one <- function(d, label, control, fe, case, A=0) {
 d$first_treat[d$first_treat!=0] <- d$first_treat[d$first_treat!=0]-A
 f <- if(case=='plain_equal') y~1 else if(case=='mpdta') y~x else y~x+z
 weight <- if(case %in% c('plain_equal','numeric_equal','mpdta')) NULL else if(case=='numeric_unit') ~unit_w else ~w
 fit <- etwfe(f,tvar='time',gvar='first_treat',data=d,ivar=if(fe=='unit') 'id' else NULL,
              cgroup=control,fe=if(fe=='cohort') 'none' else 'feo',gref=0,weights=weight,vcov=~id,
              ssc=ssc(K.adj=FALSE,G.adj=FALSE),fixef.rm='none',fixef.tol=1e-10)
 for(inference in c('conditional','unconditional')) {
  prefix <- paste(label,control,fe,case,A,inference,sep='_')
  vc <- if(inference=='conditional') TRUE else vcovUnconditional(type='HC0',cluster=~id)
  dat <- as.data.table(eval(fit$call$data,fit$call_env))
  dat <- dat[dat$first_treat!=0 & dat$.Dtreat,]
  dat$N <- if(is.null(weight)) 1 else dat[[if(case=='numeric_unit') 'unit_w' else 'w']]
  model <- fit; class(model) <- setdiff(class(model),'etwfe')
  cells <- slopes(model,newdata=dat,variables='.Dtreat',by=c('first_treat','time'),wts='N',vcov=vc)
  write_result(cells,prefix,'cells')
  for(kind in c('event','simple','group','calendar')) {
   if(A>0 && kind!='event') {
    # Windowing can disable the analytic model-matrix path. Use accurate
    # centered differences for this R fallback; do not relax test tolerances.
    m <- emfx(fit,type=kind,vcov=vc,window=c(-A,Inf),numderiv=list(method='fdcenter',eps=1e-4))
   } else m <- emfx(fit,type=kind,vcov=vc)
   write_result(m,prefix,kind)
  }
 }
}
if(length(args)>3 && args[4]=='smoke') {
 d <- read.csv(file.path(root,'tests/data/richer_panel.csv'))
 run_one(d,'smoke','never','cohort','numeric_varying')
 quit(status=0)
}
if(length(args)>3 && args[4]=='anticipation') {
 for(panel in c('balanced','unbalanced')) {
  d <- read.csv(file.path(root,'tests/data',if(panel=='balanced') 'richer_panel.csv' else 'richer_unbalanced_panel.csv'))
  for(control in c('never','notyet')) run_one(d,panel,control,'cohort','numeric_varying',1)
 }
 writeLines(capture.output(sessionInfo()),file.path(out,'R_session.txt'))
 quit(status=0)
}
for(panel in c('balanced','unbalanced')) {
 d <- read.csv(file.path(root,'tests/data',if(panel=='balanced') 'richer_panel.csv' else 'richer_unbalanced_panel.csv'))
 for(control in c('never','notyet')) for(fe in c('cohort','unit')) {
  for(case in c('plain_equal','numeric_varying')) run_one(d,panel,control,fe,case)
 }
 for(control in c('never','notyet')) for(case in c('numeric_equal','numeric_unit')) run_one(d,panel,control,'cohort',case)
 # Anticipation: shift the R adoption date and retain actual dates in Python.
 for(control in c('never','notyet')) run_one(d,panel,control,'cohort','numeric_varying',1)
}
mp <- read.csv(file.path(root,'tests/data/mpdta.csv'))
names(mp)[names(mp)=='first.treat'] <- 'first_treat'
mp$id <- mp$countyreal; mp$time <- mp$year; mp$y <- mp$lemp; mp$x <- mp$lpop
for(control in c('never','notyet')) run_one(mp,'mpdta',control,'cohort','mpdta')
writeLines(capture.output(sessionInfo()),file.path(out,'R_session.txt'))
