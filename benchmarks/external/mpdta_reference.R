# Documented did empirical example, plus a deterministic specification matrix.
args <- commandArgs(TRUE)
data_path <- args[1];out <- args[2]
dir.create(dirname(data_path),recursive=TRUE,showWarnings=FALSE)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
data(mpdta,package='did')
d <- mpdta[order(mpdta$countyreal,mpdta$year),]
stopifnot(nrow(d)==2500,length(unique(d$countyreal))==500,
          all(sort(unique(d$year))==2003:2007),all(table(d$countyreal)==5),!anyNA(d))
write.csv(d,data_path,row.names=FALSE)
if(length(args)>2 && args[3]=='--data-only') quit(status=0)
xi <- as.matrix(read.csv(gzfile(file.path(out,'multipliers.csv.gz')),header=FALSE))
ids <- sort(unique(d$countyreal));stopifnot(nrow(xi)==length(ids))
xi_current <- NULL
# Process-local common draws isolate inference agreement from cross-language RNG.
assignInNamespace('run_multiplier_bootstrap',function(inf.func,biters,pl=FALSE,cores=1) {
 stopifnot(nrow(inf.func)==nrow(xi_current),biters==ncol(xi_current))
 t(xi_current)%*%as.matrix(inf.func)/nrow(inf.func)
},ns='did')
pretests <- list()
headline <- list()
for(design in c('constant','lpop')) for(control in c('nevertreated','notyettreated')) for(base in c('varying','universal')) for(method in c('reg','ipw','dr')) {
 a <- did::att_gt(yname='lemp',tname='year',idname='countyreal',gname='first.treat',
  xformla=if(design=='constant') ~1 else ~lpop,data=d,control_group=control,
  base_period=base,est_method=method,bstrap=FALSE,cband=FALSE)
 prefix <- paste(design,control,base,method,sep='_')
 keep <- if(base=='universal') a$t!=a$group-1 else rep(TRUE,length(a$t))
 IF <- as.matrix(a$inffunc)
 V <- crossprod(IF)/a$n^2
 write.csv(data.frame(g=a$group,t=a$t,att=a$att,se=a$se),file.path(out,paste0(prefix,'_cells.csv')),row.names=FALSE)
 write.csv(V,file.path(out,paste0(prefix,'_cells_vcov.csv')),row.names=FALSE)
 clean <- a$t<a$group & keep
 W <- as.numeric(t(a$att[clean])%*%solve(V[clean,clean,drop=FALSE],a$att[clean]))
 pretests[[length(pretests)+1]] <- data.frame(case=prefix,statistic=W,df=sum(clean),pvalue=pchisq(W,sum(clean),lower.tail=FALSE))
 canonical <- control=='nevertreated' && base=='varying' && method=='dr'
 if(canonical) {
  order <- match(as.numeric(rownames(IF)),ids);stopifnot(!anyNA(order))
  xi_current <- xi[order,,drop=FALSE]
  params <- a$DIDparams;params$biters <- ncol(xi);params$alp <- .05
 }
 for(kind in c('dynamic','simple','group','calendar','balanced')) {
  z <- if(kind=='balanced') did::aggte(a,type='dynamic',balance_e=1,bstrap=FALSE,cband=FALSE) else did::aggte(a,type=kind,bstrap=FALSE,cband=FALSE)
  key <- switch(kind,dynamic='dynamic.inf.func.e',balanced='dynamic.inf.func.e',group='selective.inf.func.g',calendar='calendar.inf.func.t',simple='simple.att')
  I <- as.matrix(z$inf.function[[key]])
  levels <- z$egt;b <- if(kind=='simple') z$overall.att else z$att.egt
  se <- if(kind=='simple') z$overall.se else z$se.egt
  select <- if(kind %in% c('dynamic','balanced') && base=='universal') levels != -1 else rep(TRUE,length(b))
  I <- I[,select,drop=FALSE];b <- b[select];se <- se[select];levels <- levels[select]
  names <- if(kind=='simple') 'ATT' else paste0(switch(kind,dynamic='event',balanced='event',group='cohort',calendar='calendar'),'::',levels)
  Vz <- crossprod(I)/a$n^2
  stopifnot(max(abs(diag(Vz)-se^2))<1e-12)
  write.csv(data.frame(term=names,estimate=b,se=se),file.path(out,paste0(prefix,'_',kind,'.csv')),row.names=FALSE)
  write.csv(Vz,file.path(out,paste0(prefix,'_',kind,'_vcov.csv')),row.names=FALSE)
  if(canonical) {
   mb <- did:::mboot(I,params)
   write.csv(data.frame(term=names,se=mb$se,critical=as.numeric(mb$crit.val),lower=b-as.numeric(mb$crit.val)*mb$se,upper=b+as.numeric(mb$crit.val)*mb$se),file.path(out,paste0(prefix,'_',kind,'_bootstrap.csv')),row.names=FALSE)
   con <- gzfile(file.path(out,paste0(prefix,'_',kind,'_draws.csv.gz')),'wt')
   write.csv(t(xi_current)%*%I/a$n,con,row.names=FALSE);close(con)
  }
  if(kind=='simple') headline[[length(headline)+1]] <- data.frame(case=prefix,att=b,se=se)
 }
}
write.csv(do.call(rbind,pretests),file.path(out,'pretests.csv'),row.names=FALSE)
write.csv(do.call(rbind,headline),file.path(out,'simple_att.csv'),row.names=FALSE)
write.csv(data.frame(cohort=as.numeric(names(table(d$first.treat[d$year==2003]))),n_units=as.vector(table(d$first.treat[d$year==2003]))),file.path(out,'cohorts.csv'),row.names=FALSE)
writeLines(c(paste('R',getRversion()),paste('did',packageVersion('did')),paste('DRDID',packageVersion('DRDID')),paste('did package license',utils::packageDescription('did')$License),
 'mpdta from installed did package; 500 counties, 2500 rows, consecutive annual 2003-2007; original column names and calendar years retained.',
 'Published documented example: https://bcallaway11.github.io/did/articles/did-basics.html ; data subset from Callaway and SantAnna (2021).',
 '24 analytic specifications: constant/lpop x reg/ipw/dr x never/notyet controls x varying/universal base.',
 'Unit clustering, CRV0, no weights, anticipation=0. Analytic reference SEs disable package bootstrap.',
 'Full joint aggregate covariance uses actual R aggte influence functions.',
 'Two canonical DR/never/varying models use common Rademacher draws, seed 815, 999 repetitions, sorted county IDs.',
 'Process-local R draw generator replacement; actual mboot IQR SE and type-1 critical-value code unchanged. Not same-seed R RNG reproduction.',
 'This reproduces documented empirical specifications; it is not the complete paper replication and does not execute Stata.'),file.path(out,'versions.txt'))
