# R did influence/mboot references with shared unit-level Rademacher multipliers.
# The draw generator is replaced in this R PROCESS ONLY, to compare identical
# draws across languages. Estimation, aggregation, IQR scaling and critical-value
# computation use installed did. No installed package file is changed.
args <- commandArgs(TRUE)
d0 <- read.csv(args[1]); out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
xi <- as.matrix(read.csv(gzfile(file.path(out,'multipliers.csv.gz')),header=FALSE))
xi_current <- NULL
assignInNamespace('run_multiplier_bootstrap', function(inf.func,biters,pl=FALSE,cores=1) {
 stopifnot(nrow(xi_current)==nrow(inf.func),biters==ncol(xi_current))
 t(xi_current) %*% as.matrix(inf.func)/nrow(inf.func)
}, ns='did')
save_csv <- function(x,path) { con <- gzfile(path,'wt');write.csv(x,con,row.names=FALSE);close(con) }
export <- function(I,b,names,prefix,params) {
 I <- as.matrix(I)
 stopifnot(nrow(I)==nrow(xi_current))
 params$biters <- ncol(xi_current);params$alp <- .05
 mb <- did:::mboot(I,params)
 draws <- t(xi_current)%*%I/nrow(I)
 save_csv(data.frame(term=names,estimate=b,se=mb$se,critical=as.numeric(mb$crit.val),lower=b-as.numeric(mb$crit.val)*mb$se,upper=b+as.numeric(mb$crit.val)*mb$se),paste0(prefix,'.csv.gz'))
 save_csv(draws,paste0(prefix,'_draws.csv.gz'))
}
scenarios <- list(c('constant','dr','nevertreated','varying'),
 c('numeric','reg','nevertreated','varying'),c('numeric','ipw','notyettreated','universal'))
for(control in c('nevertreated','notyettreated')) for(base in c('varying','universal'))
 scenarios[[length(scenarios)+1]] <- c('numeric','dr',control,base)
pretests <- list()
for(sample in c('balanced','unbalanced')) for(weighted in c(FALSE,TRUE)) for(A in c(0,2)) for(scenario in scenarios) {
 design <- scenario[1];method <- scenario[2];control <- scenario[3];base <- scenario[4]
 d <- d0;d$w <- .2+d$id%%9
 if(sample=='unbalanced') d <- d[!(d$id%%4==0 & d$time %in% c(3,7)),]
 shift <- if(base=='varying') A else 0
 d$gref <- ifelse(d$first_treat==0,0,d$first_treat-shift)
 fit <- function(actual=FALSE) did::att_gt(yname='y',tname='time',idname='id',gname=if(actual) 'first_treat' else 'gref',data=d,
  xformla=if(design=='numeric') ~x+z else ~1,est_method=method,weightsname=if(weighted) 'w' else NULL,
  anticipation=if(actual) A else A-shift,control_group=control,base_period=base,
  allow_unbalanced_panel=(sample=='unbalanced'),bstrap=FALSE,cband=FALSE)
 a <- fit();IF <- as.matrix(a$inffunc)
 ids <- as.integer(rownames(IF));stopifnot(!anyNA(ids),all(sort(ids)==0:399))
 xi_current <- xi[ids+1,,drop=FALSE]
 g <- a$group+shift;t <- a$t
 keep <- if(base=='universal') t != g-A-1 else rep(TRUE,length(t))
 prefix <- paste(sample,as.integer(weighted),A,design,control,base,method,sep='_')
 export(IF[,keep,drop=FALSE],a$att[keep],paste0('g::',g[keep],':t::',t[keep]),file.path(out,paste0(prefix,'_cells')),a$DIDparams)
 pre <- (t < g-A)&keep
 V <- crossprod(IF[,pre,drop=FALSE])/a$n^2
 W <- as.numeric(t(a$att[pre])%*%solve(V,a$att[pre]))
 pretests[[length(pretests)+1]] <- data.frame(case=prefix,statistic=W,df=sum(pre),pvalue=pchisq(W,sum(pre),lower.tail=FALSE))
 for(kind in c('dynamic','simple','group','calendar')) {
  aa <- if(kind!='dynamic' && shift>0) fit(TRUE) else a
  z <- did::aggte(aa,type=kind,bstrap=FALSE,cband=FALSE)
  key <- switch(kind,dynamic='dynamic.inf.func.e',group='selective.inf.func.g',calendar='calendar.inf.func.t',simple='simple.att')
  I <- as.matrix(z$inf.function[[key]])
  levels <- z$egt
  b <- if(kind=='simple') z$overall.att else z$att.egt
  if(kind=='dynamic') {
   levels <- levels-shift
   select <- if(base=='universal') levels != -A-1 else rep(TRUE,length(levels))
   I <- I[,select,drop=FALSE];levels <- levels[select];b <- b[select]
  }
  names <- if(kind=='simple') 'ATT' else paste0(switch(kind,dynamic='event',group='cohort',calendar='calendar'),'::',levels)
  export(I,b,names,file.path(out,paste0(prefix,'_',kind)),aa$DIDparams)
 }
}
write.csv(do.call(rbind,pretests),file.path(out,'pretests.csv'),row.names=FALSE)
writeLines(c(paste('R',getRversion()),paste('did',packageVersion('did')),paste('BMisc',packageVersion('BMisc')),
 'Common Rademacher draws supplied from hdfe_stream.core.wildboot.WEIGHTS; Python seed 815, unit-major 400x199; fixture row=id+1.',
 'Process-local replacement of run_multiplier_bootstrap supplies common draws; installed did mboot computes IQR SEs/type-1 critical values.',
 'Varying-base A=2 uses effective-onset labels; actual-treatment post aggregates use actual cohort labels.',
 'Pretests explicitly select clean t<g-A cells and use the R analytic covariance.'),file.path(out,'versions.txt'))
