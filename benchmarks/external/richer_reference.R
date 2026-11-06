# Offline actual did/fixest checks and independent full-unit-dummy BJS oracle.
args <- commandArgs(TRUE)
root <- args[1]; out <- args[2]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
options(warn=1)
write_table <- function(prefix, term, beta, V) {
 write.csv(data.frame(term=term,estimate=as.vector(beta),se=sqrt(diag(V))),file.path(out,paste0(prefix,'.csv')),row.names=FALSE)
 write.csv(V,file.path(out,paste0(prefix,'_vcov.csv')),row.names=FALSE)
}
for(sample in c('balanced','unbalanced')) {
 d <- read.csv(file.path(root,if(sample=='balanced') 'richer_panel.csv' else 'richer_unbalanced_panel.csv'))
 # Actual package CS, both rowwise and balanced-panel time-varying conventions.
 for(A in 0:1) for(design in c('constant','numeric')) for(control in c('nevertreated','notyettreated')) for(base in c('varying','universal')) for(method in c('reg','ipw','dr')) {
  dd <- d
  shift <- if(base=='varying') A else 0
  dd$gref <- ifelse(dd$first_treat==0,0,dd$first_treat-shift)
  a <- did::att_gt(yname='y',tname='time',idname='id',gname='gref',data=dd,
   xformla=if(design=='constant') ~1 else ~x+z,weightsname='w',
   est_method=method,base_period=base,control_group=control,anticipation=A-shift,
   allow_unbalanced_panel=(sample=='unbalanced'),faster_mode=FALSE,bstrap=FALSE,cband=FALSE)
  prefix <- paste('cs',sample,A,design,control,base,method,sep='_')
  term <- paste0('g::',a$group+shift,':t::',a$t)
  write_table(prefix,term,a$att,as.matrix(a$V_analytical)/a$n)
  for(kind in c('dynamic','simple','group','calendar')) {
   # Only the dynamic result is used with onset-shift labels; other maps target
   # actual post-treatment cells and are exported under universal actual labels.
   if(base=='varying' && kind!='dynamic') next
   z <- did::aggte(a,type=kind,bstrap=FALSE,cband=FALSE)
   key <- switch(kind,dynamic='dynamic.inf.func.e',group='selective.inf.func.g',calendar='calendar.inf.func.t',simple='simple.att')
   S <- as.matrix(z$inf.function[[key]])/a$n
   label <- if(kind=='simple') 'ATT' else paste0(switch(kind,dynamic='event::',group='cohort::',calendar='calendar::'),z$egt-if(kind=='dynamic') shift else 0)
   write_table(paste0(prefix,'_',kind),label,if(kind=='simple') z$overall.att else z$att.egt,crossprod(S))
  }
 }
 for(A in 0:1) for(spec in c('plain_varying','numeric_varying','extra_fe_varying','extra_fe_unit','extra_fe_equal')) {
  sw <- if(spec=='extra_fe_unit') d$unit_w else if(spec=='extra_fe_equal') rep(1,nrow(d)) else d$w
  dd <- d; dd$weight <- sw; dd$gref <- ifelse(d$first_treat==0,max(d$time)+100,d$first_treat)
  rhs <- if(spec=='plain_varying') '1' else 'x + I(x^2)'
  fe <- if(startsWith(spec,'extra_fe')) 'id + time + region^time' else 'id + time'
  f <- as.formula(paste('y ~ sunab(gref,time,ref.p=',-A-1,',no_agg=TRUE) + ',rhs,' | ',fe))
  sa <- fixest::feols(f,data=dd,weights=~weight,vcov=~id,fixef.rm='none',fixef.tol=1e-10,ssc=fixest::ssc(K.adj=FALSE,G.adj=FALSE))
  keep <- grep(':cohort::',names(sa$coefficients),fixed=TRUE)
  labels <- names(sa$coefficients)[keep]
  parse <- strsplit(labels,':cohort::',fixed=TRUE)
  gg <- as.integer(vapply(parse,`[`,'',2)); hh <- as.integer(sub('time::','',vapply(parse,`[`,'',1),fixed=TRUE))
  write_table(paste('sunab',sample,A,spec,sep='_'),paste0('g::',gg,':t::',gg+hh),sa$coefficients[keep],vcov(sa)[keep,keep,drop=FALSE])
  # Independent explicit unit/time/covariate/crossed-effect dummies.
  x <- cbind(model.matrix(~0+factor(id),d),model.matrix(~factor(time),d)[,-1,drop=FALSE])
  if(spec!='plain_varying') x <- cbind(x,d$x,d$x^2)
  if(startsWith(spec,'extra_fe')) for(rg in sort(unique(d$region))[-1]) for(tt in sort(unique(d$time))[-1]) x <- cbind(x,as.numeric(d$region==rg & d$time==tt))
  treated <- d$first_treat>0 & d$time>=d$first_treat-A
  x0 <- x[!treated,,drop=FALSE]
  Q <- crossprod(x0,sw[!treated]*x0)
  b <- solve(Q,crossprod(x0,sw[!treated]*d$y[!treated]))
  residual <- as.vector(d$y-x%*%b)
  cells <- unique(d[treated,c('first_treat','time')]);cells <- cells[order(cells$first_treat,cells$time),]
  M <- nrow(cells);wc <- matrix(0,nrow(d),M); counts <- numeric(M)
  for(j in seq_len(M)) {
   sel <- d$first_treat==cells$first_treat[j] & d$time==cells$time[j]
   counts[j] <- sum(sw[sel]);wc[sel,j] <- sw[sel]/counts[j]
  }
  event <- cells$time-cells$first_treat
  maps <- list(cells=diag(M),dynamic=sapply(sort(unique(event)),function(e) {z<-counts*(event==e);z/sum(z)}),simple=matrix(counts*(event>=0)/sum(counts*(event>=0)),ncol=1))
  labels <- list(cells=paste0('g::',cells$first_treat,':t::',cells$time),dynamic=paste0('event::',sort(unique(event))),simple='ATT')
  for(kind in names(maps)) {
   w1 <- wc%*%maps[[kind]]
   v <- w1;v[!treated,] <- -sw[!treated]*(x0%*%solve(Q,crossprod(x[treated,,drop=FALSE],w1[treated,,drop=FALSE])))
   averages <- if(spec=='extra_fe_varying') c('cohort_time','event_time','cohort','all') else 'cohort_time'
   for(avg in averages) for(lo in if(spec=='extra_fe_varying') c(FALSE,TRUE) else FALSE) {
    label <- switch(avg,cohort_time=paste(d$first_treat,d$time),event_time=d$time-d$first_treat,cohort=d$first_treat,all=rep(1,nrow(d)))
    scores <- matrix(0,length(unique(d$id)),ncol(w1))
    for(j in seq_len(ncol(w1))) {
     wt<-w1[treated,j];lab<-label[treated];id<-d$id[treated];rr<-residual[treated]
     cw<-ave(wt,interaction(id,lab,drop=TRUE),FUN=sum)
     denominator<-ave(wt*cw,lab,FUN=sum); numerator<-ave(wt*cw*rr,lab,FUN=sum)
     average<-ifelse(denominator>0,numerator/denominator,0)
     eps<-residual;eps[treated]<-rr-average
     if(lo) {
      leverage<-ifelse(denominator>0,cw^2/denominator,0)
      stopifnot(all(leverage[wt!=0]<1-1e-12))
      eps[treated]<-eps[treated]/(1-leverage)
     }
     scores[,j]<-as.vector(rowsum(v[,j]*eps,d$id,reorder=TRUE))
    }
    prefix<-paste('bjs',sample,A,spec,avg,if(lo) 'leaveout' else 'plugin',kind,sep='_')
    write_table(prefix,labels[[kind]],crossprod(w1,residual),crossprod(scores))
   }
  }
  # Clean-placebo regression with the SAME first-stage covariates/extra FEs.
  leads <- seq(-A-2,-A-1)
  Xlead <- sapply(leads,function(h) as.numeric(d$first_treat>0 & d$time-d$first_treat==h))
  X <- cbind(x,Xlead)[!treated,,drop=FALSE]
  ww<-sw[!treated];yy<-d$y[!treated]; qq<-crossprod(X,ww*X)
  coef<-solve(qq,crossprod(X,ww*yy));e<-yy-as.vector(X%*%coef)
  H<-solve(qq,diag(ncol(X))[,ncol(x)+seq_along(leads),drop=FALSE])
  S<-rowsum((ww*e)*(X%*%H),d$id[!treated],reorder=TRUE)
  write_table(paste('placebo',sample,A,spec,sep='_'),paste0('event::',leads),tail(coef,2),crossprod(S))
 }
}
writeLines(c(paste('R',getRversion()),paste('did',packageVersion('did')),paste('DRDID',packageVersion('DRDID')),paste('fixest',packageVersion('fixest')),
 'Actual did default time-varying weights: earlier comparison date in balanced panel; observed-date row weights in native unbalanced panel.',
 'CS aggregation unit anchor: first period balanced; mean observed weight unbalanced, matching did.',
 'SA: actual fixest joint covariate/extra-FE regression. CRV0, no degrees-of-freedom adjustment.',
 'BJS/clean placebos: independent dense R full-unit-dummy WLS and published smart/leave-out scores; NOT authors Stata execution.',
 'Varying-base anticipation references shift onset to avoid the installed did anticipation switch-at-g corner case.'),file.path(out,'versions.txt'))
