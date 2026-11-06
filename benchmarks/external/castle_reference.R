# Frozen public castle snapshot -> actual fixest SA plus independent dummy OLS.
# Usage: Rscript castle_reference.R source.rda data.csv output-directory
args <- commandArgs(TRUE)
load(args[1])
d <- castle[order(castle$state,castle$year),]
out <- args[3];dir.create(out,recursive=TRUE,showWarnings=FALSE)
stopifnot(nrow(d)==550,length(unique(d$state))==50,all(table(d$state)==11),
 identical(sort(unique(d$year)),as.numeric(2000:2010)),!anyDuplicated(d[c('state','year')]),
 !anyNA(d[c('state','year','l_homicide','popwt','post')]))
write.csv(d,args[2],row.names=FALSE)
# Read exactly the same CSV that the Python runner will use.
d <- read.csv(args[2])
g <- ifelse(is.na(d$effyear),0,d$effyear)
stopifnot(sum(!duplicated(d$state) & g>0)==21,
 all(d$post==as.integer(g>0 & d$year>g)),sum(d$post!=as.integer(g>0 & d$year>=g))==21)
for(shift in 0:1) for(weighted in c(FALSE,TRUE)) {
 z <- d;z$first_treat <- ifelse(g==0,0,g+shift)
 z$gref <- ifelse(z$first_treat==0,max(z$year)+100,z$first_treat)
 w <- if(weighted) z$popwt else rep(1,nrow(z))
 f <- fixest::feols(l_homicide~sunab(gref,year,ref.p=-1)|state+year,data=z,
      weights=if(weighted) ~popwt else NULL,vcov=~state,fixef.rm='none',fixef.tol=1e-10,
      ssc=fixest::ssc(K.adj=FALSE,G.adj=FALSE))
 b <- f$coefficients; V <- vcov(f); terms <- names(b)
 h <- as.integer(sub('year::(-?[0-9]+):cohort::.*','\\1',terms))
 cohort <- as.integer(sub('.*:cohort::([0-9]+)','\\1',terms))
 target <- cohort+h
 stopifnot(length(b)==50,!anyNA(h),!anyNA(cohort))
 # Full unit/year-dummy WLS; independent design/cluster-score calculation.
 interactions <- sapply(seq_along(b),function(j) as.numeric(z$first_treat==cohort[j] & z$year==target[j]))
 X <- cbind(model.matrix(~factor(state)+factor(year),z),interactions)
 linear <- lm.wfit(X,z$l_homicide,w)
 stopifnot(linear$rank==ncol(X),!anyNA(linear$coefficients))
 selected <- tail(seq_len(ncol(X)),length(b))
 H <- solve(crossprod(X,w*X),diag(ncol(X))[,selected,drop=FALSE])
 scores <- rowsum(X*(w*linear$residuals),z$state,reorder=TRUE)
 denseV <- crossprod(scores%*%H)
 denseb <- unname(linear$coefficients[selected])
 stopifnot(max(abs(b-denseb))<1e-9,max(abs(V-denseV))<1e-9)
 prefix <- paste0('shift',shift,'_',if(weighted) 'popwt' else 'unweighted')
 write.csv(data.frame(term=terms,g=cohort,t=target,att=b,se=sqrt(diag(V)),dense_att=denseb,dense_se=sqrt(diag(denseV))),file.path(out,paste0(prefix,'_cells.csv')),row.names=FALSE)
 write.csv(unname(V),file.path(out,paste0(prefix,'_cells_vcov.csv')),row.names=FALSE)
 write.csv(unname(denseV),file.path(out,paste0(prefix,'_dense_vcov.csv')),row.names=FALSE)
 # Match fixest's aggregation shares from its actual interaction model matrix.
 mm <- model.matrix(f)
 mass <- colSums(w*abs(mm[,terms,drop=FALSE]))
 for(kind in c('dynamic','simple','cohort','calendar','balanced_dynamic')) {
  use <- if(kind=='dynamic') rep(TRUE,length(b)) else h>=0
  if(kind=='balanced_dynamic') use <- h %in% 0:2 & cohort<=max(z$year)-2
  variable <- switch(kind,dynamic=h,simple=rep(0,length(b)),cohort=cohort,calendar=target,balanced_dynamic=h)
  levels <- sort(unique(variable[use])); A <- matrix(0,length(levels),length(b))
  for(j in seq_along(levels)) {
   ix <- use & variable==levels[j];A[j,ix] <- mass[ix]/sum(mass[ix])
  }
  beta <- as.vector(A%*%b); covariance <- A%*%V%*%t(A)
  # Actual public fixest aggregation verifies its point and SE conventions.
  if(kind %in% c('dynamic','simple','cohort')) {
   actual <- aggregate(f,switch(kind,dynamic='period',simple='ATT',cohort='cohort'))
   actual_levels <- if(kind=='simple') 0 else as.integer(sub('.*::','',rownames(actual)))
   order_actual <- match(levels,actual_levels)
   stopifnot(!anyNA(order_actual),max(abs(beta-actual[order_actual,1]))<1e-10,
             max(abs(sqrt(diag(covariance))-actual[order_actual,2]))<1e-10)
  }
  write.csv(data.frame(level=levels,att=beta,se=sqrt(diag(covariance))),file.path(out,paste0(prefix,'_',kind,'.csv')),row.names=FALSE)
  write.csv(unname(covariance),file.path(out,paste0(prefix,'_',kind,'_vcov.csv')),row.names=FALSE)
  if(kind=='dynamic') {
   ix <- match(c(-4,-3,-2),levels)
   W <- as.numeric(t(beta[ix])%*%solve(covariance[ix,ix],beta[ix]))
   write.csv(data.frame(statistic=W,df=3,pvalue=pchisq(W,3,lower.tail=FALSE)),file.path(out,paste0(prefix,'_pretrend.csv')),row.names=FALSE)
  }
 }
}
writeLines(c(paste('R',getRversion()),paste('fixest',packageVersion('fixest')),
 'Unconditional SA reanalysis of Cheng and Hoekstra (2013), public did2s castle dataset.',
 '50 states, 550 rows, years 2000-2010; 21 adopting and 29 never-treated states in this snapshot.',
 'shift0 uses effyear; shift1 uses effyear+1 and matches source post exactly. Source columns unchanged.',
 'Baseline unweighted; optional popwt sensitivity uses original positive state-constant weights.',
 'Saturated all-supported cohort/event interactions, explicit ref.p=-1; no event binning or regression truncation.',
 'State CRV0: K.adj=FALSE, G.adj=FALSE; fixed cohort/observation shares. No estimated-share variance term.',
 'Full cell covariance from actual fixest, independently checked with full-dummy WLS unit-cluster scores.',
 'Full aggregate covariance propagated from fixest cell covariance; actual fixest dynamic/ATT/cohort outputs checked.',
 'Family intervals use normal critical values; fixest default t(49) p-values/intervals are not asserted.',
 'Calendar and balanced-window aggregates are explicit linear maps; no native fixest shortcut is claimed.'),file.path(out,'versions.txt'))
