# BJS clean-placebo regression: dense WLS, fixest and didimputation pretrends.
args <- commandArgs(TRUE)
d0 <- read.csv(args[1]);out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
fixest::setFixest_estimation(fixef.tol=1e-10)
fixest::setFixest_ssc(fixest::ssc(K.adj=FALSE,G.adj=FALSE),vcov_names=c("iid","hetero","cluster"))
tests <- list()
for(sample in c('balanced','unbalanced','complete','future')) for(weighted in c(FALSE,TRUE)) for(A in c(0,2)) for(L in c(1,2,3)) {
 d <- d0;d$w <- .2+d$id%%9
 if(sample %in% c('unbalanced','complete')) d <- d[!(d$id%%4==0 & d$time %in% c(3,7)),]
 if(sample=='complete') {keep <- names(which(table(d$id)==8));d <- d[d$id %in% as.integer(keep),]}
 if(sample=='future') d$first_treat[d$first_treat==0 & d$id%%2==0] <- 9
 sw <- if(weighted) d$w else rep(1,nrow(d))
 clean <- d$first_treat==0 | d$time<d$first_treat-A
 horizons <- seq(-A-L,-A-1)
 lead <- sapply(horizons,function(h) as.integer(d$first_treat>0 & d$time-d$first_treat==h))
 X <- cbind(model.matrix(~0+factor(id),d),model.matrix(~factor(time),d)[,-1,drop=FALSE],lead)
 X0 <- X[clean,,drop=FALSE];y <- d$y[clean];w <- sw[clean]
 Q <- crossprod(X0,w*X0);b <- solve(Q,crossprod(X0,w*y))
 residual <- as.vector(y-X0%*%b)
 score <- rowsum(X0*(w*residual),d$id[clean],reorder=TRUE)
 H <- solve(Q,diag(ncol(Q))[,seq(ncol(Q)-L+1,ncol(Q)),drop=FALSE])
 V <- crossprod(score%*%H)
 beta <- tail(as.vector(b),L)
 prefix <- paste(sample,as.integer(weighted),A,L,sep='_')
 write.csv(data.frame(event=horizons,estimate=beta,se=sqrt(diag(V))),file.path(out,paste0(prefix,'.csv')),row.names=FALSE)
 write.csv(V,file.path(out,paste0(prefix,'_vcov.csv')),row.names=FALSE)
 d$gref <- ifelse(d$first_treat==0,0,d$first_treat-A)
 for(j in seq_len(L)) d[[paste0('lead',j)]] <- lead[,j]
 formula <- as.formula(paste0('y ~ ',paste0('lead',seq_len(L),collapse='+'),' | id + time'))
 f <- fixest::feols(formula,data=d[clean,],weights=if(weighted) ~w else NULL,
                    vcov=~id,fixef.rm='none')
 stopifnot(max(abs(coef(f)-beta))<1e-8,max(abs(vcov(f)-V))<1e-8)
 pkg <- didimputation::did_imputation(d,yname='y',gname='gref',tname='time',idname='id',
       wname=if(weighted) 'w' else NULL,pretrends=seq(-L,-1))
 pkg <- as.data.frame(pkg);pkg <- pkg[as.character(pkg$term) %in% as.character(seq(-L,-1)),]
 pkg$event <- as.integer(as.character(pkg$term))-A;pkg <- pkg[order(pkg$event),]
 stopifnot(nrow(pkg)==L,max(abs(pkg$estimate-beta))<1e-8,max(abs(pkg$std.error-sqrt(diag(V))))<1e-8)
 write.csv(pkg,file.path(out,paste0(prefix,'_package.csv')),row.names=FALSE)
 W <- as.numeric(t(beta)%*%solve(V,beta))
 tests[[length(tests)+1]] <- data.frame(case=prefix,statistic=W,df=L,pvalue=pchisq(W,L,lower.tail=FALSE),n_units=length(unique(d$id)),clean_rows=sum(clean),reference_rows=sum(clean)-sum(lead))
}
write.csv(do.call(rbind,tests),file.path(out,'tests.csv'),row.names=FALSE)
writeLines(c(paste('R',getRversion()),paste('fixest',packageVersion('fixest')),paste('didimputation',packageVersion('didimputation')),
 'Independent full-unit-dummy WLS with clean observations and pooled lead indicators; full unit-clustered covariance.',
 'Actual fixest and didimputation pretrends estimates/SEs also checked; global fixest SSC K.adj=FALSE,G.adj=FALSE, fixef.tol=1e-10.',
 'Anticipation uses affected-onset cohort labels in the R package; lead labels restored to actual treatment.',
 'Future cohorts retain their actual timing in the clean-placebo regression.',
 'Authors Stata finite-sample dof/F convention is not asserted; family CRV0/CRV1 and normal/chi2 inference apply.'),file.path(out,'versions.txt'))
