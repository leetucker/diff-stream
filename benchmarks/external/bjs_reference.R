# Independent dense design oracle for the published BJS score formulas.
# Package SE comparison uses didimputation separately; this script additionally
# exports joint covariance and explicit leave-out/coarser-group cases.
args <- commandArgs(TRUE)
d <- read.csv(args[1]); out <- args[2]
A <- if(length(args)>2) as.integer(args[3]) else 0L
weightname <- if(length(args)>3) args[4] else NULL
sw <- if(is.null(weightname)) rep(1,nrow(d)) else d[[weightname]]
if(A>0 || !is.null(weightname)) fixest::setFixest_estimation(fixef.tol=1e-10)
dir.create(out, showWarnings=FALSE, recursive=TRUE)
x <- cbind(model.matrix(~ 0 + factor(id),d), model.matrix(~ factor(time),d)[,-1,drop=FALSE])
treated <- d$first_treat > 0 & d$time >= d$first_treat-A
x0 <- x[!treated,,drop=FALSE]
Q <- crossprod(x0,sw[!treated]*x0)
b <- solve(Q, crossprod(x0,sw[!treated]*d$y[!treated]))
r <- as.vector(d$y - x %*% b)
cells <- unique(d[treated,c('first_treat','time')])
cells <- cells[order(cells$first_treat,cells$time),]
M <- nrow(cells)
wcell <- matrix(0,nrow(d),M)
for(j in seq_len(M)) {
  ix <- d$first_treat==cells$first_treat[j] & d$time==cells$time[j]
  wcell[ix,j] <- sw[ix]/sum(sw[ix])
}
counts <- as.vector(crossprod(wcell>0,sw))
event <- cells$time-cells$first_treat
maps <- list(cells=diag(M),dynamic=sapply(sort(unique(event)),function(e) {
  z <- counts*(event==e); z/sum(z)
}),simple=matrix(counts*(event>=0)/sum(counts*(event>=0)),ncol=1))
cnames <- list(cells=paste0('g::',cells$first_treat,':t::',cells$time),
               dynamic=paste0('event::',sort(unique(event))),simple='ATT')
for(avg in c('cohort_time','event_time','cohort','all')) {
  label <- switch(avg,cohort_time=paste(d$first_treat,d$time),event_time=d$time-d$first_treat,
                  cohort=d$first_treat,all=rep(1,nrow(d)))
  for(lo in c(FALSE,TRUE)) for(kind in names(maps)) {
    w1 <- wcell %*% maps[[kind]]
    v <- w1
    v[!treated,] <- -sw[!treated] * (x0 %*% solve(Q, crossprod(x[treated,,drop=FALSE],w1[treated,,drop=FALSE])))
    scores <- matrix(0,length(unique(d$id)),ncol(w1))
    for(j in seq_len(ncol(w1))) {
      wt <- w1[treated,j]; lab <- label[treated]; id <- d$id[treated]; rr <- r[treated]
      clusterweight <- ave(wt,interaction(id,lab,drop=TRUE),FUN=sum)
      denom <- ave(wt*clusterweight,lab,FUN=sum)
      numerator <- ave(wt*clusterweight*rr,lab,FUN=sum)
      center <- ifelse(denom>0,numerator/denom,0)
      eps <- r
      eps[treated] <- rr-center
      if(lo) {
        leverage <- ifelse(denom>0,clusterweight^2/denom,0)
        stopifnot(all(leverage[wt!=0] < 1-1e-12))
        eps[treated] <- eps[treated]/(1-leverage)
      }
      scores[,j] <- as.vector(rowsum(v[,j]*eps,d$id,reorder=TRUE))
    }
    V <- crossprod(scores)
    prefix <- paste('bjs',avg,if(lo) 'leaveout' else 'plugin',kind,sep='_')
    write.csv(data.frame(term=cnames[[kind]],estimate=as.vector(crossprod(w1,r)),se=sqrt(diag(V))),
              file.path(out,paste0(prefix,'.csv')),row.names=FALSE)
    write.csv(V,file.path(out,paste0(prefix,'_vcov.csv')),row.names=FALSE)
  }
}
if(A>0) {
  dp <- d
  dp$actual_post <- as.numeric(dp$first_treat>0 & dp$time>=dp$first_treat)
  dp$first_treat <- ifelse(dp$first_treat==0,0,dp$first_treat-A)
  package <- didimputation::did_imputation(data=dp,yname='y',gname='first_treat',tname='time',idname='id',wtr='actual_post',wname=weightname)
  dynamic <- didimputation::did_imputation(data=dp,yname='y',gname='first_treat',tname='time',idname='id',horizon=TRUE,wname=weightname)
  dynamic$term <- as.character(as.integer(dynamic$term)-A)
  write.csv(dynamic,file.path(out,'bjs_package_dynamic.csv'),row.names=FALSE)
} else {
  package <- didimputation::did_imputation(data=d,yname='y',gname='first_treat',tname='time',idname='id',wname=weightname)
}
write.csv(package,file.path(out,'bjs_package_simple.csv'),row.names=FALSE)
writeLines(c(paste('R',getRversion()),paste('didimputation',packageVersion('didimputation')),paste('anticipation',A),paste('sampling weights',if(is.null(weightname)) 'none' else weightname),if(A>0 || !is.null(weightname)) 'didimputation first-stage fixef.tol=1e-10' else NULL,
             if(!is.null(weightname)) 'Package weighted point estimates only: its weighted variance uses squared-weight first-stage Gram; dense WLS covariance is the inference reference.' else NULL,
             'Dense full-unit-dummy oracle; independently implemented published score equations.',
             'Stata execution is external and is not claimed by this fixture.'),file.path(out,'bjs_versions.txt'))
