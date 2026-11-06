# Reuse the reviewed native exporter without changing its historical source.
# args: output directory, pinned did2s.R, pinned etwfe R directory, repo root.
args<-commandArgs(TRUE)
source(file.path(args[4],'benchmarks/external/reference_population_review.R'))
for(panel in c('balanced','unbalanced'))for(A in 0:1)for(rich in c(FALSE,TRUE)) {
 label<-paste(panel,A,as.integer(rich),sep='_')
 d<-read.csv(file.path(out,paste0(label,'.input.csv')))
 d<-d[d$time<max(d$first_treat)-A,]
 # Keep actual future/reference adoption dates for the separate clean leads.
 d$gA<-d$first_treat-A
 d$lead1<-as.integer(d$time-d$first_treat==-A-2)
 d$lead2<-as.integer(d$time-d$first_treat==-A-1)
 clean<-d$time<d$gA
 f<-if(rich) y~x+I(x^2)+lead1+lead2|id+time+region^time else y~lead1+lead2|id+time
 fit<-feols(f,data=d[clean,],weights=if(rich)~w else NULL,vcov=~id,fixef.rm='none',ssc=ssc(K.adj=FALSE,G.adj=FALSE))
 ix<-match(c('lead1','lead2'),names(coef(fit)));stopifnot(!anyNA(ix))
 b<-coef(fit)[ix];names(b)<-paste0('event::',c(-A-2,-A-1))
 save_result(paste0(label,'_bjs_placebos'),b,vcov(fit)[ix,ix,drop=FALSE])
 first<-if(rich)~x+I(x^2)|id+time+region^time else ~0|id+time
 pkg<-as.data.frame(didimputation::did_imputation(d,yname='y',gname='gA',tname='time',idname='id',first_stage=first,wname=if(rich)'w' else NULL,horizon=TRUE,pretrends=c(-2,-1)))
 pkg<-pkg[as.character(pkg$term) %in% c('-2','-1'),]
 pkg$actual_event<-as.integer(as.character(pkg$term))-A
 pkg<-pkg[order(pkg$actual_event),]
 stopifnot(nrow(pkg)==2,max(abs(pkg$estimate-b))<3e-8,max(abs(pkg$std.error-sqrt(diag(vcov(fit)[ix,ix,drop=FALSE]))))<3e-8)
 write.csv(pkg,file.path(out,paste0(label,'_bjs_placebos_package.csv')),row.names=FALSE)
}
writeLines(c('Pinned unmodified authors did2s/etwfe/emfx; BJS points and clean leads from didimputation; full clean-lead covariance from fixest.',capture.output(sessionInfo())),file.path(out,'R_session.txt'))
