# Native no-never-treated probes; explicitly restricted fits are the valid targets.
args<-commandArgs(TRUE);out<-args[1]
source(args[2]);source(file.path(args[3],'etwfe.R'));source(file.path(args[3],'emfx.R'))
library(fixest);library(marginaleffects);library(data.table);library(Formula)
setFixest_nthreads(2);setDTthreads(2)
setFixest_estimation(fixef.tol=1e-10,fixef.iter=100000)
setFixest_ssc(ssc(K.adj=FALSE,G.adj=FALSE),vcov_names=c('iid','hetero','cluster'))
probes<-list()
save_result<-function(name,b,V=NULL) {
 write.csv(data.frame(term=names(b),estimate=as.numeric(b)),file.path(out,paste0(name,'.csv')),row.names=FALSE)
 if(!is.null(V))write.table(V,file.path(out,paste0(name,'_vcov.csv')),sep=',',row.names=FALSE,col.names=FALSE)
}
probe<-function(name,code) {
 warnings<-character();error<-''
 value<-tryCatch(withCallingHandlers(code(),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e){error<<-conditionMessage(e);NULL})
 probes[[length(probes)+1]]<<-data.frame(name=name,status=if(is.null(value))'error' else 'returned',error=error,warnings=paste(warnings,collapse=' | '))
 if(!is.null(value))write.csv(as.data.frame(value),file.path(out,paste0(name,'_native.csv')),row.names=FALSE)
}
for(panel in c('balanced','unbalanced'))for(A in 0:1)for(rich in c(FALSE,TRUE)) {
 label<-paste(panel,A,as.integer(rich),sep='_')
 original<-read.csv(file.path(out,paste0(label,'.input.csv')))
 last<-max(original$first_treat);d<-original[original$time<last-A,]
 d$first_treat[d$first_treat==last]<-0
 weight<-if(rich)'w' else NULL
 first<-if(rich)~x+I(x^2)|id+time+region^time else ~0|id+time
 prepare<-function(x){x$gA<-ifelse(x$first_treat==0,0,x$first_treat-A);x$treatment<-as.integer(x$gA>0 & x$time>=x$gA);x$event<-ifelse(x$first_treat==0,Inf,x$time-x$first_treat);x}
 d<-prepare(d);full<-prepare(original)
 bjs<-function(x) didimputation::did_imputation(x,yname='y',gname='gA',tname='time',idname='id',first_stage=first,wname=weight,horizon=TRUE)
 b<-as.data.frame(bjs(d));names_b<-paste0('event::',as.integer(as.character(b$term))-A)
 save_result(paste0(label,'_bjs'),setNames(b$estimate,names_b))
 probe(paste0(label,'_bjs_full'),function()bjs(full))
 for(spec in c('static','dynamic')) {
  second<-if(spec=='static')~i(treatment,ref=0) else as.formula(sprintf('~i(event,ref=c(%d,Inf))',-A-1))
  gardner<-function(x)did2s(x,yname='y',first_stage=first,second_stage=second,treatment='treatment',cluster_var='id',weights=weight,verbose=FALSE)
  g<-gardner(d);bb<-coef(g);names(bb)<-if(spec=='static')'ATT' else names(bb)
  save_result(paste(label,'gardner',spec,sep='_'),bb,vcov(g))
  probe(paste(label,'gardner',spec,'full',sep='_'),function(){m<-gardner(full);data.frame(term=names(coef(m)),estimate=coef(m))})
 }
 make_etwfe<-function(x,control,reference) {
  x$first_treat<-ifelse(x$first_treat==0,0,x$first_treat-A)
  etwfe(if(rich)y~x+z else y~1,tvar='time',gvar='first_treat',data=x,cgroup=control,gref=reference,fe='none',vcov=~id,weights=if(rich)~w else NULL,ssc=ssc(K.adj=FALSE,G.adj=FALSE),fixef.rm='none')
 }
 e<-make_etwfe(d,'notyet',0)
 dat<-as.data.table(eval(e$call$data,e$call_env));dat<-dat[dat$first_treat!=0 & dat$.Dtreat,];dat$N<-if(rich)dat$w else 1
 model<-e;class(model)<-setdiff(class(model),'etwfe')
 for(inference in c('conditional','unconditional')) {
  v<-if(inference=='conditional')TRUE else vcovUnconditional(type='HC0',cluster=~id)
  m<-slopes(model,newdata=dat,variables='.Dtreat',by=c('first_treat','time'),wts='N',vcov=v)
  dd<-as.data.frame(m);names_b<-paste0('g::',dd$first_treat+A,':t::',dd$time)
  save_result(paste(label,'etwfe',inference,sep='_'),setNames(dd$estimate,names_b),vcov(m))
 }
 for(control in c('notyet','never'))probe(paste(label,'etwfe',control,'full',sep='_'),function(){m<-make_etwfe(original,control,NULL);as.data.frame(emfx(m,type='event',vcov=TRUE))})
}
write.csv(do.call(rbind,probes),file.path(out,'native_probes.csv'),row.names=FALSE)
writeLines(c('Unmodified pinned did2s/etwfe/emfx; installed didimputation points. Restricted samples are supplied explicitly.',capture.output(sessionInfo())),file.path(out,'R_session.txt'))
