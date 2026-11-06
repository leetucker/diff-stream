# Actual DRDID 2x2 references for explicit control-only trimming.
args<-commandArgs(TRUE)
d<-read.csv(args[1]);out<-args[2]
a<-d[d$time==1,];b<-d[d$time==2,]
X<-cbind(1,a$x);D<-as.numeric(a$first_treat>0)
for(method in c('reg','ipw','dr')) for(level in c(.8,.995,1)) {
 r<-switch(method,
   reg=DRDID::reg_did_panel(b$y,a$y,D,X,inffunc=TRUE),
   ipw=DRDID::std_ipw_did_panel(b$y,a$y,D,X,trim.level=level,inffunc=TRUE),
   dr=DRDID::drdid_panel(b$y,a$y,D,X,trim.level=level,inffunc=TRUE))
 write.csv(data.frame(att=r$ATT,se=r$se),file.path(out,paste0('trim_',method,'_',level,'.csv')),row.names=FALSE)
}
