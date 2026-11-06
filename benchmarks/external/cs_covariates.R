# Actual R did/DRDID comparisons, with time-varying covariates.
args <- commandArgs(TRUE)
d <- read.csv(args[1]); out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
forms <- list(numeric=~x+z,expanded=~x*factor(category)+I(x^2)+z)
for(f in names(forms)) for(control in c('nevertreated','notyettreated')) {
 for(base in c('varying','universal')) for(method in c('reg','ipw','dr')) {
  a <- did::att_gt(yname='y',tname='time',idname='id',gname='first_treat',xformla=forms[[f]],
                   data=d,control_group=control,base_period=base,est_method=method,
                   bstrap=FALSE,cband=FALSE)
  prefix <- paste('cs',f,control,base,method,sep='_')
  write.csv(data.frame(g=a$group,t=a$t,att=a$att,se=a$se),file.path(out,paste0(prefix,'.csv')),row.names=FALSE)
  V <- as.matrix(a$V_analytical)/a$n
  stopifnot(max(abs(diag(V)-a$se^2),na.rm=TRUE)<1e-10)
  write.csv(V,file.path(out,paste0(prefix,'_vcov.csv')),row.names=FALSE)
  for(kind in c('dynamic','simple')) {
   ag <- did::aggte(a,type=kind,bstrap=FALSE,cband=FALSE)
   z <- if(kind=='simple') data.frame(event=NA,att=ag$overall.att,se=ag$overall.se) else
      data.frame(event=ag$egt,att=ag$att.egt,se=ag$se.egt)
   write.csv(z,file.path(out,paste0(prefix,'_',kind,'.csv')),row.names=FALSE)
  }
 }
}
writeLines(c(paste('R',getRversion()),paste('did',packageVersion('did')),paste('DRDID',packageVersion('DRDID')),
             'Actual did::att_gt, est_method reg/ipw/dr; analytic covariance; earlier-period covariates.'),file.path(out,'versions.txt'))
