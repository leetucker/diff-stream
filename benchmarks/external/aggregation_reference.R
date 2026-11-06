# Actual did::aggte vectors and full joint covariance, including share scores.
# Usage: Rscript aggregation_reference.R tests/data tests/data/aggregation_reference
args <- commandArgs(TRUE)
root <- args[1]; out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
inputs <- list(balanced='reference_panel.csv',unbalanced='unbalanced_panel.csv')
for(sample in names(inputs)) {
 d <- read.csv(file.path(root,inputs[[sample]]))
 for(control in c('nevertreated','notyettreated')) for(base in c('varying','universal')) {
  a <- did::att_gt(yname='y',tname='time',idname='id',gname='first_treat',data=d,
        xformla=if(sample=='unbalanced') ~x+z else ~1, est_method='dr',
        allow_unbalanced_panel=(sample=='unbalanced'),control_group=control,
        base_period=base,bstrap=FALSE,cband=FALSE)
  cases <- list(cohort=list(type='group'),calendar=list(type='calendar'),
                dynamic_balance0=list(type='dynamic',balance_e=0),
                dynamic_balance1=list(type='dynamic',balance_e=1),
                dynamic_balance2=list(type='dynamic',balance_e=2),
                cohort_horizons=list(type='group',min_e=0,max_e=1))
  for(name in names(cases)) {
   z <- do.call(did::aggte,c(list(MP=a,bstrap=FALSE,cband=FALSE),cases[[name]]))
   prefix <- paste(sample,control,base,name,sep='_')
   write.csv(data.frame(level=z$egt,att=z$att.egt,se=z$se.egt),file.path(out,paste0(prefix,'.csv')),row.names=FALSE)
   key <- switch(z$type,group='selective.inf.func.g',calendar='calendar.inf.func.t',dynamic='dynamic.inf.func.e')
   S <- as.matrix(z$inf.function[[key]]) / a$n
   V <- crossprod(S)
   stopifnot(max(abs(diag(V)-z$se.egt^2),na.rm=TRUE)<1e-10)
   write.csv(V,file.path(out,paste0(prefix,'_vcov.csv')),row.names=FALSE)
  }
 }
}
writeLines(c(paste('R',getRversion()),paste('did',packageVersion('did')),paste('DRDID',packageVersion('DRDID')),
             'Actual aggte vectors, including estimated-share scores; analytic unit-clustered covariance.'),file.path(out,'versions.txt'))
