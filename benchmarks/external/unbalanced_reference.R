# Usage: Rscript unbalanced_reference.R input.csv output-directory
# Observed-row CS uses actual did; SA uses fixest's saturated sunab model.
args <- commandArgs(TRUE)
d <- read.csv(args[1]); out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
for (f in c('constant','numeric')) for (control in c('nevertreated','notyettreated')) {
 for (base in c('varying','universal')) for (method in c('reg','ipw','dr')) {
  a <- did::att_gt(yname='y', tname='time', idname='id', gname='first_treat',
       data=d, xformla=if(f=='constant') ~1 else ~x+z,
       allow_unbalanced_panel=TRUE, control_group=control, base_period=base,
       est_method=method, bstrap=FALSE, cband=FALSE)
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
d$gref <- ifelse(d$first_treat==0,max(d$time)+1,d$first_treat)
sa <- fixest::feols(y ~ sunab(gref,time,no_agg=TRUE) | id + time, data=d,vcov=~id,
                   fixef.rm='none',ssc=fixest::ssc(K.adj=FALSE,G.adj=FALSE))
tab <- fixest::coeftable(sa)
write.csv(data.frame(term=rownames(tab),att=tab[,1],se=tab[,2]),file.path(out,'sunab_cells.csv'),row.names=FALSE)
write.csv(stats::vcov(sa),file.path(out,'sunab_cells_vcov.csv'),row.names=FALSE)
sa <- fixest::feols(y ~ sunab(gref,time) | id + time, data=d,vcov=~id,
                   fixef.rm='none',ssc=fixest::ssc(K.adj=FALSE,G.adj=FALSE))
tab <- fixest::coeftable(sa)
write.csv(data.frame(term=rownames(tab),att=tab[,1],se=tab[,2]),file.path(out,'sunab_dynamic.csv'),row.names=FALSE)
writeLines(c(paste('R',getRversion()),paste('did',packageVersion('did')),paste('DRDID',packageVersion('DRDID')),
             paste('fixest',packageVersion('fixest')),
             'Actual did allow_unbalanced_panel=TRUE; observed-date covariates; unit clustering.'),file.path(out,'versions.txt'))
# Direct 2x2 RC checks with explicit trimming; fold row IF by original unit.
s <- d[d$first_treat %in% c(0,3) & d$time %in% c(2,4),]
X <- model.matrix(~x+z,s); D <- as.numeric(s$first_treat==3); P <- as.numeric(s$time==4)
for(method in c('reg','ipw','dr')) for(level in c(.4,.995,1)) {
 a <- switch(method,
   reg=DRDID::reg_did_rc(s$y,P,D,X,inffunc=TRUE),
   ipw=DRDID::std_ipw_did_rc(s$y,P,D,X,trim.level=level,inffunc=TRUE),
   dr=DRDID::drdid_rc(s$y,P,D,X,trim.level=level,inffunc=TRUE))
 score <- rowsum(a$att.inf.func,s$id)/nrow(s)
 write.csv(data.frame(att=a$ATT,se=sqrt(sum(score^2))),file.path(out,paste0('trim_',method,'_',level,'.csv')),row.names=FALSE)
}
