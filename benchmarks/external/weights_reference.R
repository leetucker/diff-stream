# Sampling weights: R did/fixest references; time-invariant positive weights.
args <- commandArgs(TRUE)
d0 <- read.csv(args[1]);out <- args[2]
d0$w <- 0.2 + d0$id%%9
dir.create(out,recursive=TRUE,showWarnings=FALSE)
for(sample in c('balanced','unbalanced')) {
 d <- d0
 if(sample=='unbalanced') d <- d[!(d$id%%4==0 & d$time %in% c(3,7)),]
 for(A in c(0,2)) for(design in c('constant','numeric')) for(control in c('nevertreated','notyettreated')) {
  methods <- if(design=='constant') 'dr' else c('reg','ipw','dr')
  for(base in c('varying','universal')) for(method in methods) {
   dd <- d
   # did 2.5.1's varying base switches at g, not g-A, for A>=2. Shift the
   # first-treatment labels to the affected onset and restore actual labels in
   # the exported cells. The no-anticipation estimator then supplies the correct
   # clean-base row influence for every affected-period comparison.
   shift <- if(base=='varying') A else 0
   dd$reference_g <- ifelse(dd$first_treat==0,0,dd$first_treat-shift)
   a <- did::att_gt(yname='y',tname='time',idname='id',gname='reference_g',data=dd,
       xformla=if(design=='numeric') ~x+z else ~1,est_method=method,
       anticipation=A-shift,control_group=control,base_period=base,
       allow_unbalanced_panel=(sample=='unbalanced'),weightsname='w',bstrap=FALSE,cband=FALSE)
   prefix <- paste(sample,A,design,control,base,method,sep='_')
   write.csv(data.frame(g=a$group+shift,t=a$t,att=a$att,se=a$se),file.path(out,paste0(prefix,'.csv')),row.names=FALSE)
   write.csv(as.matrix(a$V_analytical)/a$n,file.path(out,paste0(prefix,'_vcov.csv')),row.names=FALSE)
   dynamic <- did::aggte(a,type='dynamic',bstrap=FALSE,cband=FALSE)
   write.csv(data.frame(event=dynamic$egt-shift,att=dynamic$att.egt,se=dynamic$se.egt),file.path(out,paste0(prefix,'_dynamic.csv')),row.names=FALSE)
   # Post-actual-treatment aggregation references use actual labels and A.
   # Their cell scores agree under either pre-period convention.
   if(base=='universal') {
    for(kind in c('simple','group','calendar')) {
     z <- did::aggte(a,type=kind,bstrap=FALSE,cband=FALSE)
     table <- if(kind=='simple') data.frame(level=NA,att=z$overall.att,se=z$overall.se) else
         data.frame(level=z$egt,att=z$att.egt,se=z$se.egt)
     write.csv(table,file.path(out,paste0(prefix,'_',kind,'.csv')),row.names=FALSE)
    }
   }
  }
 }
 for(A in c(0,2)) {
  dd <- d; dd$gref <- ifelse(dd$first_treat==0,max(dd$time)+100,dd$first_treat)
  sa <- fixest::feols(y ~ sunab(gref,time,ref.p=-A-1,no_agg=TRUE) | id+time, data=dd,
          weights=~w,vcov=~id,fixef.rm='none',fixef.tol=1e-10,ssc=fixest::ssc(K.adj=FALSE,G.adj=FALSE))
  tab <- fixest::coeftable(sa); prefix <- paste(sample,A,'sunab',sep='_')
  write.csv(data.frame(term=rownames(tab),att=tab[,1],se=tab[,2]),file.path(out,paste0(prefix,'.csv')),row.names=FALSE)
  write.csv(stats::vcov(sa),file.path(out,paste0(prefix,'_vcov.csv')),row.names=FALSE)
 }
}
writeLines(c(paste('R',getRversion()),paste('did',packageVersion('did')),paste('fixest',packageVersion('fixest')),
 'Universal-base CS: actual anticipation option. Varying-base CS: affected-onset labels, anticipation=0; restore original cohort/event labels.',
 'This avoids the installed did varying-base switch-at-g behavior inside anticipation windows of length >=2.',
 'Post-actual-treatment aggregations use universal-base actual-anticipation did results.',
 'Sampling weight w=0.2+id modulo 9; positive and constant within unit.'),file.path(out,'versions.txt'))
