# Last-treated checks: explicit clean date window and authors' did/fixest fits.
args <- commandArgs(TRUE);root <- args[1];out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
fixest::setFixest_nthreads(2)
# Precision confirmation changes only the external propensity solver's stopping
# tolerance/iteration limit in this R process; installed package files are untouched.
precision <- length(args)>=3 && args[3]=='precision'
if(precision) {
 ns <- asNamespace('DRDID'); f <- get('fastglm_fit',envir=ns)
 txt <- paste(deparse(body(f)),collapse='\n')
 stopifnot(grepl('tol = 1e-08',txt,fixed=TRUE),grepl('maxit = 100L',txt,fixed=TRUE))
 txt <- gsub('tol = 1e-08','tol = 1e-12',txt,fixed=TRUE)
 txt <- gsub('maxit = 100L','maxit = 1000L',txt,fixed=TRUE)
 body(f) <- parse(text=txt)[[1]]
 unlockBinding('fastglm_fit',ns);assign('fastglm_fit',f,envir=ns);lockBinding('fastglm_fit',ns)
}

write_result <- function(prefix,term,b,V) {
 write.csv(data.frame(term=term,estimate=as.vector(b)),file.path(out,paste0(prefix,'.csv')),row.names=FALSE)
 write.csv(V,file.path(out,paste0(prefix,'_vcov.csv')),row.names=FALSE)
}
for(panel in c('balanced','unbalanced')) {
 original <- read.csv(file.path(root,'tests/data',if(panel=='balanced') 'richer_panel.csv' else 'richer_unbalanced_panel.csv'))
 original <- original[original$first_treat!=0,]
 last <- max(original$first_treat)
 for(A in 0:1) {
  d <- original[original$time<last-A,]
  # Shift all affected onsets during fitting: a cohort can have anticipatory
  # effects inside the retained window even if its actual adoption date is outside.
  # Export actual labels; only the designated reference is recoded.
  d$reference_g <- ifelse(d$first_treat==last,0,d$first_treat)
  for(design in c('constant','numeric')) for(control in c('nevertreated','notyettreated')) for(base in c('varying','universal')) for(method in c('reg','ipw','dr')) {
   dd <- d;shift <- A
   dd$gref <- ifelse(dd$reference_g==0,0,dd$reference_g-shift)
   fit <- did::att_gt(yname='y',tname='time',idname='id',gname='gref',data=dd,
    xformla=if(design=='constant') ~1 else ~x+z,weightsname='w',anticipation=A-shift,
    control_group=control,base_period=base,est_method=method,allow_unbalanced_panel=panel=='unbalanced',
    faster_mode=FALSE,bstrap=FALSE,cband=FALSE)
   prefix <- paste('cs',panel,A,design,control,base,method,sep='_')
   write_result(prefix,paste0('g::',fit$group+shift,':t::',fit$t),fit$att,as.matrix(fit$V_analytical)/fit$n)
   for(kind in c('dynamic','simple','group','calendar')) {
    if(A>0 && kind!='dynamic') next
    z <- did::aggte(fit,type=kind,bstrap=FALSE,cband=FALSE)
    key <- switch(kind,dynamic='dynamic.inf.func.e',group='selective.inf.func.g',calendar='calendar.inf.func.t',simple='simple.att')
    S <- as.matrix(z$inf.function[[key]])/fit$n
    labels <- if(kind=='simple') 'ATT' else paste0(switch(kind,dynamic='event::',group='cohort::',calendar='calendar::'),z$egt-if(kind=='dynamic') shift else 0)
    write_result(paste0(prefix,'_',kind),labels,if(kind=='simple') z$overall.att else z$att.egt,crossprod(S))
   }
  }
  for(design in c('constant','numeric','extra_fe')) {
   dd <- d;dd$gref <- ifelse(dd$reference_g==0,max(dd$time)+100,dd$reference_g-A)
   rhs <- if(design=='constant') '1' else 'x+z'
   f <- as.formula(paste0('y~sunab(gref,time,ref.p=-1,no_agg=TRUE)+',rhs,'|id+time',if(design=='extra_fe') '+region^time' else ''))
   fit <- fixest::feols(f,dd,weights=~w,vcov=~id,fixef.rm='none',fixef.tol=1e-10,ssc=fixest::ssc(K.adj=FALSE,G.adj=FALSE))
   ix <- grep(':cohort::',names(coef(fit)),fixed=TRUE)
   parts <- strsplit(names(coef(fit))[ix],':cohort::',fixed=TRUE)
   g <- as.integer(vapply(parts,`[`,'',2)); h <- as.integer(sub('time::','',vapply(parts,`[`,'',1),fixed=TRUE))
   write_result(paste('sunab',panel,A,design,sep='_'),paste0('g::',g+A,':t::',g+h),coef(fit)[ix],vcov(fit)[ix,ix,drop=FALSE])
  }
 }
}
writeLines(c(if(precision) 'Precision confirmation: DRDID fastglm tolerance 1e-12/maxit 1000, in-process only.' else 'Native DRDID defaults.',capture.output(sessionInfo())),file.path(out,'R_session.txt'))
