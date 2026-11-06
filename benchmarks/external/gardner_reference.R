# Execute an unmodified, pinned authors' did2s.R; no production code is sourced.
# args: repository root, output directory, cached did2s.R
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; out <- args[2]; author_source <- args[3]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
source(author_source)
fixest::setFixest_nthreads(2)
fixest::setFixest_estimation(fixef.tol=1e-10, fixef.iter=100000)
run_one <- function(d, name, spec, A, design, weight, reference=-A-1) {
  d$treatment <- as.integer(d$first_treat != 0 & d$time >= d$first_treat-A)
  d$event <- ifelse(d$first_treat == 0, Inf, d$time-d$first_treat)
  first <- if(design == 'plain') ~0 | id+time else if(design == 'numeric') ~x+I(x^2) | id+time else ~x+I(x^2) | id+time+region^time
  second <- if(spec == 'static') ~i(treatment,ref=0) else as.formula(sprintf('~i(event,ref=c(%d,Inf))',reference))
  r <- did2s(d,yname='y',first_stage=first,second_stage=second,treatment='treatment',cluster_var='id',weights=weight,verbose=FALSE)
  b <- coef(r); V <- vcov(r)
  terms <- if(spec=='static') 'ATT' else sub('event::','event::',names(b),fixed=TRUE)
  write.csv(data.frame(term=terms,estimate=as.numeric(b)),file.path(out,paste0(name,'.csv')),row.names=FALSE)
  write.table(as.matrix(V),file.path(out,paste0(name,'_vcov.csv')),sep=',',row.names=FALSE,col.names=FALSE)
}
for(panel in c('balanced','unbalanced')) {
 d <- read.csv(file.path(root,'tests/data',if(panel=='balanced') 'richer_panel.csv' else 'richer_unbalanced_panel.csv'))
 for(A in 0:1) for(spec in c('static','dynamic')) for(case in c('plain_varying','numeric_varying','extra_fe_varying','extra_fe_unit','extra_fe_equal')) {
  design <- if(startsWith(case,'plain')) 'plain' else if(startsWith(case,'numeric')) 'numeric' else 'extra'
  weight <- if(case=='extra_fe_equal') NULL else if(case=='extra_fe_unit') 'unit_w' else 'w'
  run_one(d,paste(panel,A,case,spec,sep='_'),spec,A,design,weight)
 }
 for(A in 0:1) run_one(d,paste(panel,A,'custom_reference',sep='_'),'dynamic',A,'extra','w',-A-2)
}
castle <- read.csv(file.path(root,'tests/data/castle.csv'))
castle$id <- castle$state; castle$time <- castle$year; castle$y <- castle$l_homicide
castle$first_treat <- ifelse(is.na(castle$effyear),0,castle$effyear+1)
for(spec in c('static','dynamic')) for(weighted in c(FALSE,TRUE)) {
 run_one(castle,paste('castle',weighted,spec,sep='_'),spec,0,'plain',if(weighted) 'popwt' else NULL)
}
writeLines(capture.output(sessionInfo()),file.path(out,'R_session.txt'))
