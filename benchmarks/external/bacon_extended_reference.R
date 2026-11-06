# Native Bacon outputs (where defined), plus independent original-row lm WLS.
args <- commandArgs(TRUE);original <- read.csv(args[1]);out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
fixest::setFixest_nthreads(2);fixest::setFixest_estimation(fixef.tol=1e-10)
cases <- c('plain','always','always_no_never','shifted_always','complete','future','multiple_always','castle')
checks <- list()
data('castle',package='bacondecomp')
castle$first_treat <- ave(ifelse(castle$post==1,castle$year,Inf),castle$state,FUN=min)
castle$first_treat[is.infinite(castle$first_treat)] <- 0
castle$id <- castle$state;castle$time<-castle$year;castle$y<-castle$l_homicide
castle$x<-castle$l_pop;castle$z<-castle$l_income
castle$unit_w<-1+as.integer(factor(castle$id))%%3;castle$w<-ave(castle$popwt,castle$id)
public<-castle[c('id','time','first_treat','y','x','z','unit_w','w','state','year','l_homicide','l_pop','l_income')]
write.csv(public,file.path(dirname(args[1]),'bacon_castle_covariates.csv'),row.names=FALSE)

for(case in cases) {
 d <- if(case=='castle') public else original
 if(case %in% c('always','always_no_never','shifted_always','multiple_always')) d$first_treat[d$first_treat==2] <- 1
 if(case=='always_no_never') d <- d[d$first_treat!=0,]
 if(case %in% c('shifted_always','multiple_always')) {d$time<-d$time+20;d$first_treat[d$first_treat>1]<-d$first_treat[d$first_treat>1]+20}
 if(case=='multiple_always') d$first_treat[d$id==0] <- 2
 if(case=='complete') {d<-d[!(d$id%%4==0 & d$time==3),];keep<-names(which(table(d$id)==length(unique(d$time))));d<-d[d$id %in% as.integer(keep),]}
 if(case=='future') d$first_treat[d$first_treat==0]<-100
 d$D<-as.integer(d$first_treat>0 & d$time>=d$first_treat)
 for(controlled in c(FALSE,TRUE)) for(kind in c('equal','integer','irrational')) {
  prefix<-paste(case,as.integer(controlled),kind,sep='_')
  w<-if(kind=='equal') rep(1,nrow(d)) else if(kind=='integer') d$unit_w else d$w
  f<-if(controlled) y~D+x+z+factor(id)+factor(time) else y~D+factor(id)+factor(time)
  d$reference_weight <- w
  fit<-lm(f,data=d,weights=reference_weight)
  package_status<-'not_supported';error<-'';omega<-beta_w<-NA_real_
  if(kind!='irrational') {
   dd<-d
   if(kind=='integer') {
    blocks<-list();next_id<-0L
    for(id in unique(d$id)) {
     part<-d[d$id==id,]
     for(j in seq_len(part$unit_w[1])) {part$id<-next_id;next_id<-next_id+1L;blocks[[length(blocks)+1]]<-part}
    }
    dd<-do.call(rbind,blocks)
   }
   b<-tryCatch(bacondecomp::bacon(if(controlled) y~D+x+z else y~D,data=dd,id_var='id',time_var='time',quietly=TRUE),error=function(e) {error<<-conditionMessage(e);NULL})
   if(is.null(b)) package_status<-'error' else {
    table<-if(controlled) b$two_by_twos else b
    if(controlled) {omega<-b$Omega;beta_w<-b$beta_hat_w}
    package_status<-if(all(is.finite(table$estimate)) && all(is.finite(table$weight)) && (!controlled || (is.finite(omega) && is.finite(beta_w)))) 'ok' else 'nonfinite'
    table$treated[is.infinite(table$treated)|table$treated==99999]<-0
    table$untreated[is.infinite(table$untreated)|table$untreated==99999]<-0
    write.csv(table,file.path(out,paste0(prefix,'.csv')),row.names=FALSE)
   }
  }
  checks[[length(checks)+1]]<-data.frame(name=prefix,twfe=unname(coef(fit)['D']),Omega=omega,beta_w=beta_w,package_status=package_status,error=error,n_obs=nrow(d),n_units=length(unique(d$id)))
 }
}
write.csv(do.call(rbind,checks),file.path(out,'models.csv'),row.names=FALSE)
writeLines(c('Native bacondecomp, no code replacement. Integer weights use exact unit replication.',capture.output(sessionInfo())),file.path(out,'R_session.txt'))
