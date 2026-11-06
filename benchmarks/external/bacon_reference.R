# Actual bacondecomp comparison weights/estimates and independent dense OLS TWFE.
args <- commandArgs(TRUE)
d0 <- read.csv(args[1]);out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
cases <- c('heterogeneous','no_never','single_timing','future','shifted','complete','constant','null','two_period')
checks <- list()
for(case in cases) {
 d <- d0
 if(case=='no_never') d <- d[d$first_treat!=0,]
 if(case=='single_timing') d <- d[d$first_treat %in% c(0,5),]
 if(case=='future') d$first_treat[d$first_treat==0] <- 20
 if(case=='shifted') { d$time <- d$time+20; d$first_treat[d$first_treat>0] <- d$first_treat[d$first_treat>0]+20 }
 if(case=='complete') {
  d <- d[!(d$id%%4==0 & d$time %in% c(3,7)),]
  complete <- names(which(table(d$id)==9));d <- d[d$id %in% as.integer(complete),]
 }
 if(case=='constant') d$y <- d$constant
 if(case=='null') d$y <- d$baseline
 if(case=='two_period') {
  d <- d[d$first_treat!=2 & d$time %in% c(4,5),]
  d$time <- d$time-3;d$first_treat[d$first_treat>0] <- d$first_treat[d$first_treat>0]-3
 }
 d$D <- as.integer(d$first_treat>0 & d$time>=d$first_treat)
 b <- bacondecomp::bacon(y~D,data=d,id_var='id',time_var='time',quietly=TRUE)
 b$treated[is.infinite(b$treated) | b$treated==99999] <- 0;b$untreated[is.infinite(b$untreated) | b$untreated==99999] <- 0
 write.csv(b,file.path(out,paste0(case,'.csv')),row.names=FALSE)
 fit <- lm(y~D+factor(id)+factor(time),data=d)
 checks[[length(checks)+1]] <- data.frame(case=case,twfe=unname(coef(fit)['D']),decomposed=sum(b$estimate*b$weight),weight_sum=sum(b$weight),n_units=length(unique(d$id)),n_obs=nrow(d))
}
write.csv(do.call(rbind,checks),file.path(out,'twfe.csv'),row.names=FALSE)
writeLines(c(paste('R',getRversion()),paste('bacondecomp',packageVersion('bacondecomp')),paste('fixest',packageVersion('fixest')),
 'Actual installed bacondecomp::bacon; no source replacement.',
 'TWFE is independently fitted with lm and full unit/time dummies.',
 'Never-treated Inf/99999 sentinel restored to 0; future cohorts inferred from realized treatment.'),file.path(out,'versions.txt'))
