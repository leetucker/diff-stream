# Exact public worked example from the bacondecomp README, with dense OLS checks.
args <- commandArgs(TRUE)
data_path <- args[1];out <- args[2]
dir.create(dirname(data_path),recursive=TRUE,showWarnings=FALSE)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
data(math_reform,package='bacondecomp')
d <- math_reform[order(math_reform$state,math_reform$class),]
stopifnot(nrow(d)==520,length(unique(d$state))==26,all(table(d$state)==20),
 all(sort(unique(d$class))==1975:1994),!anyNA(d[c('state','class','incearn_ln','reform_math')]),
 !anyDuplicated(d[c('state','class')]))
g <- ifelse(is.na(d$reformyr_math),0,d$reformyr_math)
stopifnot(all(d$reform_math==as.integer(g>0 & d$class>=g)))
write.csv(d,data_path,row.names=FALSE)
b <- bacondecomp::bacon(incearn_ln~reform_math,data=d,id_var='state',time_var='class',quietly=TRUE)
printed <- bacondecomp::bacon_summary(b,return_df=TRUE)
write.csv(printed,file.path(out,'documented_summary.csv'),row.names=FALSE)
b$treated[is.infinite(b$treated) | b$treated==99999] <- 0
b$untreated[is.infinite(b$untreated) | b$untreated==99999] <- 0
b$dense_estimate <- NA_real_
for(j in seq_len(nrow(b))) {
 target <- b$treated[j];control <- b$untreated[j]
 use <- g %in% c(target,control)
 if(control>target) use <- use & d$class<control
 if(control>0 && control<target) use <- use & d$class>=control
 pair <- d[use,]
 pairfit <- lm(incearn_ln~reform_math+factor(state)+factor(class),data=pair)
 b$dense_estimate[j] <- unname(coef(pairfit)['reform_math'])
}
stopifnot(nrow(b)==16,max(abs(b$estimate-b$dense_estimate))<1e-10)
write.csv(b,file.path(out,'comparisons.csv'),row.names=FALSE)
summary <- do.call(rbind,lapply(split(b,b$type),function(z) {
 data.frame(type=z$type[1],n_comparisons=nrow(z),weight=sum(z$weight),estimate=weighted.mean(z$estimate,z$weight),contribution=sum(z$weight*z$estimate))
}))
write.csv(summary,file.path(out,'type_summary.csv'),row.names=FALSE)
fit <- lm(incearn_ln~reform_math+factor(state)+factor(class),data=d)
twfe <- unname(coef(fit)['reform_math'])
stopifnot(abs(twfe-sum(b$estimate*b$weight))<1e-10)
write.csv(data.frame(twfe=twfe,decomposed=sum(b$estimate*b$weight),weight_sum=sum(b$weight),n_units=26,n_obs=520),file.path(out,'twfe.csv'),row.names=FALSE)
unit <- d[!duplicated(d$state),]
unit$cohort <- ifelse(is.na(unit$reformyr_math),0,unit$reformyr_math)
write.csv(as.data.frame(table(unit$cohort)),file.path(out,'cohorts.csv'),row.names=FALSE)
writeLines(c(paste('R',getRversion()),paste('bacondecomp',packageVersion('bacondecomp')),paste('fixest',packageVersion('fixest')),
 paste('bacondecomp package license',utils::packageDescription('bacondecomp')$License),
 'Source: https://github.com/evanjflack/bacondecomp#example ; data/math_reform.rda from installed package.',
 'Public aggregated data from Goodman (2019, Journal of Labor Economics), compulsory high-school mathematics coursework and subsequent income.',
 'Exact documented unweighted formula incearn_ln ~ reform_math; state/class panel; 26 states x 20 graduating classes, 1975-1994.',
 'Raw source columns and never-treated missing adoption-year values retained in snapshot. Analysis maps only missing reformyr_math to cohort 0.',
 'No sample exclusions, covariates, sampling weights or Stata execution.',
 'Actual bacondecomp comparison output, actual rounded bacon_summary output, and independent full-unit/time-dummy OLS on the full and every restricted pair sample.',
 'This is the documented aggregated-data worked example, not every specification or microdata table in the underlying paper.'),file.path(out,'versions.txt'))
