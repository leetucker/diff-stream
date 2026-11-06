# Weighted TWFE on original rows; actual bacondecomp on exact integer replications.
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
 if(case=='shifted') {d$time <- d$time+20;d$first_treat[d$first_treat>0] <- d$first_treat[d$first_treat>0]+20}
 if(case=='complete') {
  d <- d[!(d$id%%4==0 & d$time %in% c(3,7)),]
  keep <- names(which(table(d$id)==9));d <- d[d$id %in% as.integer(keep),]
 }
 if(case=='constant') d$y <- d$constant
 if(case=='null') d$y <- d$baseline
 if(case=='two_period') {
  d <- d[d$first_treat!=2 & d$time %in% c(4,5),]
  d$time <- d$time-3;d$first_treat[d$first_treat>0] <- d$first_treat[d$first_treat>0]-3
 }
 d$D <- as.integer(d$first_treat>0 & d$time>=d$first_treat)
 for(kind in c('rational','irrational')) {
  # Rational weights are integer multiplicities / 5, allowing exact package
  # validation. Irrational weights are checked through original-row WLS.
  d$w <- if(kind=='rational') (1+5*(d$id%%9))/5 else sqrt(2+d$id%%9)
  prefix <- paste(case,kind,sep='_')
  full <- lm(y~D+factor(id)+factor(time),data=d,weights=w)
  if(kind=='rational') {
   expanded <- list();newid <- 0L
   for(u in unique(d$id)) {
    block <- d[d$id==u,]
    reps <- as.integer(round(5*block$w[1]))
    for(j in seq_len(reps)) {block$id <- newid;expanded[[length(expanded)+1]] <- block;newid <- newid+1L}
   }
   expanded <- do.call(rbind,expanded)
   b <- bacondecomp::bacon(y~D,data=expanded,id_var='id',time_var='time',quietly=TRUE)
   b$treated[is.infinite(b$treated) | b$treated==99999] <- 0
   b$untreated[is.infinite(b$untreated) | b$untreated==99999] <- 0
   stopifnot(abs(sum(b$weight*b$estimate)-unname(coef(full)['D']))<1e-9)
  } else {
   # Independent full-dummy WLS for every comparison. Comparison weights are
   # validated separately through exact replication above.
   groups <- sort(unique(ifelse(d$first_treat>max(d$time),0,d$first_treat)))
   tt <- groups[groups>0];pairs <- list()
   if(0 %in% groups) for(g in tt) pairs[[length(pairs)+1]] <- c(g,0)
   if(length(tt)>1) for(i in seq_len(length(tt)-1)) for(j in (i+1):length(tt)) {
    pairs[[length(pairs)+1]] <- c(tt[i],tt[j]);pairs[[length(pairs)+1]] <- c(tt[j],tt[i])
   }
   b <- list()
   normalized <- ifelse(d$first_treat>max(d$time),0,d$first_treat)
   for(pair in pairs) {
    g <- pair[1];h <- pair[2]
    use <- normalized %in% pair
    if(h>g) use <- use & d$time<h
    if(h>0 && h<g) use <- use & d$time>=h
    z <- d[use,];z$pairD <- as.integer(normalized[use]==g & z$time>=g)
    f <- lm(y~pairD+factor(id)+factor(time),data=z,weights=w)
    b[[length(b)+1]] <- data.frame(treated=g,untreated=h,estimate=unname(coef(f)['pairD']))
   }
   b <- do.call(rbind,b)
  }
  write.csv(b,file.path(out,paste0(prefix,'.csv')),row.names=FALSE)
  checks[[length(checks)+1]] <- data.frame(case=case,weights=kind,twfe=unname(coef(full)['D']),n_units=length(unique(d$id)),n_obs=nrow(d),total_unit_weight=sum(d$w[!duplicated(d$id)]))
 }
}
write.csv(do.call(rbind,checks),file.path(out,'twfe.csv'),row.names=FALSE)
writeLines(c(paste('R',getRversion()),paste('bacondecomp',packageVersion('bacondecomp')),
 'Original-row lm WLS with full unit/time dummies; rational and irrational positive time-invariant weights.',
 'bacondecomp has no native weights argument: rational weights (1+5*(id%9))/5 use exact integer unit replication.',
 'Package comparison weights and estimates are actual bacondecomp outputs on that expanded panel, not patched package formulas.',
 'Irrational sqrt(2+id%9) comparison estimates use independent restricted-sample full-dummy WLS.',
 'Never-treated Inf/99999 sentinel restored to 0.'),file.path(out,'versions.txt'))
