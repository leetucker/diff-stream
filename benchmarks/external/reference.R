# Usage: Rscript reference.R input.csv output-directory
# Installs are deliberately separate; install.packages(c('did','fixest','didimputation')).
args <- commandArgs(TRUE)
d <- read.csv(args[1])
out <- args[2]
dir.create(out, showWarnings=FALSE, recursive=TRUE)
versions <- list(R=R.version.string)
for (p in c("did", "fixest", "didimputation")) {
  if (!requireNamespace(p, quietly=TRUE)) stop(paste("Missing reference package:", p))
  versions[[p]] <- as.character(packageVersion(p))
}
writeLines(capture.output(str(versions)), file.path(out, "versions.txt"))
for (control in c("nevertreated", "notyettreated")) {
  for (base in c("varying", "universal")) {
    a <- did::att_gt(yname="y", tname="time", idname="id", gname="first_treat",
                     data=d, xformla=~1, est_method="reg", control_group=control,
                     base_period=base, bstrap=FALSE, cband=FALSE)
    prefix <- paste0("cs_", control, "_", base)
    write.csv(data.frame(g=a$group, t=a$t, att=a$att, se=a$se),
              file.path(out, paste0(prefix, ".csv")), row.names=FALSE)
    # did stores the asymptotic variance of sqrt(N)*(ATT_hat-ATT).
    covariance <- as.matrix(a$V_analytical) / a$n
    stopifnot(max(abs(diag(covariance) - a$se^2), na.rm=TRUE) < 1e-10)
    write.csv(covariance,
              file.path(out, paste0(prefix, "_vcov.csv")), row.names=FALSE)
    for (type in c("dynamic", "simple")) {
      ag <- did::aggte(a, type=type, bstrap=FALSE, cband=FALSE)
      z <- if(type == "simple") data.frame(event=NA, att=ag$overall.att, se=ag$overall.se) else
        data.frame(event=ag$egt, att=ag$att.egt, se=ag$se.egt)
      write.csv(z, file.path(out, paste0(prefix, "_", type, ".csv")), row.names=FALSE)
    }
  }
}
d$gref <- ifelse(d$first_treat==0, max(d$time)+1, d$first_treat)
sa <- fixest::feols(y ~ sunab(gref,time) | id + time, data=d, vcov=~id,
                    ssc=fixest::ssc(K.adj=FALSE, G.adj=FALSE))
tab <- fixest::coeftable(sa)
write.csv(data.frame(term=rownames(tab), att=tab[,1], se=tab[,2]),
          file.path(out,"sunab_dynamic.csv"), row.names=FALSE)
bjs <- didimputation::did_imputation(data=d, yname="y", gname="first_treat",
                                    tname="time", idname="id", horizon=TRUE)
write.csv(bjs, file.path(out,"bjs_dynamic.csv"), row.names=FALSE)
