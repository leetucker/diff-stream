# Native did full refits of the exact ordinary unit-pairs samples exported by
# refit_bootstrap_validation.py. No modifications to did or DRDID are made.
args <- commandArgs(TRUE)
root <- args[1]
options(warn=1)
for (case in c("balanced", "unbalanced_last")) {
  for (method in c("reg", "ipw", "dr")) {
    points <- list()
    for (b in 0:9) {
      d <- read.csv(file.path(root, case, paste0("sample_", b, ".csv")))
      d$g <- ifelse(d$first_treat > max(d$time), 0, d$first_treat)
      fit <- did::att_gt(yname="y", tname="time", idname="id", gname="g",
        xformla=~x, data=d, weightsname="w", est_method=method,
        allow_unbalanced_panel=case != "balanced", control_group="nevertreated",
        base_period="varying", bstrap=FALSE, cband=FALSE, faster_mode=FALSE)
      ag <- did::aggte(fit, type="dynamic", bstrap=FALSE, cband=FALSE)
      points[[b+1]] <- data.frame(draw=b-1, term=paste0("event::",ag$egt), estimate=ag$att.egt)
    }
    tab <- do.call(rbind, points)
    write.csv(tab, file.path(root, paste0(case,"_",method,"_draws.csv")), row.names=FALSE)
    summaries <- lapply(points[[1]]$term, function(term) {
      beta <- tab$estimate[tab$draw == -1 & tab$term == term]
      z <- tab$estimate[tab$draw >= 0 & tab$term == term]
      q <- quantile(z, c(.025,.975), type=7)
      se <- sd(z)
      data.frame(term=term, estimate=beta, se=se, percentile_low=q[1],
        percentile_high=q[2], basic_low=2*beta-q[2], basic_high=2*beta-q[1],
        normal_low=beta-qnorm(.975)*se, normal_high=beta+qnorm(.975)*se)
    })
    write.csv(do.call(rbind,summaries), file.path(root,paste0(case,"_",method,"_summary.csv")),row.names=FALSE)
  }
}
writeLines(capture.output(sessionInfo()), file.path(root,"R_session.txt"))
