# Usage: Rscript rcs_reference.R tests/data/rcs_sample.csv tests/data/rcs_reference
# Callaway-Sant'Anna on repeated cross sections: actual did att_gt(panel=FALSE),
# analytic standard errors (each observation independent), and its aggregations.
args <- commandArgs(TRUE)
d <- read.csv(args[1]); out <- args[2]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
cases <- expand.grid(control = c("nevertreated", "notyettreated"), base = c("varying", "universal"),
                     method = c("reg", "ipw", "dr"), weighted = c(FALSE, TRUE), stringsAsFactors = FALSE)
for (i in seq_len(nrow(cases))) {
  k <- cases[i, ]
  a <- did::att_gt(yname = "y", tname = "time", gname = "first_treat", data = d, panel = FALSE,
                   xformla = ~x + z, control_group = k$control, base_period = k$base,
                   est_method = k$method, weightsname = if (k$weighted) "w" else NULL,
                   bstrap = FALSE, cband = FALSE)
  prefix <- paste("rcs", k$control, k$base, k$method, if (k$weighted) "w" else "u", sep = "_")
  write.csv(data.frame(g = a$group, t = a$t, att = a$att, se = a$se),
            file.path(out, paste0(prefix, ".csv")), row.names = FALSE)
  V <- as.matrix(a$V_analytical) / a$n
  stopifnot(max(abs(diag(V) - a$se^2), na.rm = TRUE) < 1e-10)
  write.csv(V, file.path(out, paste0(prefix, "_vcov.csv")), row.names = FALSE)
  for (kind in c("dynamic", "simple", "group", "calendar")) {
    ag <- did::aggte(a, type = kind, bstrap = FALSE, cband = FALSE)
    z <- if (kind == "simple") data.frame(event = NA, att = ag$overall.att, se = ag$overall.se) else
      data.frame(event = ag$egt, att = ag$att.egt, se = ag$se.egt)
    write.csv(z, file.path(out, paste0(prefix, "_", kind, ".csv")), row.names = FALSE)
  }
}
writeLines(c(paste("R", getRversion()), paste("did", packageVersion("did")),
             paste("DRDID", packageVersion("DRDID")),
             "did att_gt(panel=FALSE), analytic SEs, observations independent."),
           file.path(out, "versions.txt"))
