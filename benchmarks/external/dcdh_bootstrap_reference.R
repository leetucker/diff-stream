# Bootstrap standard errors of did_multiplegt_dyn (DIDmultiplegtDYN), for did_dcdh's
# refit bootstrap.
# Usage: Rscript dcdh_bootstrap_reference.R tests/data/dcdh_panel.csv tests/data/dcdh_bootstrap_reference
#
# The "switching" panel, 2 effects and 1 placebo, bootstrap = c(500, 1): resampling
# groups, and with cluster = "cl" (groups numbered id %/% 8, 50 clusters) resampling
# clusters. The draws use R's random numbers, so they are not did_dcdh's: the
# standard errors agree only up to simulation error (about 4.5% for the difference
# of two independent 500-draw standard errors).
suppressPackageStartupMessages({library(data.table); library(polars); library(DIDmultiplegtDYN)})
args <- commandArgs(trailingOnly = TRUE)
d <- fread(args[1])[panel == "switching"]
out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
d[, cl := id %/% 8]
rows <- list()
for (cl in c("none", "cl")) {
  r <- suppressMessages(did_multiplegt_dyn(as.data.frame(d), "y", "id", "time", "d",
    effects = 2, placebo = 1, cluster = if (cl == "none") NULL else "cl",
    bootstrap = c(500, 1), graph_off = TRUE))
  m <- rbind(as.matrix(r$results$Effects), as.matrix(r$results$Placebos))
  rows[[cl]] <- data.table(cluster = cl,
    coefficient = c("effect::1", "effect::2", "placebo::1"),
    estimate = m[, "Estimate"], se = m[, "SE"])
}
fwrite(rbindlist(rows), file.path(out, "bootstrap.csv"))
fwrite(data.table(package = c("DIDmultiplegtDYN", "polars", "R"),
                  version = c(as.character(packageVersion("DIDmultiplegtDYN")),
                              as.character(packageVersion("polars")),
                              paste(R.version$major, R.version$minor, sep = "."))),
       file.path(out, "versions.csv"))
