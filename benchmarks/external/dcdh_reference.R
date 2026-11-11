# de Chaisemartin and D'Haultfoeuille references from the authors' R DIDmultiplegtDYN.
# Usage: Rscript dcdh_reference.R tests/data/dcdh_panel.csv tests/data/dcdh_reference
#
# did_multiplegt_dyn with 4 effects and 3 placebos, effects_equal = TRUE, on each panel
# and with the options normalized, only_never_switchers, switchers = "in" / "out",
# controls = c("x", "z"), weight = "w" (varying over time), trends_nonparam = "region"
# and trends_lin = TRUE, alone and combined. Each case writes the estimates and standard
# errors of the effects, the placebos and the average total effect (not estimated with
# trends_lin), their numbers of switchers (and the switchers' total weight), and the
# p-values of the joint test that the placebos are zero and of the
# test that the effects are equal; those tests use the covariances between the
# estimates.
#
# The covariance matrix did_multiplegt_dyn returns (coef$vcov) is written too, as
# *_vcov_returned.csv, but not compared: its off-diagonal entries disagree with the
# package's own joint tests and are not a covariance matrix (some correlations exceed
# one). Its diagonal is the squared standard errors.
suppressPackageStartupMessages({library(data.table); library(polars); library(DIDmultiplegtDYN)})
args <- commandArgs(trailingOnly = TRUE)
all <- fread(args[1]); out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

cases <- list(
  list(name = "switching", panel = "switching"),
  list(name = "no_never", panel = "no_never"),
  list(name = "normalized", panel = "switching", normalized = TRUE),
  list(name = "only_never_switchers", panel = "switching", only_never_switchers = TRUE),
  list(name = "switchers_in", panel = "switching", switchers = "in"),
  list(name = "switchers_out", panel = "switching", switchers = "out"),
  list(name = "covariates", panel = "switching", controls = c("x", "z")),
  list(name = "unbalanced", panel = "unbalanced"),
  list(name = "unbalanced_covariates", panel = "unbalanced", controls = c("x", "z")),
  list(name = "weights", panel = "switching", weight = "w"),
  list(name = "weights_covariates", panel = "switching", weight = "w", controls = c("x", "z")),
  list(name = "weights_unbalanced", panel = "unbalanced", weight = "w"),
  list(name = "trends_by", panel = "switching", outcome = "y_trend", trends_nonparam = "region"),
  list(name = "trends_by_covariates", panel = "switching", outcome = "y_trend",
       trends_nonparam = "region", controls = c("x", "z")),
  list(name = "linear_trends", panel = "switching", outcome = "y_trend", trends_lin = TRUE),
  list(name = "linear_trends_covariates", panel = "switching", outcome = "y_trend",
       trends_lin = TRUE, controls = c("x", "z")),
  list(name = "linear_trends_normalized", panel = "switching", outcome = "y_trend",
       trends_lin = TRUE, normalized = TRUE),
  list(name = "linear_trends_weights", panel = "switching", outcome = "y_trend",
       trends_lin = TRUE, weight = "w"),
  list(name = "linear_trends_no_never", panel = "no_never", outcome = "y_trend", trends_lin = TRUE),
  list(name = "linear_trends_trends_by", panel = "switching", outcome = "y_trend",
       trends_lin = TRUE, trends_nonparam = "region")
)
for (c in cases) {
  df <- as.data.frame(all[panel == c$panel])
  r <- did_multiplegt_dyn(
    df, outcome = if (is.null(c$outcome)) "y" else c$outcome, group = "id", time = "time", treatment = "d", effects = 4, placebo = 3,
    normalized = isTRUE(c$normalized), only_never_switchers = isTRUE(c$only_never_switchers),
    switchers = if (is.null(c$switchers)) "" else c$switchers, controls = c$controls,
    weight = c$weight, trends_nonparam = c$trends_nonparam, trends_lin = isTRUE(c$trends_lin),
    effects_equal = TRUE, graph_off = TRUE
  )
  res <- r$results
  rows <- function(m, prefix) {
    if (is.null(m)) return(NULL)
    m <- as.matrix(m)
    data.table(coefficient = paste0(prefix, seq_len(nrow(m))), estimate = m[, "Estimate"],
               se = m[, "SE"], switchers = m[, "Switchers"],
               switchers_weighted = m[, "Switchers.w"])
  }
  average <- if (is.null(res$ATE)) NULL else
    data.table(coefficient = "average_total_effect", estimate = res$ATE[1, "Estimate"],
               se = res$ATE[1, "SE"], switchers = res$ATE[1, "Switchers"],
               switchers_weighted = res$ATE[1, "Switchers.w"])
  table <- rbind(rows(res$Effects, "effect::"), rows(res$Placebos, "placebo::"), average)
  fwrite(table, file.path(out, paste0(c$name, ".csv")))
  fwrite(data.table(test = c("placebos_zero", "effects_equal"),
                    pvalue = c(res$p_jointplacebo, res$p_equality_effects)),
         file.path(out, paste0(c$name, "_tests.csv")))
  fwrite(as.data.table(r$coef$vcov), file.path(out, paste0(c$name, "_vcov_returned.csv")))
}
fwrite(data.table(package = c("DIDmultiplegtDYN", "polars", "R"),
                  version = c(as.character(packageVersion("DIDmultiplegtDYN")),
                              as.character(packageVersion("polars")),
                              paste(R.version$major, R.version$minor, sep = "."))),
       file.path(out, "versions.csv"))
