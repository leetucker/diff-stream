# Clustered and continuous-baseline references for did_dcdh from the authors' R
# DIDmultiplegtDYN.
# Usage: Rscript dcdh_extensions_reference.R tests/data/dcdh_panel.csv \
#   tests/data/dcdh_continuous_panel.csv tests/data/dcdh_extensions_reference
#
# did_multiplegt_dyn with 4 effects and 3 placebos and effects_equal = TRUE:
#
# - cluster_*: dcdh_panel.csv with standard errors clustered by `cl` (groups numbered
#   id %/% 8, so 50 clusters of 8 groups) or by `region` (3), alone and with
#   normalized, controls, an unbalanced panel, weights that vary over time,
#   trends_nonparam and trends_lin;
# - continuous_*: dcdh_continuous_panel.csv, whose first-period treatment is
#   continuous, with continuous = 1 or 2 (a polynomial of that degree in it), alone and
#   with normalized, controls, an unbalanced panel, weights, switchers in, and
#   clustering;
# - continuous_unbalanced_construction: the estimator the continuous option describes,
#   built by hand and run without the option on the unbalanced panel: the treatment
#   binarized (0 before the first switch, then +1 or -1 by the switch's direction) and
#   the controls 1{t >= j} d1^k (j = 2..8, k = 1, 2), with d1 the first observed
#   treatment. With gaps in switchers' periods, the continuous option itself
#   (continuous_unbalanced, kept for the record) differs from this construction; with
#   none, they agree.
#
# Each case writes the estimates and standard errors of the effects, the placebos and
# the average total effect, their numbers of switchers (and the switchers' total
# weight), and the p-values of the joint placebo test and the equal-effects test.
suppressPackageStartupMessages({library(data.table); library(polars); library(DIDmultiplegtDYN)})
args <- commandArgs(trailingOnly = TRUE)
switching <- fread(args[1]); continuous <- fread(args[2]); out <- args[3]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
switching[, cl := id %/% 8]
switching[, region_code := as.integer(factor(region))]  # DIDmultiplegtDYN needs a numeric cluster

cases <- list(
  list(name = "cluster", data = "switching", panel = "switching", cluster = "cl"),
  list(name = "cluster_region", data = "switching", panel = "switching", cluster = "region_code"),
  list(name = "cluster_normalized", data = "switching", panel = "switching", cluster = "cl",
       normalized = TRUE),
  list(name = "cluster_covariates", data = "switching", panel = "switching", cluster = "cl",
       controls = c("x", "z")),
  list(name = "cluster_unbalanced", data = "switching", panel = "unbalanced", cluster = "cl"),
  list(name = "cluster_weights", data = "switching", panel = "switching", cluster = "cl",
       weight = "w"),
  list(name = "cluster_trends_by", data = "switching", panel = "switching", cluster = "cl",
       outcome = "y_trend", trends_nonparam = "region"),
  list(name = "cluster_linear_trends", data = "switching", panel = "switching", cluster = "cl",
       outcome = "y_trend", trends_lin = TRUE),
  list(name = "continuous_1", data = "continuous", panel = "continuous", continuous = 1),
  list(name = "continuous_2", data = "continuous", panel = "continuous", continuous = 2),
  list(name = "continuous_normalized", data = "continuous", panel = "continuous",
       continuous = 2, normalized = TRUE),
  list(name = "continuous_covariates", data = "continuous", panel = "continuous",
       continuous = 2, controls = "x"),
  list(name = "continuous_unbalanced", data = "continuous", panel = "continuous_unbalanced",
       continuous = 2),
  list(name = "continuous_weights", data = "continuous", panel = "continuous",
       continuous = 2, weight = "w"),
  list(name = "continuous_switchers_in", data = "continuous", panel = "continuous",
       continuous = 2, switchers = "in"),
  list(name = "continuous_cluster", data = "continuous", panel = "continuous",
       continuous = 2, cluster = "cluster")
)
# the continuous option's estimator, by hand
built <- copy(continuous[panel == "continuous_unbalanced"])
setorder(built, id, time)
built[, d1 := d[1], by = id]
# (numeric: an integer column would turn the never-switchers' Inf into NA)
built[, first_switch := {w <- time[d != d1]; if (length(w)) as.numeric(min(w)) else Inf}, by = id]
built[, direction := if (is.finite(first_switch[1]))
  sign(mean(d[time >= first_switch[1]]) - d1[1]) else 0, by = id]
built[, d_bin := ifelse(time >= first_switch, direction, 0)]
polynomial <- character(0)
for (j in 2:8) for (k in 1:2) {
  name <- sprintf("fe_%d_%d", j, k)
  built[, (name) := as.numeric(time >= j) * d1^k]
  polynomial <- c(polynomial, name)
}
cases[[length(cases) + 1]] <- list(name = "continuous_unbalanced_construction", data = "built",
                                   panel = "continuous_unbalanced", treatment = "d_bin",
                                   controls = polynomial)
for (c in cases) {
  source_data <- switch(c$data, switching = switching, continuous = continuous, built = built)
  df <- as.data.frame(source_data[panel == c$panel])
  r <- suppressMessages(did_multiplegt_dyn(
    df, outcome = if (is.null(c$outcome)) "y" else c$outcome, group = "id", time = "time",
    treatment = if (is.null(c$treatment)) "d" else c$treatment, effects = 4, placebo = 3,
    normalized = isTRUE(c$normalized), switchers = if (is.null(c$switchers)) "" else c$switchers,
    controls = c$controls, weight = c$weight, trends_nonparam = c$trends_nonparam,
    trends_lin = isTRUE(c$trends_lin), cluster = c$cluster, continuous = c$continuous,
    effects_equal = TRUE, graph_off = TRUE
  ))
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
}
fwrite(data.table(package = c("DIDmultiplegtDYN", "polars", "R"),
                  version = c(as.character(packageVersion("DIDmultiplegtDYN")),
                              as.character(packageVersion("polars")),
                              paste(R.version$major, R.version$minor, sep = "."))),
       file.path(out, "versions.csv"))
