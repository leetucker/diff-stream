# One fit with an R package, for benchmarks/_harness.py: read the Parquet file
# into memory (timed as "read", as the Python libraries' reading is), estimate
# the event study, and print one line,
#
#   RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {"event::0": ...}, ...}
#
# Peak memory is the process's VmHWM, as for the Python configurations.
#
#   Rscript did_fit.R '<spec as JSON>'     (see the benchmarks' external_command)
#
# spec$fit is one of
#   cs           did::att_gt and aggte(type = "dynamic")
#   sunab        fixest::feols with sunab(): the interaction-weighted event study
#   bjs          didimputation::did_imputation(horizon = TRUE)
#   gardner      did2s::did2s with an event-study second stage
#   etwfe        etwfe::etwfe and emfx(type = "event")
#   stacked      the stack of Wing, Freedman and Hollingsworth built with data.table,
#                and fixest on it (as tests/data/stacked_reference was made)
#   bacon        bacondecomp::bacon (the TWFE estimate and the weight of each comparison type)
#   twfe_weights TwoWayFEWeights::twowayfeweights, type "feTR"
# spec$columns are the columns to read; spec$covariates, spec$fe (extra fixed
# effects, fixest syntax: "region^time"), spec$adjustment (cs: reg, ipw, dr),
# spec$unbalanced, spec$window (stacked: [lo, hi]), spec$threads.
#
# Estimates are reported as "event::<k>", k the periods since treatment, so that
# they can be compared with diff-stream's aggregate("event_time").

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})
spec <- fromJSON(commandArgs(TRUE)[1], simplifyVector = TRUE)

hwm_mb <- function() {
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  as.numeric(strsplit(trimws(sub("VmHWM:", "", line)), " +")[[1]][1]) / 1024
}
now <- function() proc.time()[["elapsed"]]
baseline <- hwm_mb()
need <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop(paste("Missing package:", pkg))
}
threads <- if (is.null(spec$threads)) 1L else as.integer(spec$threads)
data.table::setDTthreads(threads)
covariates <- spec$covariates
extra_fe <- spec$fe
rhs <- if (length(covariates)) paste(covariates, collapse = " + ") else NULL
event <- function(k, estimate) setNames(as.list(as.numeric(estimate)), paste0("event::", k))

t0 <- now()
if (requireNamespace("arrow", quietly = TRUE)) {
  d <- as.data.table(arrow::read_parquet(spec$path, col_select = spec$columns))
} else {
  need("nanoparquet")
  d <- as.data.table(nanoparquet::read_parquet(spec$path)[, spec$columns])
}
t_read <- now() - t0

t0 <- now()
coefs <- switch(spec$fit,
  cs = {
    need("did")
    xf <- if (length(covariates)) as.formula(paste("~", rhs)) else ~1
    a <- did::att_gt(yname = "y", tname = "time", idname = "id", gname = "first_treat",
                     xformla = xf, data = as.data.frame(d), est_method = spec$adjustment,
                     allow_unbalanced_panel = isTRUE(spec$unbalanced), faster_mode = TRUE,
                     bstrap = FALSE, cband = FALSE)
    e <- did::aggte(a, type = "dynamic", bstrap = FALSE, cband = FALSE)
    event(e$egt, e$att.egt)
  },
  sunab = {
    need("fixest")
    fixest::setFixest_nthreads(threads)
    d[, gref := ifelse(first_treat == 0, max(time) + 100L, first_treat)]
    fe <- paste(c("id", "time", extra_fe), collapse = " + ")
    f <- as.formula(paste("y ~", paste(c(rhs, "sunab(gref, time)"), collapse = " + "), "|", fe))
    m <- fixest::feols(f, data = d, vcov = ~id, nthreads = threads,
                       ssc = fixest::ssc(K.adj = FALSE, G.adj = FALSE))
    b <- coef(m)
    b <- b[grepl("^time::", names(b))]
    event(sub("^time::", "", names(b)), b)
  },
  bjs = {
    need("didimputation")
    fixest::setFixest_nthreads(threads)
    fe <- paste(c("id", "time", extra_fe), collapse = " + ")
    first <- as.formula(paste("~", if (is.null(rhs)) "0" else rhs, "|", fe))
    r <- didimputation::did_imputation(data = d, yname = "y", gname = "first_treat",
                                       tname = "time", idname = "id", first_stage = first,
                                       horizon = TRUE)
    event(r$term, r$estimate)
  },
  gardner = {
    need("did2s")
    fixest::setFixest_nthreads(threads)
    d[, rel := ifelse(first_treat == 0, Inf, time - first_treat)]
    d[, treated := as.integer(first_treat > 0 & time >= first_treat)]
    fe <- paste(c("id", "time", extra_fe), collapse = " + ")
    m <- did2s::did2s(d, yname = "y", treatment = "treated",
                      first_stage = as.formula(paste("~", if (is.null(rhs)) "0" else rhs, "|", fe)),
                      second_stage = ~ i(rel, ref = c(-1, Inf)), cluster_var = "id", verbose = FALSE)
    b <- coef(m)
    event(sub("^rel::", "", names(b)), b)
  },
  etwfe = {
    need("etwfe")
    fixest::setFixest_nthreads(threads)
    m <- etwfe::etwfe(fml = as.formula(paste("y ~", if (is.null(rhs)) "0" else rhs)), tvar = time,
                      gvar = first_treat, data = d, ivar = id, cgroup = spec$control_group,
                      vcov = ~id)
    e <- etwfe::emfx(m, type = "event")
    event(e$event, e$estimate)
  },
  stacked = {
    need("fixest")
    fixest::setFixest_nthreads(threads)
    lo <- spec$window[1]; hi <- spec$window[2]
    first <- min(d$time); last <- max(d$time)
    pieces <- list()
    for (a in sort(unique(d$first_treat[d$first_treat > 0]))) {
      if (a + lo < first || a + hi > last) next
      sub <- d[(first_treat == a | first_treat == 0 | first_treat > a + hi) &
                 time >= a + lo & time <= a + hi]
      sub[, `:=`(sub_exp = a, D = as.integer(first_treat == a), e = time - a)]
      if (sum(sub$D == 0) == 0) next
      pieces[[length(pieces) + 1]] <- sub
    }
    s <- rbindlist(pieces)
    units <- s[, .(uw = 1), by = .(sub_exp, D, id)][, .(n = sum(uw)), by = .(sub_exp, D)]
    units[, share := n / sum(n), by = D]
    s <- merge(s, units[D == 1, .(sub_exp, sd = share)], by = "sub_exp")
    s <- merge(s, units[D == 0, .(sub_exp, sc = share)], by = "sub_exp")
    s[, Q := ifelse(D == 1, 1, sd / sc)]
    m <- fixest::feols(y ~ D + i(e, ref = -1) + i(e, D, ref = -1), data = s, weights = ~Q,
                       cluster = ~id, nthreads = threads,
                       ssc = fixest::ssc(adj = FALSE, cluster.adj = FALSE))
    b <- coef(m)
    b <- b[grepl("^e::-?[0-9]+:D$", names(b))]
    event(sub("^e::(-?[0-9]+):D$", "\\1", names(b)), b)
  },
  bacon = {
    need("bacondecomp")
    d[, treat := as.integer(first_treat > 0 & time >= first_treat)]
    b <- bacondecomp::bacon(y ~ treat, data = as.data.frame(d), id_var = "id", time_var = "time",
                            quietly = TRUE)
    out <- list(twfe = sum(b$estimate * b$weight))
    kinds <- c("Treated vs Untreated" = "treated_vs_never", "Earlier vs Later Treated" = "earlier_vs_later",
               "Later vs Earlier Treated" = "later_vs_earlier")
    for (type in unique(b$type)) {
      out[[paste0("weight::", kinds[[type]])]] <- sum(b$weight[b$type == type])
    }
    out
  },
  twfe_weights = {
    need("TwoWayFEWeights")
    d[, treat := as.integer(first_treat > 0 & time >= first_treat)]
    w <- TwoWayFEWeights::twowayfeweights(as.data.frame(d), "y", "id", "time", "treat",
                                          type = "feTR", summary_measures = TRUE)
    list(n_positive = w$nr_plus, n_negative = w$nr_minus, sum_negative = w$sum_minus,
         sensitivity = w$sensibility)
  },
  stop(paste("unknown fit:", spec$fit))
)
t_fit <- now() - t0

cat("RESULT ", toJSON(list(t_read = t_read, t_fit = t_fit, peak_memory_mb = hwm_mb(),
                           baseline_memory_mb = baseline, peak_disk_mb = 0,
                           n_obs = nrow(d), coefs = coefs), auto_unbox = TRUE, digits = NA),
    "\n", sep = "")
