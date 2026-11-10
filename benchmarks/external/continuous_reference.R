# Continuous-treatment (dose) DiD references from the authors' R contdid.
# Usage: Rscript continuous_reference.R tests/data/continuous_panel.csv tests/data/continuous_reference
#
# contdid 0.1.1 always uses not-yet-treated controls and a varying base period: it
# does not pass control_group, base_period or weights on to ptetools (contdid issue
# #19). These references therefore use those settings only. The curves are
# evaluated on a grid spanning every cell's doses exactly, the only case in which
# contdid evaluates them with the basis it fitted (contdid issue #16).
#
# Point estimates are contdid's own. Standard errors are computed from the
# influence functions it returns, sqrt(sum(IF^2)) / n, not from its multiplier
# bootstrap, after three corrections, each a one-line change to the installed code:
#
# 1. contdid cont_did_acrt: a cell's ACRT influence function is an average over the
#    cell's treated units, but ptetools rescales it as one over all the cell's units,
#    so the ACRT standard errors are too small by the treated share (contdid issue
#    #13). Rescaled as in contdid PR #17.
# 2. ptetools process_dose_gt: the control units' part of the ATT(d) influence
#    function is minus the binarized ATT's, which has the opposite sign from the
#    derivative of ATT(d) = fit(d) - control mean. It reverses the covariance
#    between a unit's role as a treated unit in one cell and as a control in another
#    (not-yet-treated controls); the sign is removed.
# 3. ptetools pte_aggte: the term for the estimated cohort shares (wif) centers each
#    cohort indicator at the cohort's share of the treated units instead of all
#    units, and the "group" aggregation passes it each cell's cohort instead of the
#    list of cohorts. R did's compute.aggte uses the unnormalized shares and
#    group = glist; so do these.
#
# overall_*.csv also keeps the standard errors of the unchanged code (se_contdid).
suppressPackageStartupMessages({library(data.table); library(contdid); library(did)})
args <- commandArgs(trailingOnly = TRUE)
panel <- as.data.frame(fread(args[1])); out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
se_of <- function(IF) sqrt(colSums(as.matrix(IF)^2)) / nrow(as.matrix(IF))

originals <- list(
  contdid = list(cont_did_acrt = get("cont_did_acrt", envir = asNamespace("contdid"))),
  ptetools = list(process_dose_gt = get("process_dose_gt", envir = asNamespace("ptetools")),
                  pte_aggte = get("pte_aggte", envir = asNamespace("ptetools")))
)
# Replace text in one function's source; each change must match exactly once.
edited <- function(f, changes) {
  src <- deparse(f, width.cutoff = 500L)
  for (from in names(changes)) {
    hits <- grep(from, src, fixed = TRUE)
    if (length(hits) != 1) stop("expected one match for: ", from)
    src[hits] <- sub(from, changes[[from]], src[hits], fixed = TRUE)
  }
  g <- eval(parse(text = src))
  environment(g) <- environment(f)
  g
}
rescaled_acrt <- function(gt_data, ...) {
  res <- originals$contdid$cont_did_acrt(gt_data, ...)
  post <- gt_data[gt_data$name == "post", ]
  res$inf_func <- res$inf_func * nrow(post) / sum(post$D > 0)
  res
}
corrected <- list(
  contdid = list(cont_did_acrt = rescaled_acrt),
  ptetools = list(
    process_dose_gt = edited(originals$ptetools$process_dose_gt,
                             list("] <- -replicate(" = "] <- replicate(")),
    pte_aggte = edited(originals$ptetools$pte_aggte,
                       list("pg <- pg/sum(pg)" = "pg <- pg",
                            "pg = pgg, weights.ind = weights.ind, G = G, group = group)" =
                              "pg = pgg, weights.ind = weights.ind, G = G, group = glist)"))
  )
)
use <- function(set) for (ns in names(set)) for (f in names(set[[ns]]))
  assignInNamespace(f, set[[ns]][[f]], ns)

fit <- function(target, aggregation, degree, knots, dvals = NULL, fixed = TRUE) {
  use(if (fixed) corrected else originals)
  on.exit(use(originals))
  suppressWarnings(cont_did(
    yname = "y", tname = "time", idname = "id", dname = "dose", gname = "first_treat",
    data = panel, target_parameter = target, aggregation = aggregation,
    treatment_type = "continuous", degree = degree, num_knots = knots, dvals = dvals,
    control_group = "notyettreated", base_period = "varying", bstrap = FALSE, biters = 20
  ))
}

cases <- list(
  list(name = "cubic", degree = 3, knots = 0),
  list(name = "quadratic_knots", degree = 2, knots = 2)
)
grid <- seq(1, 10, length.out = 46)
for (c in cases) {
  level <- fit("level", "dose", c$degree, c$knots, grid)
  slope <- fit("slope", "dose", c$degree, c$knots, grid)
  fwrite(data.table(dose = level$dose, att = level$att.d, att_se = se_of(level$att.d_inffunc),
                    acrt = slope$acrt.d, acrt_se = se_of(slope$acrt.d_inffunc)),
         file.path(out, paste0("curves_", c$name, ".csv")))
  raw_level <- fit("level", "dose", c$degree, c$knots, grid, fixed = FALSE)
  raw_slope <- fit("slope", "dose", c$degree, c$knots, grid, fixed = FALSE)
  fwrite(data.table(coefficient = c("ATT", "ACRT"),
                    estimate = c(raw_level$overall_att, raw_slope$overall_acrt),
                    se = c(level$overall_att_se_if <- se_of(level$overall_att_inffunc),
                           se_of(slope$overall_acrt_inffunc)),
                    se_contdid = c(se_of(raw_level$overall_att_inffunc),
                                   se_of(raw_slope$overall_acrt_inffunc))),
         file.path(out, paste0("overall_", c$name, ".csv")))
  fwrite(data.table(dose = raw_level$dose, att_se_contdid = se_of(raw_level$att.d_inffunc),
                    acrt_se_contdid = se_of(raw_slope$acrt.d_inffunc)),
         file.path(out, paste0("curves_contdid_se_", c$name, ".csv")))
  es <- fit("slope", "eventstudy", c$degree, c$knots)$event_study
  fwrite(data.table(event = es$egt, estimate = es$att.egt,
                    se = se_of(es$inf.function$dynamic.inf.func.e)),
         file.path(out, paste0("event_slope_", c$name, ".csv")))
}
# The event study of the level is the Callaway-Sant'Anna event study of being
# treated, which contdid computes with ptetools::did_attgt. contdid 0.1.1 passes it
# the dose, which DRDID 1.4.0 rejects, so it comes from R did itself, as does the
# "group" aggregation the overall ATT is (for its standard error, as a check of
# correction 3).
gt <- att_gt(yname = "y", tname = "time", idname = "id", gname = "first_treat", data = panel,
             control_group = "notyettreated", base_period = "varying", bstrap = FALSE)
es <- aggte(gt, type = "dynamic", bstrap = FALSE)
fwrite(data.table(event = es$egt, estimate = es$att.egt, se = se_of(es$inf.function$dynamic.inf.func.e)),
       file.path(out, "event_level.csv"))
group <- aggte(gt, type = "group", bstrap = FALSE)
fwrite(data.table(estimate = group$overall.att, se = group$overall.se),
       file.path(out, "overall_att_did.csv"))
fwrite(data.table(package = c("contdid", "ptetools", "splines2", "did", "DRDID", "R"),
                  version = c(as.character(packageVersion("contdid")),
                              as.character(packageVersion("ptetools")),
                              as.character(packageVersion("splines2")),
                              as.character(packageVersion("did")),
                              as.character(packageVersion("DRDID")),
                              paste(R.version$major, R.version$minor, sep = "."))),
       file.path(out, "versions.csv"))
