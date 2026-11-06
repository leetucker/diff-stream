# Sun-Abraham and BJS references with high-cardinality fixed effects.
# Usage: Rscript absorbed_reference.R tests/data/absorbed_panel.csv tests/data/absorbed_reference
# Sun-Abraham: fixest sunab(no_agg = TRUE), never-treated controls, clustered
# by unit without small-sample adjustments (CRV0). BJS: didimputation with the
# fixed effects in its first stage, event study (horizon = TRUE) and overall
# (post-treatment periods only, wtr). Its weighted standard errors use a
# squared-weight first-stage Gram, so the weighted case checks estimates only.
suppressPackageStartupMessages({library(data.table); library(fixest)})
args <- commandArgs(trailingOnly = TRUE)
panel <- fread(args[1]); out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
setFixest_notes(FALSE)
# didimputation's first stage is a fixest fit: demean to the same tolerance.
setFixest_estimation(fixef.tol = 1e-11)

cases <- list(
  list(name = "county", rhs = "1", fe = "county^time", A = 0, weighted = FALSE, unbalanced = FALSE),
  list(name = "two_terms", rhs = "x", fe = "county^time + industry^time", A = 0, weighted = FALSE, unbalanced = FALSE),
  list(name = "weighted", rhs = "x", fe = "county^time + industry^time", A = 0, weighted = TRUE, unbalanced = FALSE),
  list(name = "unbalanced", rhs = "x", fe = "county^time + industry^time", A = 0, weighted = FALSE, unbalanced = TRUE),
  list(name = "anticipation", rhs = "x", fe = "county^time + industry^time", A = 1, weighted = FALSE, unbalanced = FALSE),
  # Rows missing from never-treated units too: some county-periods are left
  # without untreated observations, and didimputation drops the treated rows
  # there that cannot be imputed.
  list(name = "unimputable", rhs = "x", fe = "county^time + industry^time", A = 0, weighted = FALSE, unbalanced = FALSE,
       unimputable = TRUE)
)
for (c in cases) {
  d <- copy(panel)
  if (!c$weighted) d[, w := 1]
  # Rows missing from treated units only, so every treated outcome can be imputed.
  if (c$unbalanced) d <- d[!(id %% 7 == 0 & time %in% c(3, 6) & first_treat > 0)]
  if (isTRUE(c$unimputable)) d <- d[!(id %% 7 == 0 & time %in% c(3, 6))]
  d[, gref := ifelse(first_treat == 0, 1000L, first_treat)]
  model <- as.formula(sprintf("y ~ sunab(gref, time, ref.p = %d, no_agg = TRUE) + %s | id + time + %s",
                              -c$A - 1, c$rhs, c$fe))
  sa <- feols(model, data = d, weights = ~w, vcov = ~id, fixef.rm = "none", fixef.tol = 1e-11,
              ssc = ssc(K.adj = FALSE, G.adj = FALSE))
  keep <- grep(":cohort::", names(coef(sa)), fixed = TRUE)
  parts <- strsplit(names(coef(sa))[keep], ":cohort::", fixed = TRUE)
  g <- as.integer(vapply(parts, `[`, "", 2))
  e <- as.integer(sub("time::", "", vapply(parts, `[`, "", 1), fixed = TRUE))
  fwrite(data.table(term = sprintf("g::%d:t::%d", g, g + e), estimate = coef(sa)[keep],
                    se = sqrt(diag(vcov(sa))[keep])), file.path(out, paste0("sunab_", c$name, ".csv")))
  fwrite(as.data.table(vcov(sa)[keep, keep]), file.path(out, paste0("sunab_", c$name, "_vcov.csv")),
         col.names = FALSE)

  # didimputation reads anticipation as an earlier cohort date.
  d[, gbjs := ifelse(first_treat == 0, 0L, first_treat - c$A)]
  first <- as.formula(sprintf("~ %s | id + time + %s", if (c$rhs == "1") "0" else c$rhs, c$fe))
  weights <- if (c$weighted) "w" else NULL
  dynamic <- didimputation::did_imputation(d, yname = "y", gname = "gbjs", tname = "time", idname = "id",
                                           first_stage = first, horizon = TRUE, wname = weights)
  dynamic$term <- as.character(as.integer(dynamic$term) - c$A)
  d[, post := as.numeric(first_treat > 0 & time >= first_treat)]
  overall <- didimputation::did_imputation(d, yname = "y", gname = "gbjs", tname = "time", idname = "id",
                                           first_stage = first, wname = weights, wtr = "post")
  fwrite(rbind(dynamic, overall)[, c("term", "estimate", "std.error")],
         file.path(out, paste0("bjs_", c$name, ".csv")))
}
# Library paths are machine state: keep file names only.
writeLines(c(gsub("/[^ ;]*/", "", capture.output(sessionInfo())),
             paste("didimputation", packageVersion("didimputation"))),
           file.path(out, "R_session.txt"))
