# Stacked DiD references (Wing, Freedman and Hollingsworth 2024) with fixest.
# Usage: Rscript stacked_reference.R tests/data/stacked_panel.csv tests/data/stacked_reference
# Builds the stacked data and corrective weights as in the authors' published
# R code, then fits the weighted event-study regression clustered by unit,
# without small-sample adjustments (CRV0).
suppressPackageStartupMessages({library(data.table); library(fixest)})
args <- commandArgs(trailingOnly = TRUE)
panel <- fread(args[1]); out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

stack <- function(d, lo, hi, anticipation) {
  first <- min(d$time); last <- max(d$time)
  pieces <- list()
  for (a in sort(unique(d$first_treat[d$first_treat > 0]))) {
    if (a + lo < first || a + hi > last) next
    sub <- d[(first_treat == a | first_treat == 0 | first_treat - anticipation > a + hi) &
               time >= a + lo & time <= a + hi]
    sub[, `:=`(sub_exp = a, D = as.integer(first_treat == a), e = time - a)]
    if (sum(sub$D == 0) == 0) next
    pieces[[length(pieces) + 1]] <- sub
  }
  s <- rbindlist(pieces)
  units <- s[, .(uw = w[1]), by = .(sub_exp, D, id)][, .(n = sum(uw)), by = .(sub_exp, D)]
  units[, share := n / sum(n), by = D]
  s <- merge(s, units[D == 1, .(sub_exp, sd = share)], by = "sub_exp")
  s <- merge(s, units[D == 0, .(sub_exp, sc = share)], by = "sub_exp")
  s[, Q := ifelse(D == 1, 1, sd / sc) * w]
  s
}

cases <- list(
  list(name = "balanced", lo = -3, hi = 2, A = 0, weighted = FALSE, unbalanced = FALSE),
  list(name = "weighted", lo = -3, hi = 2, A = 0, weighted = TRUE, unbalanced = FALSE),
  list(name = "unbalanced", lo = -2, hi = 3, A = 0, weighted = FALSE, unbalanced = TRUE),
  list(name = "anticipation", lo = -4, hi = 2, A = 1, weighted = FALSE, unbalanced = FALSE)
)
for (c in cases) {
  d <- copy(panel)
  if (!c$weighted) d[, w := 1]
  if (c$unbalanced) d <- d[!(id %% 7 == 0 & time %in% c(3, 6))]
  if (c$A > 0) d <- d[first_treat != 2]
  s <- stack(d, c$lo, c$hi, c$A)
  ref <- -c$A - 1
  model <- as.formula(sprintf("y ~ D + i(e, ref = %d) + i(e, D, ref = %d)", ref, ref))
  fit <- feols(model, data = s, weights = ~Q,
               cluster = ~id, ssc = ssc(adj = FALSE, cluster.adj = FALSE))
  keep <- grep("^e::-?[0-9]+:D$", names(coef(fit)))
  b <- coef(fit)[keep]
  fwrite(data.table(term = sub("^e::(-?[0-9]+):D$", "event::\\1", names(b)), estimate = b,
                    se = sqrt(diag(vcov(fit))[keep])), file.path(out, paste0(c$name, ".csv")))
  fwrite(as.data.table(vcov(fit)[keep, keep]), file.path(out, paste0(c$name, "_vcov.csv")),
         col.names = FALSE)
}
writeLines(capture.output(sessionInfo()), file.path(out, "R_session.txt"))
