# HonestDiD references for tests/test_sensitivity.py, on the mpdta panel of R's did.
# Usage: Rscript honestdid_reference.R tests/data/mpdta.csv tests/data/honestdid_reference
#
# 1. The package README's example: did::att_gt with base_period = "universal" (never
#    treated controls, doubly robust), the dynamic aggregate, and honest_did() (not
#    exported in 0.2.8: HonestDiD:::honest_did.AGGTEobj) with
#    type = "relative_magnitude" (methods C-LF and Conditional) and "smoothness" (FLCI
#    and C-LF), at honest_did's default 100 grid points. The event study it uses
#    (beta and the covariance from the influence functions) is written too.
# 2. A two-way fixed effects event study without staggered timing: the never-treated
#    counties and the 2006 cohort, fixest::feols(lemp ~ i(rel, ref = -1) |
#    countyreal + year, cluster = ~countyreal), with createSensitivityResults(_relative
#    Magnitudes) on its coefficients and clustered covariance. A case in which R stops
#    with an error is recorded with its message.
#
# R's two simulated quantities are made reproducible as in hdfe-stream's
# benchmarks/external/honestdid_reference.R: the FLCI's folded-normal quantile is
# computed exactly, and every least-favorable critical value is recorded with a
# fingerprint of its covariance, so the test can use R's and compare exactly.
suppressPackageStartupMessages({
  library(did)
  library(HonestDiD)
  library(fixest)
})
args <- commandArgs(TRUE)
d <- read.csv(args[1])
out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
stopifnot(as.character(packageVersion("HonestDiD")) == "0.2.8")

qfold_exact <- function(p, mu = 0, sd = 1, numSims = 10^6, seed = 0) {
  vapply(mu, function(m) {
    m <- abs(m)
    f <- function(c) pnorm(c - m, sd = sd) - pnorm(-c - m, sd = sd) - p
    uniroot(f, c(0, m + 12 * sd), tol = 1e-15)$root
  }, numeric(1))
}
assignInNamespace(".qfoldednormal", qfold_exact, "HonestDiD")
orig_lf <- HonestDiD:::.compute_least_favorable_cv
rec <- new.env()
rec$items <- list()
assignInNamespace(".compute_least_favorable_cv", function(X_T, sigma, hybrid_kappa, sims = 1000,
                                                         rowsForARP = NULL, seed = 0) {
  v <- orig_lf(X_T, sigma, hybrid_kappa, sims, rowsForARP, seed)
  S <- sigma
  X <- X_T
  if (!is.null(rowsForARP)) {
    if (!is.null(X)) X <- if (is.vector(X)) X[rowsForARP] else X[rowsForARP, , drop = FALSE]
    S <- S[rowsForARP, rowsForARP]
  }
  rec$items[[length(rec$items) + 1]] <- list(sum = sum(S), trace = sum(diag(S)),
                                             xsum = if (is.null(X)) NA else sum(X),
                                             m = NROW(S), cv = v)
  v
}, "HonestDiD")

runs <- list()
record <- function(source, restriction, method, M, f) {
  for (m in M) {
    rec$items <- list()
    r <- tryCatch(suppressWarnings(f(m)), error = function(e) e)
    if (inherits(r, "error")) {
      runs[[length(runs) + 1]] <<- list(source = source, restriction = restriction,
                                        method = method, M = m, error = conditionMessage(r))
      next
    }
    runs[[length(runs) + 1]] <<- list(source = source, restriction = restriction,
                                      method = method, M = m, lb = r$lb[1], ub = r$ub[1],
                                      lf = rec$items)
  }
}

# ---- 1. Callaway and Sant'Anna, universal base period ---------------------
a <- att_gt(yname = "lemp", tname = "year", idname = "countyreal", gname = "first.treat",
            data = d, base_period = "universal", bstrap = FALSE, cband = FALSE)
es <- aggte(a, type = "dynamic", bstrap = FALSE, cband = FALSE)
inf <- es$inf.function$dynamic.inf.func.e
V <- t(inf) %*% inf / nrow(inf) / nrow(inf)
keep <- es$egt != -1
cs_es <- list(event = es$egt[keep], beta = es$att.egt[keep], vcov = V[keep, keep])
for (method in c("C-LF", "Conditional")) {
  record("cs", "relative_magnitudes", method, seq(0.5, 2, by = 0.5), function(m)
    HonestDiD:::honest_did.AGGTEobj(es, type = "relative_magnitude", Mbarvec = m, method = method)$robust_ci)
}
for (method in c("FLCI", "C-LF")) {
  record("cs", "smoothness", method, c(0, 0.01, 0.02), function(m)
    HonestDiD:::honest_did.AGGTEobj(es, type = "smoothness", Mvec = m, method = method)$robust_ci)
}

# ---- 2. TWFE event study, one adoption cohort ------------------------------
d2 <- subset(d, first.treat %in% c(0, 2006))
d2$rel <- ifelse(d2$first.treat == 0, -1, d2$year - d2$first.treat)
fit <- feols(lemp ~ i(rel, ref = -1) | countyreal + year, data = d2, cluster = ~countyreal)
b <- coef(fit)
Vt <- vcov(fit)
ev <- as.integer(sub("rel::", "", names(b)))
o <- order(ev)
b <- b[o]
Vt <- Vt[o, o]
ev <- ev[o]
npre <- sum(ev < -1)
npost <- sum(ev > -1)
twfe_es <- list(event = ev, beta = unname(b), vcov = unname(Vt))
for (method in c("C-LF", "Conditional")) {
  record("twfe", "relative_magnitudes", method, c(0.5, 1, 2), function(m)
    createSensitivityResults_relativeMagnitudes(b, Vt, npre, npost, Mbarvec = m, method = method))
}
for (method in c("FLCI", "Conditional", "C-F", "C-LF")) {
  record("twfe", "smoothness", method, c(0, 0.02), function(m)
    createSensitivityResults(b, Vt, npre, npost, Mvec = m, method = method))
}

payload <- list(
  generated_by = "benchmarks/external/honestdid_reference.R",
  honestdid = as.character(packageVersion("HonestDiD")),
  did = as.character(packageVersion("did")),
  fixest = as.character(packageVersion("fixest")),
  r = R.version.string,
  event_studies = list(cs = cs_es, twfe = twfe_es),
  runs = runs
)
con <- gzfile(file.path(out, "honestdid.json.gz"), "w")
writeLines(jsonlite::toJSON(payload, digits = NA, auto_unbox = TRUE, matrix = "rowmajor",
                            na = "null"), con)
close(con)
cat("wrote", length(runs), "runs\n")
