# Installs did2s when the normal install fails. did2s lists HonestDiD and
# staggered as Imports, but only its sensitivity-analysis helpers
# (honest_did_did2s, get_honestdid_obj_did2s) and event_study()'s staggered
# option use them, and they pull in a chain of compiled optimization packages
# (CVXR, clarabel, gmp, ...). This installs the CRAN source with those two
# dropped from DESCRIPTION; did2s() itself, the estimator the benchmarks run,
# does not call them. The setup log says which route was taken.
#
#   Rscript install_did2s.R <library> <repository> [dependencies already installed]
args <- commandArgs(TRUE)
lib <- args[1]
repos <- args[2]
work <- tempfile("did2s")
dir.create(work)
tarball <- download.packages("did2s", destdir = work, repos = repos, type = "source")[1, 2]
untar(tarball, exdir = work)
description <- file.path(work, "did2s", "DESCRIPTION")
lines <- readLines(description)
start <- grep("^Imports:", lines)
stop_at <- c(grep("^[A-Za-z/@-]+:", lines), length(lines) + 1)
end <- min(stop_at[stop_at > start]) - 1
imports <- paste(lines[start:end], collapse = " ")
imports <- sub("^Imports:\\s*", "", imports)
kept <- trimws(strsplit(imports, ",")[[1]])
kept <- kept[!kept %in% c("HonestDiD", "staggered")]
lines <- c(lines[seq_len(start - 1)], paste("Imports:", paste(kept, collapse = ", ")),
           if (end < length(lines)) lines[(end + 1):length(lines)])
writeLines(lines, description)
# the rest of the Imports (boot, broom, ...) are installed here: the normal install
# stopped at HonestDiD and staggered before it got to them
needed <- trimws(sub("\\(.*\\)", "", kept))
needed <- setdiff(needed, c("R", rownames(installed.packages())))
if (length(needed)) install.packages(needed, lib = lib, repos = repos)
install.packages(file.path(work, "did2s"), lib = lib, repos = NULL, type = "source",
                 dependencies = FALSE)
cat("did2s installed without HonestDiD and staggered\n")
