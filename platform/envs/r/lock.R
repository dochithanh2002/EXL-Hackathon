# Re-resolve renv.lock: installs packages.txt, renv and all their dependencies (recommended packages included)
# from the snapshot into the environment library, then records them. Run by a maintainer when the package set or the snapshot changes; bootstrap.sh only restores.
# Usage: Rscript --vanilla lock.R <snapshot-date>
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1, grepl("^\\d{4}-\\d{2}-\\d{2}$", args[1]))
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))))
lib <- file.path(here, "library")
dir.create(lib, showWarnings = FALSE)
.libPaths(lib)
options(repos = c(CRAN = sprintf("https://packagemanager.posit.co/cran/%s", args[1])), Ncpus = parallel::detectCores())
pkgs <- c("renv", readLines(file.path(here, "packages.txt")))
base <- rownames(installed.packages(priority = "base"))
deps <- tools::package_dependencies(pkgs, db = available.packages(), recursive = TRUE,
                                    which = c("Depends", "Imports", "LinkingTo"))
need <- setdiff(unique(c(pkgs, unlist(deps))), c(base, rownames(installed.packages(lib))))
if (length(need)) install.packages(need, lib = lib, dependencies = FALSE)
project <- tempfile("renv-lock-")
dir.create(project)
renv::snapshot(project = project, library = lib, lockfile = file.path(here, "renv.lock"), packages = pkgs,
               prompt = FALSE)
