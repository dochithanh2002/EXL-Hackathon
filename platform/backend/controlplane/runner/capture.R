# R baseline capture harness, evolved from legacy/agent/tools/r_capture.R.
# Sources baseline.R unmodified in a fresh environment, then exports everything the validator compares against.
# Usage: Rscript --vanilla capture.R <config.json>
#   config: script, dataset, response, objects {model, train, test (or null)}, seed (or null), out
# Everything written to `out` is deterministic except timing.json, so a rerun reproduces the same baseline id.
# Row identity: base R's readers (read.csv, read.table, ...) are masked so that the data frame read from the dataset
# file gets row names TAG + its 1-based data row. Subsetting, reordering and merge() (masked too) carry row names
# along; any name outside the tagged range (reset, rebuilt, or a frame not read through these readers) fails the
# capture. Exported row ids are the data rows, so Python takes exactly the same observations.
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1)
cfg <- fromJSON(args[1], simplifyVector = TRUE)
dir.create(cfg$out, recursive = TRUE, showWarnings = FALSE)
out <- function(name) file.path(cfg$out, name)

write_full <- function(df, name) {                    # doubles with 17 significant digits round-trip exactly
  num <- vapply(df, is.double, logical(1))
  df[num] <- lapply(df[num], function(x) ifelse(is.na(x), NA, sprintf("%.17g", x)))
  write.csv(df, out(name), row.names = FALSE, na = "")
}
# A failure writes R's message (never the log, which can hold printed rows) for the Analyst to see.
stage <- "the capture harness"
options(error = function() {
  write_json(list(stage = stage, message = substr(trimws(geterrmessage()), 1, 1000)), out("error.json"),
             auto_unbox = TRUE)
  quit(save = "no", status = 1)
})
write_meta <- function(x, name) write_json(x, out(name), auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null")

TAG <- 1000000000L
dataset <- normalizePath(cfg$dataset)
n_raw <- NA_integer_
tagged <- function(reader) { force(reader); function(file, ...) {
  df <- reader(file, ...)
  if (is.character(file) && length(file) == 1 && file.exists(file) && normalizePath(file) == dataset) {
    if (nrow(df) > .Machine$integer.max - TAG) stop("the dataset has too many rows to tag")
    attr(df, "row.names") <- TAG + seq_len(nrow(df))
    n_raw <<- nrow(df)
  }
  df
}}
is_tagged <- function(df) {
  rn <- if (is.data.frame(df)) .row_names_info(df, type = 0L) else NULL
  is.numeric(rn) && length(rn) > 0 && all(rn > TAG)
}
# merge() rebuilds row names; a join keeps each row of the tagged side's name when the row survives exactly once.
keep_rows <- function(x, y, ...) {
  side <- if (is_tagged(x)) "x" else if (is_tagged(y)) "y" else return(base::merge(x, y, ...))
  key <- ".capture_row"                               # a name neither frame has, so no input column is touched
  while (key %in% c(names(x), names(y))) key <- paste0(key, "_")
  if (side == "x") x[[key]] <- .row_names_info(x, type = 0L) else y[[key]] <- .row_names_info(y, type = 0L)
  res <- base::merge(x, y, ...)
  r <- res[[key]]
  res[[key]] <- NULL
  if (!anyNA(r) && !anyDuplicated(r)) attr(res, "row.names") <- as.integer(r)
  res
}
env <- new.env(parent = globalenv())
for (f in c("read.csv", "read.csv2", "read.table", "read.delim", "read.delim2"))
  assign(f, tagged(get(f, envir = asNamespace("utils"))), envir = env)
assign("merge", keep_rows, envir = env)
if (!is.null(cfg$seed)) set.seed(cfg$seed)
t0 <- Sys.time()
stage <- "baseline.R"
sys.source(cfg$script, envir = env, keep.source = FALSE)
stage <- "the capture harness"
runtime <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

get_obj <- function(role) {
  name <- cfg$objects[[role]]
  if (!exists(name, envir = env, inherits = FALSE)) stop(sprintf("baseline.R did not create the %s object '%s'", role, name))
  get(name, envir = env)
}
m <- get_obj("model")
sets <- list(train = get_obj("train"))
if (!is.null(cfg$objects$test)) sets$test <- get_obj("test")
for (s in names(sets)) if (!is.data.frame(sets[[s]])) stop(sprintf("the %s object is not a data frame", s))
if (!inherits(m, c("glm", "lm"))) stop(sprintf("capture is not implemented for model class '%s'", class(m)[1]))

# ---- row identity
if (is.na(n_raw)) stop("baseline.R did not read ", cfg$dataset, " with read.csv or read.table, so row identity ",
                       "with the dataset file cannot be established")
source_rows <- function(names) {
  r <- suppressWarnings(as.integer(names)) - TAG
  ifelse(!is.na(r) & r >= 1L & r <= n_raw, r, NA_integer_)
}
rows <- lapply(names(sets), function(s) {
  r <- source_rows(rownames(sets[[s]]))
  if (anyNA(r) || anyDuplicated(r))
    stop(sprintf("the %s set's rows cannot be traced to rows of the dataset file; ", s),
         "baseline.R must keep R's row names (avoid resetting them, e.g. via tibbles or rownames<-)")
  r
})
names(rows) <- names(sets)

y_levels <- if (is.factor(m$model[[1]])) levels(m$model[[1]]) else NULL
numeric_response <- function(y) {                     # the 0/1 or numeric scale the model fits
  if (is.logical(y)) return(as.numeric(y))
  if (is.factor(y) || is.character(y)) {
    if (is.null(y_levels)) stop(sprintf("the response '%s' is not numeric but the model response is not a factor", cfg$response))
    return(ifelse(is.na(y), NA_real_, as.numeric(as.character(y) != y_levels[1])))
  }
  as.numeric(y)
}
response_of <- function(df) if (cfg$response %in% names(df)) numeric_response(df[[cfg$response]]) else rep(NA_real_, nrow(df))

# ---- coefficients, design matrix, predictions
cf <- coef(m)
sm <- summary(m)$coefficients
term_stat <- function(j) { v <- rep(NA_real_, length(cf)); v[match(rownames(sm), names(cf))] <- sm[, j]; v }
write_full(data.frame(term = names(cf), estimate = unname(cf), std_error = term_stat(2), statistic = term_stat(3),
                      p_value = term_stat(4)), "coefficients.csv")

fit_rows <- setdiff(rows$train, source_rows(names(m$na.action)))
write_full(do.call(rbind, lapply(names(sets), function(s) data.frame(
  row = rows[[s]], set = s, in_fit = if (s == "train") rows[[s]] %in% fit_rows else NA))), "split.csv")

predictions <- do.call(rbind, lapply(names(sets), function(s) {
  df <- sets[[s]]
  response <- unname(predict(m, newdata = df, type = "response", na.action = na.pass))
  link <- if (inherits(m, "glm")) unname(predict(m, newdata = df, type = "link", na.action = na.pass)) else response
  data.frame(row = rows[[s]], set = s, y = response_of(df), link = link, response = response)
}))
write_full(predictions, "predictions.csv")

# ---- fit statistics
is_glm <- inherits(m, "glm")
s <- summary(m)
fit <- list(nobs = nobs(m), df_residual = df.residual(m), log_lik = as.numeric(logLik(m)), aic = AIC(m), bic = BIC(m))
if (is_glm) {
  fit <- c(fit, list(deviance = m$deviance, null_deviance = m$null.deviance, df_null = m$df.null,
                     dispersion = s$dispersion, pearson_chi2 = sum(residuals(m, type = "pearson")^2)))
} else {
  fit <- c(fit, list(sigma = s$sigma, r_squared = s$r.squared, adj_r_squared = s$adj.r.squared,
                     rss = sum(residuals(m)^2)))
}
write_meta(fit, "fit.json")

# ---- metadata
packages <- sort(setdiff(loadedNamespaces(), c("base", "stats", "utils", "methods", "graphics", "grDevices", "datasets")))
write_meta(list(
  r_version = R.version.string, seed = cfg$seed,
  packages = setNames(lapply(packages, function(p) as.character(packageVersion(p))), packages),
  class = I(class(m)), formula = paste(deparse(formula(m)), collapse = " "),
  family = if (is_glm) m$family$family else "gaussian", link = if (is_glm) m$family$link else "identity",
  response = cfg$response, response_levels = y_levels,
  design_columns = colnames(model.matrix(m)), aliased = names(cf)[is.na(cf)],
  xlevels = m$xlevels, contrasts = m$contrasts,
  offset = !is.null(model.offset(model.frame(m))), weights = !is.null(weights(m)) && any(weights(m) != 1),
  n = lapply(rows, length), n_fit = length(fit_rows), row_identity_check = "passed",
  converged = if (is_glm) isTRUE(m$converged) else TRUE, iterations = if (is_glm) m$iter else NULL,
  control = if (is_glm) list(epsilon = m$control$epsilon, maxit = m$control$maxit) else NULL,
  column_classes = lapply(sets$train, function(x) class(x)[1])
), "meta.json")
writeLines(capture.output(sessionInfo()), out("sessionInfo.txt"))
write_meta(list(runtime_s = runtime), "timing.json")
cat("captured", nrow(predictions), "predictions\n")
