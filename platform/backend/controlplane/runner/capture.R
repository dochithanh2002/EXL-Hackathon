# R baseline capture harness, evolved from legacy/agent/tools/r_capture.R.
# Sources baseline.R unmodified in a fresh environment, then exports everything the validator compares against.
# Usage: Rscript --vanilla capture.R <config.json>
#   config: script, dataset, objects {model, train, test (or null)}, seed (or null), out, and the spec's semantics
#   (ModelSpec.capture_config): response, response_type, event_coding, grouped, weights, prediction {kind, keys,
#   r_predict}, period_column, segments (the validation plan's subgroup columns)
# lm and glm fits are read through summary(); any other class through coef(), vcov(), logLik() and the spec's
# r_predict expression, so the extraction follows the spec rather than a list of supported classes.
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
fit_warnings <- character()
withCallingHandlers(sys.source(cfg$script, envir = env, keep.source = FALSE), warning = function(w) {
  fit_warnings <<- c(fit_warnings, conditionMessage(w))
})
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

is_glm <- inherits(m, "glm")
is_lm <- inherits(m, "lm")
family <- if (is_glm) m$family$family else if (is_lm) "gaussian" else NULL
pred_cfg <- cfg$prediction
keys <- as.character(unlist(pred_cfg$keys))
in_frame <- function(expr, df) eval(parse(text = expr), df, env)
optional <- function(expr) tryCatch(expr, error = function(e) NULL)
field <- function(name) if (is.list(m)) m[[name]] else NULL      # S4 fits (lme4) have no `$`

# ---- response semantics: the observed value each prediction is compared with
coding <- cfg$event_coding
if (!is.null(coding) && cfg$response_type == "binary" && cfg$response %in% names(sets$train)) {
  y0 <- sets$train[[cfg$response]]
  if (is_glm && family %in% c("binomial", "quasibinomial")) {
    if (is.factor(y0) && coding == levels(y0)[1])
      stop(sprintf("the spec codes the event as '%s', but glm models the probability of the levels after the first ",
                   coding), sprintf("('%s'); the event coding must name the outcome R predicts", levels(y0)[1]))
    if ((is.numeric(y0) || is.logical(y0)) && coding != "1")
      stop("glm models the probability that a 0/1 or logical response is 1, so the event coding must be '1'")
  }
}
event_of <- function(y) {
  if (is.logical(y)) y <- as.integer(y)
  ifelse(is.na(y), NA_real_, as.numeric(as.character(y) == coding))
}
observed <- function(df) {
  n <- nrow(df)
  if (cfg$response_type == "grouped_binomial") {
    g <- cfg$grouped
    trials <- as.numeric(in_frame(g$trials, df))
    if (g$form == "successes_and_trials") return(as.numeric(in_frame(g$successes, df)) / trials)
    return(as.numeric(df[[cfg$response]]))
  }
  if (!cfg$response %in% names(df)) return(rep(NA, n))
  y <- df[[cfg$response]]
  if (pred_cfg$kind == "class_probability") return(as.character(y))
  if (cfg$response_type == "binary") return(event_of(y))
  as.numeric(y)
}
weight_of <- function(df) {                           # the model's prior weight of each row (trials when grouped)
  if (cfg$response_type == "grouped_binomial") return(as.numeric(in_frame(cfg$grouped$trials, df)))
  if (!is.null(cfg$weights)) return(as.numeric(in_frame(cfg$weights, df)))
  rep(1, nrow(df))
}

# ---- coefficients
coefficient_table <- function() {
  if (is_lm) {                                        # glm and lm: summary() keeps aliased terms as NA
    cf <- coef(m)
    sm <- summary(m)$coefficients
    stat <- function(j) { v <- rep(NA_real_, length(cf)); v[match(rownames(sm), names(cf))] <- sm[, j]; v }
    return(data.frame(term = names(cf), estimate = unname(cf), std_error = stat(2), statistic = stat(3),
                      p_value = stat(4)))
  }
  cf <- optional(coef(m))
  if (is.null(cf) || !is.numeric(cf)) return(data.frame(term = character(), estimate = numeric(),
                                                        std_error = numeric(), statistic = numeric(),
                                                        p_value = numeric()))
  if (is.matrix(cf)) cf <- setNames(as.vector(t(cf)), paste(rep(rownames(cf), each = ncol(cf)),
                                                            rep(colnames(cf), times = nrow(cf)), sep = ":"))
  if (is.numeric(field("zeta"))) cf <- c(cf, field("zeta"))          # MASS::polr cut points
  v <- optional(suppressWarnings(sqrt(diag(vcov(m)))))
  se <- rep(NA_real_, length(cf))
  if (!is.null(v) && !is.null(names(v)) && all(names(cf) %in% names(v))) se <- unname(v[names(cf)])
  else if (!is.null(v) && length(v) == length(cf)) se <- unname(v)
  data.frame(term = names(cf), estimate = unname(cf), std_error = se, statistic = NA_real_, p_value = NA_real_)
}
coefs <- coefficient_table()
write_full(coefs, "coefficients.csv")

fit_rows <- setdiff(rows$train, source_rows(names(field("na.action"))))
split <- do.call(rbind, lapply(names(sets), function(s) data.frame(
  row = rows[[s]], set = s, in_fit = if (s == "train") rows[[s]] %in% fit_rows else NA)))
if (!is.null(cfg$period_column)) {
  split$period <- unlist(lapply(names(sets), function(s) {
    if (!cfg$period_column %in% names(sets[[s]])) stop(sprintf("the %s set has no period column '%s'", s,
                                                               cfg$period_column))
    as.character(sets[[s]][[cfg$period_column]])
  }))
}
write_full(split, "split.csv")

# ---- predictions: one row per set, data row and key (class or quantile level; empty for one value per row)
predictions <- do.call(rbind, lapply(names(sets), function(s) {
  df <- sets[[s]]
  p <- eval(parse(text = pred_cfg$r_predict), list(model = m, newdata = df), env)
  if (pred_cfg$kind %in% c("scalar", "time_indexed")) {
    p <- as.numeric(p)
    if (length(p) != nrow(df)) stop(sprintf("r_predict returned %d values for the %d rows of the %s set", length(p),
                                            nrow(df), s))
    p <- matrix(p, ncol = 1)
    k <- ""
  } else {
    if (is.null(dim(p)) && pred_cfg$kind == "class_probability" && length(keys) == 2) p <- cbind(1 - p, p)
    p <- as.matrix(p)
    if (nrow(p) != nrow(df) || ncol(p) != length(keys))
      stop(sprintf("r_predict returned a %d x %d result for the %d rows of the %s set and the %d keys", nrow(p),
                   ncol(p), nrow(df), s, length(keys)))
    if (pred_cfg$kind == "class_probability" && setequal(colnames(p), keys)) p <- p[, keys, drop = FALSE]
    k <- keys
  }
  y <- observed(df)
  w <- weight_of(df)
  do.call(rbind, lapply(seq_along(k), function(j) data.frame(
    row = rows[[s]], set = s, key = k[j],
    y = if (pred_cfg$kind == "class_probability") ifelse(is.na(y), NA_real_, as.numeric(y == k[j])) else y,
    weight = w, prediction = unname(as.numeric(p[, j])))))
}))
write_full(predictions, "predictions.csv")

# ---- subgroup values of the validation plan's columns, one row per set, data row and column
segment_cols <- as.character(unlist(cfg$segments))
segment_types <- list()
segments <- data.frame(set = character(), row = integer(), column = character(), value = character())
for (col in segment_cols) {
  present <- vapply(sets, function(df) col %in% names(df), logical(1))
  if (!all(present)) { segment_types[[col]] <- "missing"; next }
  x <- sets$train[[col]]
  numeric <- is.numeric(x) && !is.factor(x) && length(unique(na.omit(x))) > 12
  segment_types[[col]] <- if (numeric) "numeric" else "categorical"
  segments <- rbind(segments, do.call(rbind, lapply(names(sets), function(s) {
    v <- sets[[s]][[col]]
    data.frame(set = s, row = rows[[s]], column = col,
               value = if (numeric) ifelse(is.na(v), NA, sprintf("%.17g", as.numeric(v))) else as.character(v))
  })))
}
write.csv(segments, out("segments.csv"), row.names = FALSE, na = "")

# ---- fit statistics
stat_of <- function(expr) { v <- optional(as.numeric(expr)); if (length(v) == 1) v else NULL }
fit <- list(nobs = stat_of(nobs(m)), df_residual = stat_of(df.residual(m)), log_lik = stat_of(logLik(m)),
            aic = stat_of(AIC(m)), bic = stat_of(BIC(m)))
if (is_glm) {
  s <- summary(m)
  fit <- c(fit, list(deviance = m$deviance, null_deviance = m$null.deviance, df_null = m$df.null,
                     dispersion = s$dispersion, pearson_chi2 = sum(residuals(m, type = "pearson")^2)))
} else if (is_lm) {
  s <- summary(m)
  fit <- c(fit, list(sigma = s$sigma, r_squared = s$r.squared, adj_r_squared = s$adj.r.squared,
                     rss = sum(residuals(m)^2)))
} else {
  fit$deviance <- stat_of(deviance(m))
}
write_meta(fit, "fit.json")

# ---- training curve: glm.fit's deviance by IRLS iteration, laid out like statsmodels' fit_history (the deviance
# at the starting values, then after each iteration). glm keeps only the last one, so the fit is replayed with
# maxit = 1, 2, ...; written only if the full replay reproduces the fit, so the curve is the path R took.
deviance_path <- function() {
  if (!identical(m$method, "glm.fit") || inherits(m, "negbin") ||
      any(c("start", "etastart", "mustart") %in% names(m$call))) return(NULL)
  x <- model.matrix(m)
  replay <- function(maxit) suppressWarnings(glm.fit(
    x, m$y, weights = m$prior.weights, offset = m$offset, family = m$family,
    control = modifyList(m$control, list(maxit = maxit, trace = FALSE)),
    intercept = attr(terms(m), "intercept") > 0))
  full <- replay(m$iter)
  if (full$iter != m$iter || !isTRUE(all.equal(full$deviance, m$deviance, tolerance = 1e-12)) ||
      !isTRUE(all.equal(unname(full$coefficients), unname(coef(m)), tolerance = 1e-10))) return(NULL)
  init <- list2env(list(y = m$y, weights = m$prior.weights, nobs = NROW(m$y), x = x, family = m$family,
                        start = NULL, etastart = NULL, mustart = NULL))
  suppressWarnings(eval(m$family$initialize, init))     # glm.fit's own starting values
  start <- sum(m$family$dev.resids(init$y, m$family$linkinv(m$family$linkfun(init$mustart)), init$weights))
  c(start, vapply(seq_len(m$iter - 1), function(k) replay(k)$deviance, numeric(1)), full$deviance)
}
if (is_glm && !is.null(path <- optional(deviance_path())) && all(is.finite(path)))
  write_meta(list(deviance = path), "fit_history.json")

# ---- design summary for the Diagnostician: aggregates of what the fit consumed, never a row
summary_of <- function(x) {
  if (is.null(x) || !length(x)) return(NULL)
  x <- as.numeric(x)
  list(n = length(x), sum = sum(x), mean = mean(x), min = min(x), max = max(x), n_zero = sum(x == 0))
}
X <- optional(model.matrix(m))
mf <- optional(model.frame(m))
prior <- if (is_glm) m$prior.weights else optional(weights(m))
write_meta(list(
  n_fit = length(fit_rows), n_train = length(rows$train),
  columns = if (is.null(X)) NULL else lapply(seq_len(ncol(X)), function(j) {
    x <- X[, j]
    list(name = colnames(X)[j], mean = mean(x), sd = if (length(x) > 1) sd(x) else NA, min = min(x), max = max(x))
  }),
  reference_levels = lapply(field("xlevels"), function(l) l[1]), contrasts = field("contrasts"),
  weights = summary_of(prior), offset = summary_of(if (is.null(mf)) NULL else model.offset(mf))
), "design.json")

# ---- metadata
converged <- if (is_glm) isTRUE(m$converged) else if (is_lm) TRUE else {
  v <- if (!is.null(field("converged"))) field("converged") else if (is.numeric(field("convergence")))
    field("convergence") == 0 else NA
  if (length(v) == 1 && !is.na(v)) as.logical(v) else NULL
}
if (!is.null(cfg$convergence) && cfg$convergence$mode == "expression") {
  converged <- eval(parse(text = cfg$convergence$r_check), list(model = m, fit_warnings = fit_warnings), env)
  if (!is.logical(converged) || length(converged) != 1 || is.na(converged))
    stop("convergence.r_check must return one non-missing logical value")
}
if (!is.null(cfg$convergence) && cfg$convergence$mode == "not_applicable") converged <- NULL
packages <- sort(setdiff(loadedNamespaces(), c("base", "stats", "utils", "methods", "graphics", "grDevices", "datasets")))
y_levels <- if (cfg$response %in% names(sets$train) && is.factor(sets$train[[cfg$response]]))
  levels(sets$train[[cfg$response]]) else NULL
write_meta(list(
  r_version = R.version.string, seed = cfg$seed,
  packages = setNames(lapply(packages, function(p) as.character(packageVersion(p))), packages),
  class = I(class(m)), formula = optional(paste(deparse(formula(m)), collapse = " ")),
  family = family, link = if (is_glm) m$family$link else if (is_lm) "identity" else NULL,
  response = cfg$response, response_type = cfg$response_type, event_coding = coding, response_levels = y_levels,
  prediction = list(kind = pred_cfg$kind, keys = I(keys)),
  design_columns = if (is.null(X)) coefs$term else colnames(X), aliased = coefs$term[is.na(coefs$estimate)],
  xlevels = field("xlevels"), contrasts = field("contrasts"),
  offset = !is.null(mf) && !is.null(model.offset(mf)), weights = !is.null(prior) && any(prior != 1),
  n = lapply(rows, length), n_fit = length(fit_rows), row_identity_check = "passed",
  segments = segment_types,
  separation_warning = any(grepl("fitted probabilities numerically 0 or 1", fit_warnings, fixed = TRUE)),
  converged = converged, closed_form = is_lm && !is_glm, iterations = if (is_glm) m$iter else NULL,
  control = if (is_glm) list(epsilon = m$control$epsilon, maxit = m$control$maxit) else NULL,
  column_classes = lapply(sets$train, function(x) class(x)[1])
), "meta.json")
writeLines(capture.output(sessionInfo()), out("sessionInfo.txt"))
write_meta(list(runtime_s = runtime), "timing.json")
cat("captured", nrow(predictions), "predictions\n")
