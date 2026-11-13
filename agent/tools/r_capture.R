# Generic reference-capture harness for the R->Python migration agent.
# Sources a legacy R script unmodified, then exports everything needed to prove parity.
# Usage: Rscript r_capture.R <script> <model_obj> <train_obj> <test_obj> <seed> <out_dir>
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 6)
script <- args[1]; model_obj <- args[2]; train_obj <- args[3]; test_obj <- args[4]
seed <- as.integer(args[5]); out_dir <- args[6]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(jsonlite))

full <- function(df, path) {                       # 17 significant digits
  num <- vapply(df, is.double, logical(1))
  df[num] <- lapply(df[num], function(x) sprintf("%.17g", x))
  write.csv(df, path, row.names = FALSE, quote = FALSE)
}

env <- new.env(parent = globalenv())
set.seed(seed)
t0 <- Sys.time()
sys.source(script, envir = env, keep.source = FALSE)
runtime <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

m <- get(model_obj, envir = env)
tr <- get(train_obj, envir = env); te <- get(test_obj, envir = env)
stopifnot(inherits(m, "glm") || inherits(m, "lm"))

s <- summary(m)$coefficients
full(data.frame(term = rownames(s), estimate = s[, 1], std_error = s[, 2], row.names = NULL),
     file.path(out_dir, "coefficients.csv"))

yname <- all.vars(formula(m))[1]
rows_tr <- suppressWarnings(as.integer(rownames(tr))); rows_te <- suppressWarnings(as.integer(rownames(te)))
if (anyNA(c(rows_tr, rows_te)) || anyDuplicated(c(rows_tr, rows_te)))
  stop("train/test rownames are not unique original row numbers (script reset or rebuilt them); ",
       "row identity with the raw file cannot be established")
# Response on the numeric scale the model uses: factor/character/logical binomial responses become the
# event indicator (1 = any level except the model's first, as glm does); numeric responses are kept exactly.
y_lv <- if (is.factor(m$model[[1]]) || is.character(m$model[[1]])) levels(as.factor(m$model[[1]])) else NULL
resp <- function(df) {
  y <- df[[yname]]
  if (is.logical(y)) return(as.numeric(y))
  if (is.factor(y) || is.character(y)) {
    if (is.null(y_lv)) stop("non-numeric response '", yname, "' but the model response is not a factor")
    return(ifelse(is.na(y), NA_real_, as.numeric(as.character(y) != y_lv[1])))
  }
  as.numeric(y)
}
full(data.frame(row = c(rows_tr, rows_te), set = c(rep("train", length(rows_tr)), rep("test", length(rows_te)))),
     file.path(out_dir, "split.csv"))
pred <- data.frame(
  row = c(rows_tr, rows_te),
  set = c(rep("train", length(rows_tr)), rep("test", length(rows_te))),
  y = c(resp(tr), resp(te)),
  pd = c(predict(m, newdata = tr, type = "response"), predict(m, newdata = te, type = "response")))
full(pred[order(pred$row), ], file.path(out_dir, "predictions.csv"))

meta <- list(
  r_version = R.version.string, seed = seed, runtime_s = runtime,
  packages = sort(setdiff(loadedNamespaces(), c("base", "stats", "utils", "methods", "graphics", "grDevices", "datasets"))),
  class = class(m), formula = paste(deparse(formula(m)), collapse = " "),
  family = if (!is.null(m$family)) m$family$family else "gaussian",
  link = if (!is.null(m$family)) m$family$link else "identity",
  terms = rownames(s), response = yname, response_levels = y_lv,
  xlevels = m$xlevels, contrasts = m$contrasts,
  n_train = nrow(tr), n_test = nrow(te),
  converged = isTRUE(m$converged), iterations = m$iter,
  null_deviance = m$null.deviance, deviance = m$deviance, aic = AIC(m),
  control = list(epsilon = m$control$epsilon, maxit = m$control$maxit),
  column_classes_after_preprocessing = lapply(tr, function(x) class(x)[1])
)
write_json(meta, file.path(out_dir, "meta.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("captured", nrow(pred), "predictions to", out_dir, "\n")
