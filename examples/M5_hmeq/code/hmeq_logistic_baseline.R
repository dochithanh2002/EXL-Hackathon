# Standalone HMEQ benchmark; see ../README.md for scope and provenance.
# Usage: Rscript --vanilla hmeq_logistic_baseline.R [input.csv] [output_dir]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("Usage: Rscript --vanilla hmeq_logistic_baseline.R [input.csv] [output_dir]")
}
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
root <- dirname(dirname(normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)))
input <- if (length(args) >= 1L) args[[1L]] else file.path(root, "data", "hmeq.csv")
output <- if (length(args) == 2L) args[[2L]] else file.path(root, "outputs")

hmeq <- read.csv(input, na.strings = c("", "NA", "."), stringsAsFactors = FALSE)
predictors <- c("LOAN", "MORTDUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO")
required <- c("BAD", predictors, "JOB")
if (!all(required %in% names(hmeq))) {
  stop("Missing required columns: ", paste(setdiff(required, names(hmeq)), collapse = ", "))
}
if (anyNA(hmeq$BAD) || !setequal(unique(hmeq$BAD), 0:1) ||
    any(table(hmeq$BAD) < 5L)) {
  stop("BAD must be coded as 0/1, with at least five rows per class and no missing values.")
}
if (!all(vapply(hmeq[predictors], is.numeric, logical(1))) ||
    any(!is.finite(hmeq$LOAN) | hmeq$LOAN <= 0)) {
  stop("Predictors must be numeric; LOAN must be finite and strictly positive.")
}

# Stratify by outcome so the small holdout preserves the benchmark default rate.
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(42)
train_id <- sort(unlist(lapply(split(seq_len(nrow(hmeq)), hmeq$BAD), function(ids) {
  sample(ids, floor(0.8 * length(ids)))
}), use.names = FALSE))
test_id <- setdiff(seq_len(nrow(hmeq)), train_id)

train <- hmeq[train_id, required]
test <- hmeq[test_id, required]
for (v in setdiff(predictors, "LOAN")) {
  fill <- median(train[[v]], na.rm = TRUE)
  if (!is.finite(fill)) stop("Cannot estimate a training median for ", v)
  train[[v]][is.na(train[[v]])] <- fill
  test[[v]][is.na(test[[v]])] <- fill
}

fit <- glm(BAD ~ log(LOAN) + MORTDUE + YOJ + DEROG + DELINQ + CLAGE + NINQ + CLNO,
           data = train, family = binomial(), na.action = na.fail)
if (!fit$converged || any(!is.finite(coef(fit)))) stop("Logistic fit failed.")
p <- predict(fit, newdata = test, type = "response")
y <- test$BAD
if (length(p) != length(y) || any(!is.finite(p))) stop("Invalid holdout predictions.")

auc <- function(y, p) {
  positive <- y == 1
  n_pos <- sum(positive)
  n_neg <- sum(!positive)
  if (!n_pos || !n_neg) return(NA_real_)
  (sum(rank(p, ties.method = "average")[positive]) - n_pos * (n_pos + 1) / 2) /
    (n_pos * n_neg)
}

metrics <- data.frame(
  train_n = nrow(train), test_n = nrow(test), test_defaults = sum(y),
  default_rate = mean(y), mean_predicted_pd = mean(p), roc_auc = auc(y, p),
  brier = mean((p - y)^2), brier_constant = mean((mean(train$BAD) - y)^2)
)

# JOB is used for an audit view only. Keep missing jobs as a visible group.
job <- ifelse(is.na(test$JOB), "(missing)", test$JOB)
by_job <- do.call(rbind, lapply(split(seq_along(y), job), function(i) {
  data.frame(job = job[i[1L]], n = length(i), defaults = sum(y[i]),
             observed_rate = mean(y[i]), mean_predicted_pd = mean(p[i]),
             roc_auc = auc(y[i], p[i]))
}))

dir.create(output, recursive = TRUE, showWarnings = FALSE)
write.csv(metrics, file.path(output, "holdout_metrics.csv"), row.names = FALSE)
write.csv(data.frame(csv_row = test_id, BAD = y, pd = unname(p)),
          file.path(output, "holdout_predictions.csv"), row.names = FALSE)
write.csv(data.frame(csv_row = train_id), file.path(output, "training_rows.csv"),
          row.names = FALSE)
write.csv(data.frame(term = names(coef(fit)), estimate = unname(coef(fit))),
          file.path(output, "coefficients.csv"), row.names = FALSE)
write.csv(by_job, file.path(output, "job_diagnostics.csv"), row.names = FALSE)
print(metrics, row.names = FALSE, digits = 5)
cat("Outputs:", normalizePath(output), "\n")
