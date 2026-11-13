# Split sensitivity: refit the M2 final GLM over many seeds to see whether the notebook's
# reported test AUC (0.7542) is within normal split-to-split variation.
args <- commandArgs(trailingOnly = TRUE)   # <data.csv> <out_dir>
raw <- read.csv(args[1], check.names = FALSE)
colnames(raw) <- tolower(colnames(raw)); names(raw)[25] <- "y"; raw$id <- NULL
raw$education[raw$education %in% c(0,4,5,6)] <- 4; raw$marriage[raw$marriage == 0] <- 3
for (v in c("pay_0","pay_2","pay_3","pay_4","pay_5","pay_6")) raw[[v]][raw[[v]] %in% c(-2,-1,0)] <- 0
for (v in c("sex","education","marriage")) raw[[v]] <- factor(raw[[v]])
auc <- function(p, y) { r <- rank(p); n1 <- sum(y); n0 <- sum(1-y); (sum(r[y==1]) - n1*(n1+1)/2)/(n1*n0) }
res <- sapply(1:200, function(s) {
  set.seed(s)
  tr <- unlist(lapply(split(seq_len(nrow(raw)), raw$y), function(ix) sample(ix, ceiling(0.7*length(ix)))))
  m <- glm(y ~ limit_bal+sex+education+marriage+pay_0+pay_2+pay_3+pay_5+pay_6+pay_amt1+pay_amt2,
           data=raw[tr,], family=binomial)
  auc(predict(m, raw[-tr,], type="response"), raw$y[-tr])
})
q <- quantile(res, c(0.025, 0.5, 0.975))
out <- list(n_seeds=200, mean=mean(res), sd=sd(res), p2_5=q[[1]], median=q[[2]], p97_5=q[[3]],
            reported_notebook=0.7542, reported_percentile=mean(res <= 0.7542))
dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(out, file.path(args[2], "seed_sensitivity.json"), auto_unbox=TRUE, digits=NA, pretty=TRUE)
print(unlist(out))
