# Aggregate profile of the dataset as baseline.R sees it: read.csv with its defaults, the reader the capture harness
# requires. Writes column names, R classes, missing counts, level counts for low-cardinality columns and numeric
# summaries; never a row. Usage: Rscript --vanilla profile.R <dataset> <out.json> <max levels>
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
out <- args[2]
max_levels <- as.integer(args[3])
options(error = function() {
  writeLines(substr(geterrmessage(), 1, 1000), paste0(out, ".error"))
  quit(save = "no", status = 1)
})

d <- read.csv(args[1])
header <- names(read.csv(args[1], nrows = 1, check.names = FALSE))
num <- function(x) if (length(x) == 1 && is.finite(x)) x else NULL

columns <- lapply(seq_along(d), function(i) {
  x <- d[[i]]
  present <- x[!is.na(x)]
  values <- unique(present)
  col <- list(name = header[i], r_name = names(d)[i], class = class(x)[1], missing = sum(is.na(x)),
              distinct = length(values))
  # levels only where they summarise: few values, each shared by rows on average
  if (length(values) <= max_levels && length(values) <= nrow(d) / 2) {
    counts <- table(present)
    col$levels <- lapply(seq_along(counts), function(k) list(value = type.convert(names(counts)[k], as.is = TRUE),
                                                              n = unname(as.integer(counts[k]))))
    if (is.character(x)) for (k in seq_along(col$levels)) col$levels[[k]]$value <- names(counts)[k]
  }
  if (is.numeric(x)) {
    f <- x[is.finite(x)]
    col$non_finite <- sum(!is.finite(x) & !is.na(x))
    if (length(f)) {
      col$min <- min(f)
      col$max <- max(f)
      col$mean <- mean(f)
      col$sd <- num(if (length(f) > 1) sd(f) else NA)
    }
  }
  col
})

write_json(list(file = basename(args[1]), rows = nrow(d), row_names_from_file = .row_names_info(d) > 0,
                columns = columns), out, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE)
