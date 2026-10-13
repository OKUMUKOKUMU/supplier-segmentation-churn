# =====================================================================================
# Supplier Segmentation & Churn Analysis in R (base R only)
# Author: Fordrane Albert Okumu
#
# PURPOSE
#   An independent rebuild of python/supplier_segmentation_churn.ipynb using only base R
#   (stats::kmeans, stats::glm, graphics). Every feature is re-derived from the raw
#   collection, payment, visit and credit files, not read from the Python outputs.
#   If two separate implementations produce the same features, segments, coefficients
#   and AUC, the analysis is very unlikely to contain a silent bug.
#
# STEPS
#   1. Load raw data (the weekly deliveries file is gzip-compressed; base R reads it directly)
#   2. Feature engineering at the 28-Jun-2026 snapshot (no data from after the snapshot)
#   3. Value framework: ABC volume classes x reliability x quality x fading
#   4. K-means clustering (k = 4) and profile-based naming
#   5. Churn label for suppliers active at the snapshot
#   6. Logistic regression (glm, binomial) on the same features and train/test split
#   7. Evaluation: AUC (Mann-Whitney), gains by decile; dry-season litres at risk
#   8. Cross-check against the Python outputs + Markdown report
#
# Run from the repository root:   Rscript R/supplier_segmentation_churn.R
# =====================================================================================

CUTOFF <- as.Date("2026-06-28")
ACTIVE_DAYS <- 28; GRACE_DAYS <- 28; SILENT_DAYS <- 63
out_dir <- file.path("outputs", "r"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
TEAL <- "#0f766e"; AMBER <- "#b45309"; GREY <- "#94a3b8"; RED <- "#b91c1c"

# -------------------------------------------------------------------------------------
# 1. Load
# -------------------------------------------------------------------------------------
sup  <- read.csv("data/suppliers.csv", stringsAsFactors = FALSE)
dl   <- read.csv(gzfile("data/deliveries_weekly.csv.gz"), stringsAsFactors = FALSE)
pay  <- read.csv("data/payments_monthly.csv", stringsAsFactors = FALSE)
mkt  <- read.csv("data/market_prices.csv", stringsAsFactors = FALSE)
vis  <- read.csv("data/extension_visits.csv", stringsAsFactors = FALSE)
cred <- read.csv("data/input_credit.csv", stringsAsFactors = FALSE)
dl$week_ending <- as.Date(dl$week_ending); pay$paid_date <- as.Date(pay$paid_date)
vis$visit_date <- as.Date(vis$visit_date); cred$advance_date <- as.Date(cred$advance_date)
sup$registered_date <- as.Date(sup$registered_date)
dl$rejection_reason[is.na(dl$rejection_reason)] <- ""
cat(sprintf("%d suppliers | %d supplier-weeks | %.1fm litres\n", nrow(sup), nrow(dl), sum(dl$litres_delivered) / 1e6))

# -------------------------------------------------------------------------------------
# 2. Feature engineering - ONLY data observable on or before the snapshot
# -------------------------------------------------------------------------------------
past <- dl[dl$week_ending <= CUTOFF, ]
ids  <- sort(unique(past$supplier_id))
fac  <- function(d) factor(d$supplier_id, levels = ids)
sum_by  <- function(d, col) { x <- tapply(d[[col]], fac(d), sum); x[is.na(x)] <- 0; as.numeric(x) }
win     <- function(days) past[past$week_ending > CUTOFF - days, ]

first_week <- as.Date(tapply(past$week_ending, fac(past), min), origin = "1970-01-01")
last_week  <- as.Date(tapply(past$week_ending, fac(past), max), origin = "1970-01-01")
F <- data.frame(supplier_id = ids, stringsAsFactors = FALSE)
F$recency_days <- as.numeric(CUTOFF - last_week)
F$weeks_since_first_delivery <- as.numeric(CUTOFF - first_week) %/% 7 + 1
F$litres_365d <- sum_by(win(364), "litres_delivered")
F$litres_13w <- sum_by(win(91), "litres_delivered")
F$days_delivered_13w <- sum_by(win(91), "days_delivered")
obs <- function(n) pmin(n, F$weeks_since_first_delivery)
F$litres_per_day_52w <- F$litres_365d / (7 * obs(52))
F$delivery_day_share_13w <- F$days_delivered_13w / (7 * obs(13))
F$avg_daily_litres_13w <- ifelse(F$days_delivered_13w > 0, F$litres_13w / pmax(F$days_delivered_13w, 1), 0)

# weekly litres matrix for the 26 weeks ending at the snapshot, masked to observable weeks
weeks <- CUTOFF - 7 * (25:0)
w26 <- past[past$week_ending >= weeks[1], ]
M <- matrix(0, length(ids), 26)
M[cbind(match(w26$supplier_id, ids), match(w26$week_ending, weeks))] <- w26$litres_delivered
mask <- outer(as.numeric(first_week), as.numeric(weeks), "<=") * 1
n <- rowSums(mask)
mn <- rowSums(M * mask) / n
sdv <- sqrt(rowSums((M - mn)^2 * mask) / pmax(n - 1, 1))
F$litres_cv_26w <- ifelse(n >= 2 & mn > 0, sdv / ifelse(mn > 0, mn, 1), 0)

# season-adjusted trend: last 8 weeks vs the 18 before, relative to the supplier's centre
L8 <- 19:26; P18 <- 1:18
last8 <- rowSums(M[, L8] * mask[, L8]) / pmax(rowSums(mask[, L8]), 1)
n_prior <- rowSums(mask[, P18])
prior <- rowSums(M[, P18] * mask[, P18]) / pmax(n_prior, 1)
centre <- sup$centre_id[match(ids, sup$supplier_id)]
c_ratio <- tapply(rowSums(M[, L8]) / 8, centre, sum) / tapply(rowSums(M[, P18]) / 18, centre, sum)
raw <- ifelse(n_prior > 0 & prior > 0, last8 / ifelse(prior > 0, prior, 1), 1)
F$litres_trend <- pmin(pmax(raw / c_ratio[centre], .1), 3)

# quality over the last 26 weeks
q <- past[past$week_ending > CUTOFF - 182, ]
qf <- factor(q$supplier_id, levels = ids)
ql <- as.numeric(tapply(q$litres_delivered, qf, sum))
F$fat_pct <- as.numeric(tapply(q$fat_pct * q$litres_delivered, qf, sum)) / ql
F$snf_pct <- as.numeric(tapply(q$snf_pct * q$litres_delivered, qf, sum)) / ql
F$rejection_rate <- as.numeric(tapply(q$litres_rejected, qf, sum)) / ql
F$rejection_weeks_26w <- sum_by(transform(q, r = as.numeric(litres_rejected > 0)), "r")
F$antibiotic_26w <- as.integer(ids %in% q$supplier_id[q$rejection_reason == "Antibiotic residue"])
F$adulteration_26w <- as.integer(ids %in% q$supplier_id[q$rejection_reason == "Adulteration (added water)"])

# price & payment: last 3 monthly payments actually PAID by the snapshot
p <- pay[pay$paid_date <= CUTOFF, ]
p <- p[order(p$supplier_id, p$month), ]
p <- p[ave(seq_len(nrow(p)), p$supplier_id, FUN = function(x) rev(seq_along(x))) <= 3, ]
p$county <- sup$county[match(p$supplier_id, sup$supplier_id)]
p$competitor_price_kes <- mkt$competitor_price_kes[match(paste(p$county, p$month), paste(mkt$county, mkt$month))]
pf <- factor(p$supplier_id, levels = ids)
F$price_realised_kes <- as.numeric(tapply(p$gross_kes, pf, sum) / tapply(p$litres_accepted, pf, sum))
F$competitor_price_kes <- as.numeric(tapply(p$competitor_price_kes, pf, mean))
F$price_gap_kes <- F$competitor_price_kes - F$price_realised_kes
F$payment_delay_days <- as.numeric(tapply(p$payment_delay_days, pf, mean))

# visits and credit
v <- vis[vis$visit_date <= CUTOFF & vis$visit_date > CUTOFF - 180, ]
F$advisory_visits_180d <- as.numeric(table(factor(v$supplier_id[v$visit_type != "Quality training"], levels = ids)))
F$quality_training_180d <- as.numeric(table(factor(v$supplier_id[v$visit_type == "Quality training"], levels = ids)))
adv <- sum_by(transform(cred[cred$advance_date <= CUTOFF, ], x = amount_kes), "x")
ded <- sum_by(pay[pay$paid_date <= CUTOFF, ], "credit_deduction_kes")
F$credit_balance_kes <- pmax(adv - ded, 0)
F$has_credit <- as.integer(F$credit_balance_kes > 0)

F <- merge(F, sup, by = "supplier_id")
F$no_payment_yet <- as.integer(is.na(F$price_gap_kes))
for (cc in c("price_realised_kes", "competitor_price_kes", "price_gap_kes", "payment_delay_days")) {
  med <- ave(F[[cc]], F$centre_id, FUN = function(x) median(x, na.rm = TRUE))
  F[[cc]][is.na(F[[cc]])] <- med[is.na(F[[cc]])]
}
F$tenure_months <- round(as.numeric(CUTOFF - F$registered_date) / 30.44, 1)
F$pays_bank <- as.integer(F$payment_method == "Bank")

# -------------------------------------------------------------------------------------
# 3. Value framework on the active supply base
# -------------------------------------------------------------------------------------
S <- F[F$recency_days < ACTIVE_DAYS, ]
S <- S[order(-S$litres_per_day_52w), ]
cum_before <- c(0, head(cumsum(S$litres_per_day_52w), -1)) / sum(S$litres_per_day_52w)
S$abc_class <- ifelse(cum_before < .80, "A", ifelse(cum_before < .95, "B", "C"))
S$reliable <- S$delivery_day_share_13w >= .85 & S$litres_cv_26w <= .30
S$quality_risk <- S$rejection_rate >= .03
S$fading <- S$litres_trend < .75 | S$recency_days >= 14
S$value_segment <- with(S, ifelse(fading & abc_class %in% c("A", "B"), "At-risk",
                     ifelse(quality_risk, "Quality-risk",
                     ifelse(abc_class == "A" & reliable, "Strategic",
                     ifelse(abc_class %in% c("A", "B"), "Core",
                     ifelse(abc_class == "C" & reliable & !fading, "Developing", "Low-value"))))))
VSEG <- c("Strategic", "Core", "Developing", "Quality-risk", "At-risk", "Low-value")
vs <- do.call(rbind, lapply(VSEG, function(s) { d <- S[S$value_segment == s, ]
  data.frame(segment = s, suppliers = nrow(d), litres_share = sum(d$litres_per_day_52w) / sum(S$litres_per_day_52w),
             median_litres_day = median(d$litres_per_day_52w), rejection_rate = mean(d$rejection_rate)) }))
cat("\nValue framework (R):\n"); print(transform(vs, litres_share = round(litres_share, 3), rejection_rate = round(rejection_rate, 4)), row.names = FALSE)

# -------------------------------------------------------------------------------------
# 4. K-means (k = 4) on log-transformed, standardised behaviour
# -------------------------------------------------------------------------------------
Z <- scale(cbind(log_litres_per_day = log1p(S$litres_per_day_52w), delivery_day_share = S$delivery_day_share_13w,
                 log_cv = log(S$litres_cv_26w + .05), log_trend = log(S$litres_trend),
                 fat_pct = S$fat_pct, snf_pct = S$snf_pct, log_rejection = log1p(100 * S$rejection_rate)))
set.seed(42)
km <- kmeans(Z, 4, nstart = 25, iter.max = 100)
zc <- aggregate(as.data.frame(Z), list(cl = km$cluster), mean)
nm <- character(4)
bulk <- zc$cl[which.max(zc$log_litres_per_day)]; nm[bulk] <- "Bulk suppliers"
rest <- zc[zc$cl != bulk, ]
erratic <- rest$cl[which.min(rest$delivery_day_share)]; nm[erratic] <- "Erratic / fading smallholders"
rest <- rest[rest$cl != erratic, ]
nm[rest$cl[which.max(rest$log_rejection)]] <- "Smallholders with quality issues"
nm[rest$cl[which.min(rest$log_rejection)]] <- "Steady clean smallholders"
S$cluster <- nm[km$cluster]
CSEG <- c("Bulk suppliers", "Steady clean smallholders", "Smallholders with quality issues", "Erratic / fading smallholders")
prof <- do.call(rbind, lapply(CSEG, function(s) { d <- S[S$cluster == s, ]
  data.frame(cluster = s, suppliers = nrow(d), median_litres_day = round(median(d$litres_per_day_52w), 1),
             delivery_day_share = round(median(d$delivery_day_share_13w), 3), weekly_cv = round(median(d$litres_cv_26w), 3),
             rejection_rate = round(mean(d$rejection_rate), 4),
             litres_share = sum(d$litres_per_day_52w) / sum(S$litres_per_day_52w)) }))
cat("\nK-means clusters (R):\n"); print(transform(prof, litres_share = round(litres_share, 3)), row.names = FALSE)

png(file.path(out_dir, "R_01_cluster_profiles.png"), width = 1800, height = 900, res = 170)
par(mar = c(8, 15, 3, 2))
zm <- t(sapply(CSEG, function(s) colMeans(Z[S$cluster == s, , drop = FALSE])))
image(1:ncol(zm), 1:nrow(zm), t(zm)[, nrow(zm):1], col = hcl.colors(21, "Blue-Red 3"), zlim = c(-2.4, 2.4),
      axes = FALSE, xlab = "", ylab = "", main = "K-means cluster profiles in R (standardised means)")
axis(1, 1:ncol(zm), colnames(zm), las = 2, cex.axis = .75); axis(2, 1:nrow(zm), rev(rownames(zm)), las = 1, cex.axis = .75)
for (i in 1:ncol(zm)) for (j in 1:nrow(zm)) text(i, nrow(zm) - j + 1, sprintf("%+.1f", zm[j, i]), cex = .7)
invisible(dev.off())

# -------------------------------------------------------------------------------------
# 5. Churn label: no delivery in the 63 days after a 4-week grace period
# -------------------------------------------------------------------------------------
lo <- CUTOFF + GRACE_DAYS; hi <- CUTOFF + GRACE_DAYS + SILENT_DAYS
fut <- unique(dl$supplier_id[dl$week_ending > lo & dl$week_ending <= hi])
A <- S
A$churned <- as.integer(!A$supplier_id %in% fut)
cat(sprintf("\nActive suppliers: %d | churned: %d (%.1f%%)\n", nrow(A), sum(A$churned), 100 * mean(A$churned)))

# -------------------------------------------------------------------------------------
# 6. Logistic regression - same features, same deterministic split as Python
# -------------------------------------------------------------------------------------
A$log_avg_daily_litres <- log1p(A$avg_daily_litres_13w)
A$log_litres_trend <- log(A$litres_trend)
A$type_commercial <- as.numeric(A$supplier_type == "Commercial farm")
A$type_cooperative <- as.numeric(A$supplier_type == "Farmer group/cooperative")
feats <- c("recency_days", "delivery_day_share_13w", "log_avg_daily_litres", "litres_cv_26w", "log_litres_trend",
           "rejection_rate", "price_gap_kes", "payment_delay_days", "distance_km", "advisory_visits_180d",
           "quality_training_180d", "has_credit", "group_member", "tenure_months", "pays_bank",
           "type_commercial", "type_cooperative")
test <- as.integer(substring(A$supplier_id, 2)) %% 4 == 0          # identical to Python's is_test()
X <- as.matrix(A[, feats])
mu <- colMeans(X[!test, ]); sdx <- apply(X[!test, ], 2, sd)
Xs <- sweep(sweep(X, 2, mu), 2, sdx, "/")
fit <- glm(churned ~ ., data = data.frame(churned = A$churned[!test], Xs[!test, ]), family = binomial,
           control = glm.control(epsilon = 1e-14, maxit = 100))
p_all <- predict(fit, newdata = data.frame(Xs), type = "response")
p_test <- p_all[test]; y_test <- A$churned[test]

auc <- function(y, p) {      # Mann-Whitney U formulation of ROC-AUC
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
test_auc <- auc(y_test, p_test)
co <- summary(fit)$coefficients[-1, ]
or_tab <- data.frame(feature = rownames(co), coef = co[, 1], odds_ratio = exp(co[, 1]),
                     ci_low = exp(co[, 1] - qnorm(.975) * co[, 2]), ci_high = exp(co[, 1] + qnorm(.975) * co[, 2]),
                     p_value = co[, 4], row.names = NULL)
or_tab <- or_tab[order(or_tab$odds_ratio), ]
write.csv(or_tab, file.path(out_dir, "logit_odds_ratios_R.csv"), row.names = FALSE)
cat(sprintf("Logistic regression test AUC (R): %.3f\n", test_auc))

png(file.path(out_dir, "R_02_odds_ratios.png"), width = 1500, height = 1250, res = 170)
par(mar = c(4.5, 12, 3, 2))
k <- nrow(or_tab); colr <- ifelse(or_tab$p_value < .05, ifelse(or_tab$odds_ratio > 1, RED, TEAL), GREY)
plot(or_tab$odds_ratio, 1:k, log = "x", xlim = range(c(or_tab$ci_low, or_tab$ci_high)), pch = 19, col = colr,
     yaxt = "n", ylab = "", xlab = "Odds ratio per 1 SD (95% CI, log scale)", main = "Supplier churn drivers - logistic regression (R)")
segments(or_tab$ci_low, 1:k, or_tab$ci_high, 1:k, col = colr, lwd = 2); abline(v = 1, lty = 2)
axis(2, 1:k, or_tab$feature, las = 1, cex.axis = .7)
invisible(dev.off())

# -------------------------------------------------------------------------------------
# 7. Gains by risk decile (test set) and dry-season litres at risk
# -------------------------------------------------------------------------------------
dec <- ceiling(10 * rank(-p_test, ties.method = "first") / length(p_test))
gains <- data.frame(decile = 1:10, churners = as.numeric(tapply(y_test, dec, sum)), suppliers = as.numeric(table(dec)))
gains$cum_capture <- cumsum(gains$churners) / sum(gains$churners)
gains$lift <- (gains$churners / gains$suppliers) / mean(y_test)
png(file.path(out_dir, "R_03_gains.png"), width = 1500, height = 900, res = 170)
par(mar = c(4.5, 4.5, 3, 4.5))
bp <- barplot(gains$lift, names.arg = gains$decile, col = TEAL, border = NA, ylab = "Lift",
              xlab = "Risk decile (1 = highest risk)", main = "Lift and cumulative capture by decile (R, test set)")
par(new = TRUE); plot(bp, 100 * gains$cum_capture, type = "o", col = AMBER, pch = 19, axes = FALSE, xlab = "", ylab = "", ylim = c(0, 105))
axis(4, col.axis = AMBER); mtext("Cumulative % of churners captured", 4, 3, col = AMBER)
invisible(dev.off())

y25 <- dl[dl$week_ending > as.Date("2025-06-29") & dl$week_ending <= as.Date("2025-09-28"), ]
b25 <- dl[dl$week_ending > as.Date("2025-05-04") & dl$week_ending <= as.Date("2025-06-29"), ]
dry_factor <- mean(y25$litres_delivered) / mean(b25$litres_delivered)
m8 <- dl[dl$week_ending > CUTOFF - 56 & dl$week_ending <= CUTOFF, ]
m8s <- tapply(m8$litres_delivered, m8$supplier_id, sum) / 8
A$exp_litres_13w <- ifelse(is.na(m8s[A$supplier_id]), 0, m8s[A$supplier_id]) * 13 * dry_factor
A$litres_at_risk <- p_all * A$exp_litres_13w
risk <- do.call(rbind, lapply(VSEG, function(s) { d <- A[A$value_segment == s, ]
  data.frame(segment = s, suppliers = nrow(d), actual_churn = sprintf("%.1f%%", 100 * mean(d$churned)),
             litres_at_risk = round(sum(d$litres_at_risk)), share = sprintf("%.1f%%", 100 * sum(d$litres_at_risk) / sum(A$litres_at_risk))) }))

# -------------------------------------------------------------------------------------
# 8. Cross-check with Python and write the report
# -------------------------------------------------------------------------------------
check <- "Python outputs not found - run the notebook first to enable the cross-check."
py_or <- file.path("outputs", "python", "logit_odds_ratios.csv")
py_ft <- file.path("outputs", "python", "supplier_features.csv")
py_sg <- file.path("outputs", "python", "supplier_segments.csv")
feat_tab <- NULL
if (file.exists(py_or) && file.exists(py_ft) && file.exists(py_sg)) {
  py <- read.csv(py_or, stringsAsFactors = FALSE); names(py)[1] <- "feature"
  m <- merge(py, or_tab, by = "feature", suffixes = c("_py", "_r"))
  max_coef_diff <- max(abs(m$coef_py - m$coef_r))
  max_or_diff <- max(abs(m$odds_ratio_py - m$odds_ratio_r) / m$odds_ratio_py)
  pf_ <- read.csv(py_ft, stringsAsFactors = FALSE)
  pf_ <- pf_[match(A$supplier_id, pf_$supplier_id), ]
  cmp <- c(feats, "litres_per_day_52w", "fat_pct", "snf_pct", "litres_trend")
  feat_tab <- data.frame(feature = cmp, max_abs_diff = sapply(cmp, function(cn) max(abs(pf_[[cn]] - A[[cn]]))), row.names = NULL)
  label_match <- mean(pf_$churned == A$churned)
  ps <- read.csv(py_sg, stringsAsFactors = FALSE)
  ps <- ps[match(S$supplier_id, ps$supplier_id), ]
  vs_match <- mean(ps$value_segment == S$value_segment); cl_match <- mean(ps$cluster == S$cluster)
  check <- c(sprintf("Suppliers matched: %d of %d active suppliers; churn labels identical for %.2f%%", sum(!is.na(pf_$supplier_id)), nrow(A), 100 * label_match),
             sprintf("Features rebuilt from raw data: largest absolute difference across %d features = %.1e (rounding of the exported CSV is 1e-8)", length(cmp), max(feat_tab$max_abs_diff)),
             sprintf("Logistic regression: %d coefficients compared, max |coef difference| = %.1e, max relative odds-ratio difference = %.1e", nrow(m), max_coef_diff, max_or_diff),
             sprintf("Value-framework segment agreement: %.2f%% | K-means cluster agreement: %.2f%%", 100 * vs_match, 100 * cl_match))
}
cat("\nCROSS-CHECK:\n", paste("-", check, collapse = "\n"), "\n")

md_table <- function(df) c(paste("|", paste(names(df), collapse = " | "), "|"),
                           paste("|", paste(rep("---", ncol(df)), collapse = " | "), "|"),
                           apply(df, 1, function(r) paste("|", paste(trimws(r), collapse = " | "), "|")))
report <- c(
  "# Supplier Segmentation & Churn — R results (auto-generated by `R/supplier_segmentation_churn.R`)", "",
  sprintf("*Snapshot %s · %d active suppliers segmented and modelled · churn = no delivery from day %d to day %d after the snapshot*",
          CUTOFF, nrow(A), GRACE_DAYS + 1, GRACE_DAYS + SILENT_DAYS), "",
  "## Cross-check with the Python notebook", "", paste("-", check), "",
  "R's `kmeans()` (Hartigan-Wong, 25 random starts) and scikit-learn (Lloyd, k-means++) use different algorithms and starting points, so a few borderline suppliers can land in a neighbouring cluster; the cluster *profiles* are the same. All other quantities are deterministic and are expected to match to floating-point precision.", "",
  "## Value framework", "",
  md_table(data.frame(segment = vs$segment, suppliers = vs$suppliers, litres_share = sprintf("%.1f%%", 100 * vs$litres_share),
                      median_litres_day = sprintf("%.1f", vs$median_litres_day), rejection_rate = sprintf("%.2f%%", 100 * vs$rejection_rate))), "",
  "## K-means clusters (k = 4)", "",
  md_table(transform(prof, litres_share = sprintf("%.1f%%", 100 * litres_share), rejection_rate = sprintf("%.2f%%", 100 * rejection_rate))), "",
  "![Cluster profiles](R_01_cluster_profiles.png)", "",
  "## Churn model (logistic regression, glm)", "",
  sprintf("- Active suppliers: %d, churn rate %.1f%% · train %d / test %d · test AUC **%.3f**", nrow(A), 100 * mean(A$churned), sum(!test), sum(test), test_auc), "",
  md_table(data.frame(feature = or_tab$feature, odds_ratio = sprintf("%.3f", or_tab$odds_ratio),
                      `95% CI` = sprintf("%.2f – %.2f", or_tab$ci_low, or_tab$ci_high),
                      p_value = sprintf("%.3f", or_tab$p_value), check.names = FALSE)), "",
  "![Odds ratios](R_02_odds_ratios.png)", "",
  "## Gains (test set)", "",
  md_table(data.frame(decile = gains$decile, churners = gains$churners, suppliers = gains$suppliers,
                      cumulative_capture = sprintf("%.0f%%", 100 * gains$cum_capture), lift = sprintf("%.2f", gains$lift))), "",
  "![Gains](R_03_gains.png)", "",
  "## Dry-season litres at risk (next 13 weeks)", "",
  sprintf("Dry-season factor %.3f · expected litres %.2fm · litres at risk %.2fm (%.1f%%)", dry_factor,
          sum(A$exp_litres_13w) / 1e6, sum(A$litres_at_risk) / 1e6, 100 * sum(A$litres_at_risk) / sum(A$exp_litres_13w)), "",
  md_table(risk))
if (!is.null(feat_tab)) report <- c(report, "", "## Feature-level comparison with Python", "",
                                    md_table(transform(feat_tab, max_abs_diff = sprintf("%.1e", max_abs_diff))))
writeLines(report, file.path(out_dir, "R_results.md"))
write.csv(S[, c("supplier_id", "supplier_type", "abc_class", "value_segment", "cluster")],
          file.path(out_dir, "supplier_segments_R.csv"), row.names = FALSE)
cat("Wrote outputs/r/R_results.md, CSVs and charts\n")
