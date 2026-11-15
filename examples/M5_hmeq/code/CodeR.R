# ============================================================================
# PROJET HMEQ - PARTIE R
# Validation et approfondissements des analyses SAS
# Version adaptée au code SAS avec imputation multiple PMM
# ============================================================================

# ============================================================================
# BLOC 1 - CHARGEMENT DES BIBLIOTHÈQUES
# ============================================================================

# Créer une liste de toutes les bibliothèques
libraries <- c("tidyverse", "corrplot", "psych", "pROC", "ResourceSelection", 
               "InformationValue", "car", "gridExtra", "knitr", "broom")

# Vérifier si chaque bibliothèque est installée, sinon l'installer
install_if_missing <- function(pkg){
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# Appliquer la fonction à chaque bibliothèque
lapply(libraries, install_if_missing)

install.packages("conflicted")
library(conflicted)

conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")



# ============================================================================
# BLOC 2 - IMPORTATION DES DONNÉES DEPUIS SAS
# ============================================================================

cat("========================================\n")
cat("PROJET HMEQ - PARTIE R\n")
cat("Validation des analyses SAS avec imputation PMM\n")
cat("========================================\n\n")

setwd("/Users/konearounaromeo/ScoringProjet")

# Import des fichiers générés par SAS
cat("1. CHARGEMENT DES DONNÉES SAS\n")
cat("--------------------------------\n")

# Données imputées par PMM
hmeq_imp <- read.csv("donnees_imputees_PMM.csv")
cat("✓ Données imputées PMM :", nrow(hmeq_imp), "observations\n")
cat("✓ Nombres de données manquantes:", colSums(is.na(hmeq_imp)))
View(hmeq_imp)

# Scores et prédictions
scores <- read.csv("scores_PMM.csv")
cat("✓ Scores et prédictions :", nrow(scores), "observations\n")
View(scores)

# Information Value calculée par SAS
iv_sas <- read.csv("information_value_PMM.csv")
cat("✓ Information Value SAS :", nrow(iv_sas), "variables\n\n")
View(iv_sas)

# ============================================================================
# BLOC 3 - VALIDATION CROISÉE DES TESTS STATISTIQUES SAS
# ============================================================================

cat("2. VALIDATION CROISÉE DES TESTS SAS\n")
cat("--------------------------------\n")

# 3.1 Tests t de Student
vars_continues <- c("DEBTINC", "YOJ", "CLAGE", "DELINQ", "DEROG", 
                    "NINQ", "LTV", "LOAN", "MORTDUE", "VALUE")

validation_tests <- data.frame()

for(var in vars_continues) {
  # Test t
  t_test <- t.test(hmeq_imp[[var]] ~ hmeq_imp$BAD)
  
  # Test de Mann-Whitney
  mw_test <- wilcox.test(hmeq_imp[[var]] ~ hmeq_imp$BAD)
  
  validation_tests <- rbind(validation_tests, data.frame(
    Variable = var,
    t_stat = round(t_test$statistic, 3),
    p_value_t = round(t_test$p.value, 4),
    p_value_mw = round(mw_test$p.value, 4),
    significatif_t = ifelse(t_test$p.value < 0.05, "OUI", "NON"),
    significatif_mw = ifelse(mw_test$p.value < 0.05, "OUI", "NON"),
    conclusion = ifelse(t_test$p.value < 0.05 & mw_test$p.value < 0.05, 
                        "ROBUSTE", "À VÉRIFIER")
  ))
}

cat("✓ Tests t et Mann-Whitney exécutés\n")

# sélection correcte des colonnes
print(validation_tests[, c("Variable", "p_value_t", "p_value_mw", "conclusion")])

# 3.2 Tests du Chi2 pour variables catégorielles
cat("\nTests Chi2 - Validation SAS:\n")

chi2_reason <- chisq.test(table(hmeq_imp$REASON, hmeq_imp$BAD))
chi2_job <- chisq.test(table(hmeq_imp$JOB, hmeq_imp$BAD))
chi2_ltv_cat <- chisq.test(table(hmeq_imp$LTV_CAT, hmeq_imp$BAD))

cat("  REASON : p =", round(chi2_reason$p.value, 4), 
    "-", ifelse(chi2_reason$p.value < 0.05, "Significatif", "Non significatif"), "\n")
cat("  JOB    : p =", round(chi2_job$p.value, 4),
    "-", ifelse(chi2_job$p.value < 0.05, "Significatif", "Non significatif"), "\n")
cat("  LTV_CAT: p =", round(chi2_ltv_cat$p.value, 4),
    "-", ifelse(chi2_ltv_cat$p.value < 0.05, "Significatif", "Non significatif"), "\n")


# 8. Analyse de corrélation (sans les colonnes commençant par sqrt ou log)
cor_matrix <- cor(
  hmeq_imp %>% 
    select(where(is.numeric)) %>% 
    select(-starts_with("SQRT"), -starts_with("LOG")),
  use = "complete.obs"
)

print(cor_matrix)

# Visualisation
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,
         title = "Matrice de corrélation")

# 10. Régression logistique avancée (SAS-like)

# Vérifier et fixer les références des variables catégorielles
# On choisit le premier niveau existant comme référence pour éviter les erreurs
hmeq_imp$REASON <- factor(hmeq_imp$REASON)
hmeq_imp$REASON <- relevel(hmeq_imp$REASON, ref = levels(hmeq_imp$REASON)[1])

hmeq_imp$JOB <- factor(hmeq_imp$JOB)
hmeq_imp$JOB <- relevel(hmeq_imp$JOB, ref = levels(hmeq_imp$JOB)[1])

# Modèle logistique
logit_model <- glm(BAD ~ LOAN + DEBTINC + CLAGE + DEROG + DELINQ + 
                     REASON + JOB,
                   data = hmeq_imp, family = binomial(link = "logit"))

# Résumé des coefficients (comme SAS)
summary(logit_model)

# Odds Ratios avec intervalles de confiance à 95%
odds_ratios <- exp(coef(logit_model))
conf_int <- exp(confint(logit_model))  # IC à 95%
odds_table <- data.frame(
  Variable = names(odds_ratios),
  Odds_Ratio = odds_ratios,
  CI_lower = conf_int[,1],
  CI_upper = conf_int[,2]
)

print(odds_table)

levels(hmeq_imp$JOB)
# ============================================================================
# BLOC 4 - INFORMATION VALUE COMPLÈTE (TOUTES VARIABLES)
# ============================================================================

cat("\n3. INFORMATION VALUE - ANALYSE COMPLÈTE\n")
cat("--------------------------------\n")

# Fonction de calcul IV
calc_iv <- function(data, var_name, target = "BAD") {
  df <- data[!is.na(data[[var_name]]), ]
  
  # Découpage en 10 classes
  df$bin <- cut(df[[var_name]], 
                breaks = unique(quantile(df[[var_name]], 
                                         probs = seq(0, 1, 0.1), 
                                         na.rm = TRUE)),
                include.lowest = TRUE,
                labels = FALSE)
  
  # Table de contingence
  tbl <- table(df$bin, df[[target]])
  
  # Calcul IV
  bons <- tbl[,1]
  mauvais <- tbl[,2]
  tot_bons <- sum(bons)
  tot_mauvais <- sum(mauvais)
  
  wi <- (bons/tot_bons - mauvais/tot_mauvais)
  iv <- sum(wi * log((bons/tot_bons)/(mauvais/tot_mauvais)), na.rm = TRUE)
  
  return(iv)
}

# Liste des variables pour IV
vars_iv <- c("DEBTINC", "CLAGE", "YOJ", "LOAN", "LTV", 
             "DELINQ", "DEROG", "NINQ", "MORTDUE", "VALUE")

iv_complet <- data.frame(
  Variable = vars_iv,
  IV = sapply(vars_iv, function(v) calc_iv(hmeq_imp, v)),
  IV_SAS = iv_sas$IV[match(vars_iv, iv_sas$Variable)]
)

iv_complet$Pouvoir_Predictif <- cut(iv_complet$IV,
                                    breaks = c(-Inf, 0.02, 0.1, 0.3, Inf),
                                    labels = c("Faible", "Moyen", "Fort", "Très fort"))

iv_complet$Ecart_SAS_R <- round(iv_complet$IV - iv_complet$IV_SAS, 4)

cat("Information Value - Comparaison SAS vs R:\n")
print(iv_complet %>% 
        arrange(desc(IV)) %>%
        select(Variable, IV, Pouvoir_Predictif, IV_SAS, Ecart_SAS_R))

# ============================================================================
# BLOC 5 - COURBE ROC ET AUC (MODÈLE LOGISTIQUE SAS)
# ============================================================================

cat("\n4. COURBE ROC ET AUC\n")
cat("--------------------------------\n")

# Utilisation des probabilités du modèle SAS
roc_obj <- roc(scores$BAD, scores$prob)
auc_value <- auc(roc_obj)
ci_auc <- ci.auc(roc_obj)

cat("AUC (modèle logistique SAS avec PMM):\n")
cat("  AUC =", round(auc_value, 4), "\n")
cat("  IC 95% : [", round(ci_auc[1], 4), "-", round(ci_auc[3], 4), "]\n")
cat("  Interprétation :", 
    ifelse(auc_value >= 0.9, "Exceptionnel",
           ifelse(auc_value >= 0.8, "Excellent",
                  ifelse(auc_value >= 0.7, "Acceptable",
                         ifelse(auc_value >= 0.6, "Faible", "Insuffisant")))), "\n")

# Sauvegarde du graphique ROC
png("graph_roc_logit_R_PMM.png", width = 800, height = 600)
plot(roc_obj, 
     main = paste("Courbe ROC - Modèle Logistique SAS (PMM)\nAUC =", 
                  round(auc_value, 3)),
     col = "blue", lwd = 2, 
     legacy.axes = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", 
       legend = c(paste("Modèle (AUC =", round(auc_value, 3), ")"), "Aléatoire"),
       col = c("blue", "gray"), lwd = c(2, 1), lty = c(1, 2))
dev.off()
cat("✓ Graphique ROC sauvegardé: graph_roc_logit_R_PMM.png\n")


#library(pROC)

# Probabilités et BAD
roc_obj <- roc(scores$BAD, scores$prob)

# AUC et IC
auc_value <- auc(roc_obj)
ci_auc <- ci.auc(roc_obj)

cat("AUC (modèle logistique SAS avec PMM):\n")
cat("  AUC =", round(auc_value, 4), "\n")
cat("  IC 95% : [", round(ci_auc[1], 4), "-", round(ci_auc[3], 4), "]\n")
cat("  Interprétation :",
    ifelse(auc_value >= 0.9, "Exceptionnel",
           ifelse(auc_value >= 0.8, "Excellent",
                  ifelse(auc_value >= 0.7, "Acceptable",
                         ifelse(auc_value >= 0.6, "Faible", "Insuffisant")))), "\n")

# ===========================
# 1. Affichage à l'écran sur Mac
# ===========================
quartz()  # ouvre une nouvelle fenêtre graphique sur Mac
plot(roc_obj,
     main = paste("Courbe ROC - Modèle Logistique SAS (PMM)\nAUC =", round(auc_value, 3)),
     col = "blue", lwd = 2,
     legacy.axes = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(paste("Modèle (AUC =", round(auc_value, 3), ")"), "Aléatoire"),
       col = c("blue", "gray"), lwd = c(2, 1), lty = c(1, 2))

# ===========================
# 2. Sauvegarde dans PNG (pas d'affichage dans RStudio)
# ===========================
png("graph_roc_logit_R_PMM.png", width = 1200, height = 800)
plot(roc_obj,
     main = paste("Courbe ROC - Modèle Logistique SAS (PMM)\nAUC =", round(auc_value, 3)),
     col = "blue", lwd = 2,
     legacy.axes = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(paste("Modèle (AUC =", round(auc_value, 3), ")"), "Aléatoire"),
       col = c("blue", "gray"), lwd = c(2, 1), lty = c(1, 2))
dev.off()

cat("✓ Graphique ROC affiché et sauvegardé: graph_roc_logit_R_PMM.png\n")





# ============================================================================
# BLOC 6 - TEST DE HOSMER-LEMESHOW (CALIBRAGE)
# ============================================================================

cat("\n5. TEST DE HOSMER-LEMESHOW\n")
cat("--------------------------------\n")

hl_test <- hoslem.test(scores$BAD, scores$prob, g = 10)
cat("Test de Hosmer-Lemeshow (g=10):\n")
cat("  Statistique H-L :", round(hl_test$statistic, 3), "\n")
cat("  Degrés de liberté :", hl_test$parameter, "\n")
cat("  p-value :", round(hl_test$p.value, 4), "\n")
cat("  Interprétation :", 
    ifelse(hl_test$p.value > 0.05, 
           "Bon calibrage (p > 0.05)", 
           "Mauvais calibrage (p < 0.05)"), "\n")

# Tableau observés/attendus
hl_tableau <- cbind(hl_test$observed, hl_test$expected)
colnames(hl_tableau) <- c("Obs_0", "Obs_1", "Exp_0", "Exp_1")
cat("\nTableau des effectifs observés/attendus par décile:\n")
print(round(hl_tableau, 2))

# ============================================================================
# BLOC 7 - OPTIMISATION DU SEUIL DE DÉCISION
# ============================================================================

cat("\n6. OPTIMISATION DU SEUIL DE DÉCISION\n")
cat("--------------------------------\n")

# Recherche du seuil optimal
seuils <- seq(0.1, 0.9, 0.01)
performance <- data.frame()

for(s in seuils) {
  pred_class <- ifelse(scores$prob > s, 1, 0)
  
  cm <- table(Predicted = pred_class, Actual = scores$BAD)
  
  if(nrow(cm) == 2 & ncol(cm) == 2) {
    tn <- cm[1,1]; fp <- cm[2,1]
    fn <- cm[1,2]; tp <- cm[2,2]
    
    accuracy <- (tn + tp) / sum(cm)
    precision <- tp / (tp + fp)
    recall <- tp / (tp + fn)
    f1_score <- 2 * (precision * recall) / (precision + recall)
    specificity <- tn / (tn + fp)
    
    performance <- rbind(performance, data.frame(
      seuil = s,
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      f1_score = f1_score,
      specificity = specificity
    ))
  }
}

# Seuil optimal selon F1-score
optimal_idx <- which.max(performance$f1_score)
seuil_optimal <- performance$seuil[optimal_idx]
f1_optimal <- performance$f1_score[optimal_idx]

cat("Seuil par défaut (0.50) :\n")
cat("  F1-score =", round(performance$f1_score[performance$seuil == 0.5], 4), "\n")
cat("  Recall   =", round(performance$recall[performance$seuil == 0.5], 4), "\n")
cat("  Précision=", round(performance$precision[performance$seuil == 0.5], 4), "\n\n")

cat("Seuil OPTIMAL (max F1) :\n")
cat("  Seuil =", round(seuil_optimal, 3), "\n")
cat("  F1-score =", round(f1_optimal, 4), "\n")
cat("  Recall   =", round(performance$recall[optimal_idx], 4), 
    "(+", round((performance$recall[optimal_idx] - performance$recall[performance$seuil == 0.5])*100, 1), "pts)\n")
cat("  Précision=", round(performance$precision[optimal_idx], 4),
    "(", round((performance$precision[optimal_idx] - performance$precision[performance$seuil == 0.5])*100, 1), "pts)\n")

# Graphique d'évolution des métriques
png("graph_evolution_seuil_R_PMM.png", width = 800, height = 600)
plot(performance$seuil, performance$f1_score, type = "l", col = "blue", lwd = 2,
     xlab = "Seuil de décision", ylab = "Score",
     main = "Optimisation du seuil de décision - Modèle PMM",
     ylim = c(0, 1))
lines(performance$seuil, performance$recall, col = "green", lwd = 2, lty = 2)
lines(performance$seuil, performance$precision, col = "red", lwd = 2, lty = 2)
lines(performance$seuil, performance$specificity, col = "orange", lwd = 2, lty = 2)
abline(v = seuil_optimal, col = "purple", lty = 3, lwd = 2)
abline(v = 0.5, col = "gray", lty = 3, lwd = 1.5)
legend("bottomleft", 
       legend = c("F1-score", "Recall", "Précision", "Spécificité", 
                  paste("Seuil optimal =", round(seuil_optimal, 2)), "Seuil 0.5"),
       col = c("blue", "green", "red", "orange", "purple", "gray"),
       lwd = c(2, 2, 2, 2, 2, 1.5),
       lty = c(1, 2, 2, 2, 3, 3))
grid()
dev.off()
cat("✓ Graphique optimisation seuil: graph_evolution_seuil_R_PMM.png\n")

# ============================================================================
# BLOC 8 - VALIDATION DU SCORE SAS
# ============================================================================

cat("\n7. VALIDATION DU SCORE SAS\n")
cat("--------------------------------\n")

# Calcul du score R (même formule)
scores$score_R <- 600 - (20/log(2)) * log(scores$prob/(1-scores$prob))

# Corrélation scores SAS vs R
cor_test <- cor.test(scores$score, scores$score_R)
cat("Corrélation scores SAS vs R :\n")
cat("  r =", round(cor_test$estimate, 4), "\n")
cat("  IC 95% : [", round(cor_test$conf.int[1], 4), 
    "-", round(cor_test$conf.int[2], 4), "]\n")
cat("  p-value :", round(cor_test$p.value, 4), "\n")
cat("  Cohérence :", ifelse(cor_test$estimate > 0.99, 
                            "PARFAITE", "À VÉRIFIER"), "\n")

# Statistiques du score par statut BAD
score_stats <- scores %>%
  group_by(BAD) %>%
  summarise(
    N = n(),
    Score_moyen = mean(score, na.rm = TRUE),
    Score_median = median(score, na.rm = TRUE),
    Score_sd = sd(score, na.rm = TRUE),
    Score_min = min(score, na.rm = TRUE),
    Score_max = max(score, na.rm = TRUE)
  )

cat("\nStatistiques du score par statut BAD:\n")
print(score_stats)

# Boxplot du score par BAD
png("graph_score_par_BAD_R_validation.png", width = 800, height = 600)
boxplot(score ~ BAD, data = scores,
        names = c("Non défaillant (0)", "Défaillant (1)"),
        col = c("steelblue", "coral"),
        main = "Distribution du score selon le statut de défaut - Validation R",
        ylab = "Score de crédit",
        xlab = "Statut BAD",
        outline = FALSE,
        notch = TRUE)
grid()
legend("topright", 
       fill = c("steelblue", "coral"),
       legend = c(paste("Non défaillant (n=", sum(scores$BAD==0), ")"),
                  paste("Défaillant (n=", sum(scores$BAD==1), ")")))
dev.off()
cat("✓ Boxplot score: graph_score_par_BAD_R_validation.png\n")

# ============================================================================
# BLOC 9 - ANALYSE DES DÉCILES DE SCORE
# ============================================================================

cat("\n8. ANALYSE DES DÉCILES DE SCORE\n")
cat("--------------------------------\n")

# Création des déciles
scores$decile <- ntile(scores$score, 10)

# Statistiques par décile
decile_stats <- scores %>%
  group_by(decile) %>%
  summarise(
    N = n(),
    N_defauts = sum(BAD),
    Taux_defaut = mean(BAD),
    Score_min = min(score),
    Score_max = max(score),
    Score_moyen = mean(score)
  ) %>%
  arrange(decile)

cat("Performance par décile de score:\n")
print(decile_stats %>% 
        select(decile, N, N_defauts, Taux_defaut, Score_moyen) %>%
        mutate(Taux_defaut = round(Taux_defaut, 4)))

# Indice de Gini
scores <- scores %>% arrange(desc(prob))
scores$cum_defauts <- cumsum(scores$BAD) / sum(scores$BAD)
scores$cum_pop <- (1:nrow(scores)) / nrow(scores)

# Aire sous courbe de Lorenz
library(pracma)
gini <- 2 * trapz(scores$cum_pop, scores$cum_defauts) - 1
cat("\nIndice de Gini:", round(gini, 4), "\n")
cat("Équivalent AUC:", round((gini + 1)/2, 4), "\n")


# --- EXPORT DES RÉSULTATS DANS UNE TABLE ---
# Compilation des indicateurs globaux
synthèse_metrics <- data.frame(
  Indicateur = c("Corrélation SAS-R", "Indice de Gini", "AUC (ROC Equivalent)"),
  Valeur = c(round(cor_test$estimate, 4), round(gini, 4), round((gini + 1)/2, 4))
)

# Export des tables
write.csv(decile_stats, "stats_deciles_score.csv", row.names = FALSE)
write.csv(synthèse_metrics, "metrics_performance_globale.csv", row.names = FALSE)
cat("✓ Tables exportées : stats_deciles_score.csv et metrics_performance_globale.csv\n")

# --- VISUALISATION DU GINI ET DE LA COURBE DE LORENZ ---
png("graph_lorenz_gini_validation.png", width = 800, height = 600)

# Tracer la courbe de Lorenz (Concentration des défauts)
plot(scores$cum_pop, scores$cum_defauts, type = "l", col = "darkblue", lwd = 3,
     main = "Courbe de Lorenz & Indice de Gini",
     xlab = "% Cumulé de la population (classée par risque)",
     ylab = "% Cumulé des défauts (BAD)")

# Ajouter la diagonale du hasard (Gini = 0)
abline(0, 1, col = "red", lty = 2, lwd = 2)

# Colorer l'aire entre la diagonale et la courbe (Aire de Gini)
polygon(c(scores$cum_pop, rev(scores$cum_pop)), 
        c(scores$cum_defauts, rev(scores$cum_pop)), 
        col = rgb(0, 0, 1, 0.1), border = NA)

# Ajouter les textes informatifs
text(0.7, 0.3, paste("Gini =", round(gini, 4)), cex = 1.2, font = 2)
text(0.7, 0.2, paste("AUC =", round((gini + 1)/2, 4)), cex = 1.2, font = 2)

grid()
legend("topleft", legend = c("Modèle PMM", "Hasard (Aléatoire)"), 
       col = c("darkblue", "red"), lty = c(1, 2), lwd = 2)

dev.off()
cat("✓ Graphique Gini/Lorenz : graph_lorenz_gini_validation.png\n")

# ============================================================================
# BLOC 10 - COMPARAISON DES MÉTHODES D'IMPUTATION
# ============================================================================

cat("\n9. COMPARAISON AVEC LES STATISTIQUES SAS\n")
cat("--------------------------------\n")

# Import des statistiques avant imputation (à générer par SAS)
# Note: Ces données doivent être exportées par SAS

cat("\nSynthèse de la validation:\n")
cat("  ✓ Tests statistiques :", 
    sum(validation_tests$conclusion == "ROBUSTE"), 
    "/", nrow(validation_tests), "variables robustes\n")
cat("  ✓ Information Value : corrélation SAS/R =", 
    round(cor(iv_complet$IV, iv_complet$IV_SAS, use = "complete.obs"), 4), "\n")
cat("  ✓ Performance modèle : AUC =", round(auc_value, 4), "\n")
cat("  ✓ Calibrage : p-value H-L =", round(hl_test$p.value, 4), "\n")
cat("  ✓ Seuil optimal :", round(seuil_optimal, 3), 
    "(vs 0.5 par défaut)\n")
cat("  ✓ Cohérence scores : r =", round(cor_test$estimate, 4), "\n")

# ============================================================================
# BLOC 11 - EXPORT DES RÉSULTATS R
# ============================================================================

cat("\n10. EXPORT DES RÉSULTATS\n")
cat("--------------------------------\n")

# Sauvegarde des tableaux
write.csv(validation_tests, "validation_tests_R_PMM.csv", row.names = FALSE)
write.csv(iv_complet, "information_value_R_PMM.csv", row.names = FALSE)
write.csv(performance, "optimisation_seuil_R_PMM.csv", row.names = FALSE)
write.csv(decile_stats, "deciles_score_R_PMM.csv", row.names = FALSE)

cat("✓ Tableaux exportés:\n")
cat("  - validation_tests_R_PMM.csv\n")
cat("  - information_value_R_PMM.csv\n")
cat("  - optimisation_seuil_R_PMM.csv\n")
cat("  - deciles_score_R_PMM.csv\n")

# Sauvegarde des objets R pour analyse ultérieure
saveRDS(roc_obj, "roc_object_R_PMM.rds")
saveRDS(performance, "performance_seuils_R_PMM.rds")

cat("✓ Objets R sauvegardés: roc_object_R_PMM.rds, performance_seuils_R_PMM.rds\n")

# ============================================================================
# BLOC 12 - RAPPORT DE SYNTHÈSE R
# ============================================================================

cat("\n11. RAPPORT DE SYNTHÈSE R\n")
cat("--------------------------------\n")

sink("rapport_validation_R_PMM.txt")

cat("================================================================\n")
cat("RAPPORT DE VALIDATION R - PROJET HMEQ\n")
cat("Imputation multiple PMM (Predictive Mean Matching)\n")
cat("================================================================\n\n")
cat("Date :", date(), "\n\n")

cat("1. VALIDATION DES TESTS STATISTIQUES SAS\n")
cat("----------------------------------------\n")
cat("✓ Convergence des tests t et Mann-Whitney :\n")
print(validation_tests %>% 
        select(Variable, p_value_t, p_value_mw) %>%
        mutate(across(starts_with("p_"), ~round(., 4))))
cat("\n✓ Tests Chi2 catégoriels validés\n\n")

cat("2. INFORMATION VALUE\n")
cat("-------------------\n")
print(iv_complet %>% 
        arrange(desc(IV)) %>%
        select(Variable, IV, Pouvoir_Predictif) %>%
        mutate(IV = round(IV, 4)))
cat("\nCorrélation IV SAS/R :", 
    round(cor(iv_complet$IV, iv_complet$IV_SAS, use = "complete.obs"), 4), "\n\n")

cat("3. PERFORMANCE DU MODÈLE LOGISTIQUE SAS\n")
cat("---------------------------------------\n")
cat("AUC :", round(auc_value, 4), "\n")
cat("IC 95% : [", round(ci_auc[1], 4), "-", round(ci_auc[3], 4), "]\n")
cat("Test de Hosmer-Lemeshow : p =", round(hl_test$p.value, 4), "\n")
cat("Interprétation :", ifelse(hl_test$p.value > 0.05, 
                               "Bon calibrage", "Mauvais calibrage"), "\n\n")

cat("4. OPTIMISATION OPÉRATIONNELLE\n")
cat("-------------------------------\n")
cat("Seuil par défaut (0.5) :\n")
cat("  - F1-score :", round(performance$f1_score[performance$seuil == 0.5], 4), "\n")
cat("  - Recall   :", round(performance$recall[performance$seuil == 0.5], 4), "\n")
cat("  - Précision:", round(performance$precision[performance$seuil == 0.5], 4), "\n\n")
cat("Seuil OPTIMAL :\n")
cat("  - Seuil    :", round(seuil_optimal, 3), "\n")
cat("  - F1-score :", round(f1_optimal, 4), "\n")
cat("  - Recall   :", round(performance$recall[optimal_idx], 4), "\n")
cat("  - Précision:", round(performance$precision[optimal_idx], 4), "\n")
cat("  - Gain Recall : +", 
    round((performance$recall[optimal_idx] - performance$recall[performance$seuil == 0.5])*100, 1), 
    "points\n\n")

cat("5. VALIDATION DU SCORE\n")
cat("----------------------\n")
cat("Corrélation scores SAS/R : r =", round(cor_test$estimate, 4), "\n")
cat("Score moyen - Bons :", round(score_stats$Score_moyen[1], 1), "\n")
cat("Score moyen - Défaillants :", round(score_stats$Score_moyen[2], 1), "\n")
cat("Écart de score :", 
    round(score_stats$Score_moyen[1] - score_stats$Score_moyen[2], 1), "points\n\n")

cat("6. SYNTHÈSE DES APPARTS R\n")
cat("------------------------\n")
cat("✓ Validation indépendante des résultats SAS\n")
cat("✓ Calcul de l'AUC et IC à 95% (non disponible dans SAS)\n")
cat("✓ Test de Hosmer-Lemeshow pour le calibrage\n")
cat("✓ Optimisation du seuil de décision (gain Recall : +", 
    round((performance$recall[optimal_idx] - performance$recall[performance$seuil == 0.5])*100, 1), "pts)\n")
cat("✓ Information Value sur toutes les variables\n")
cat("✓ Analyse des déciles et indice de Gini\n")
cat("✓ Cohérence scores SAS/R validée (r =", round(cor_test$estimate, 3), ")\n")

cat("\n================================================================\n")
cat("FIN DU RAPPORT DE VALIDATION R\n")
cat("================================================================\n")

sink()

cat("✓ Rapport de synthèse: rapport_validation_R_PMM.txt\n")

# ============================================================================
# BLOC 13 - MESSAGE DE FIN
# ============================================================================

cat("\n========================================\n")
cat("VALIDATION R TERMINÉE AVEC SUCCÈS\n")
cat("========================================\n")
cat("Fichiers générés dans le dossier Resultats :\n")
cat("  - Graphiques :\n")
cat("    * graph_roc_logit_R_PMM.png\n")
cat("    * graph_evolution_seuil_R_PMM.png\n")
cat("    * graph_score_par_BAD_R_validation.png\n")
cat("  - Tableaux :\n")
cat("    * validation_tests_R_PMM.csv\n")
cat("    * information_value_R_PMM.csv\n")
cat("    * optimisation_seuil_R_PMM.csv\n")
cat("    * deciles_score_R_PMM.csv\n")
cat("  - Rapports :\n")
cat("    * rapport_validation_R_PMM.txt\n")
cat("  - Objets R :\n")
cat("    * roc_object_R_PMM.rds\n")
cat("    * performance_seuils_R_PMM.rds\n")
cat("========================================\n")

# ============================================================================
# FIN DU CODE R
# ============================================================================


# Charger le fichier dans un objet nommé 'perf_data'
perf_data <- readRDS("performance_seuils_R_PMM.rds")

# Visualiser le contenu
print(perf_data)
# Ou pour voir la structure complexe
str(perf_data)

# Charger l'objet ROC
roc_obj <- readRDS("roc_object_R_PMM.rds")

# Vérifier la classe de l'objet (probablement un objet de type 'roc' du package pROC)
class(roc_obj)

# Afficher les informations principales (AUC, nombre de cas, etc.)
print(roc_obj)

# Si vous avez le package pROC
library(pROC)

plot(roc_obj, 
     main = "Courbe ROC - Modèle avec Imputation PMM", 
     col = "#2196F3", 
     lwd = 4, 
     print.auc = TRUE, 
     auc.polygon = TRUE, 
     grid = TRUE)