# Scoring de crédit immobilier — HMEQ

Projet de modélisation du risque de défaut de crédit immobilier sur la base de données **HMEQ**.
Approche multi-langages : **SAS**, **R** et **Python**.

---

## Structure du projet

| Fichier | Langage | Contenu |
|---------|---------|---------|
| `CodeSas.sas` | SAS | Imputation multiple PMM + construction du score |
| `CodeR.R` | R | Validation croisée + tests de robustesse |
| `CodePython.ipynb` | Python | Benchmark 4 algorithmes ML |
| `Rapport scoring.pdf` | — | Rapport complet |

---

## Méthodologie

### 1. Analyse SAS
- Imputation multiple par **PMM** (Predictive Mean Matching)
- Construction du score de défaut
- Analyse explicative des drivers du défaut

### 2. Validation R
- Validation croisée
- Tests de robustesse
- Vérification indépendante des résultats SAS

### 3. Modélisation Python — Benchmark comparatif

| Modèle | Description |
|--------|-------------|
| Régression logistique | Modèle de référence interprétable |
| Arbre de décision | Interprétabilité visuelle |
| Random Forest | Robustesse, réduction du surapprentissage |
| XGBoost | Performance maximale |

---

## Dataset HMEQ

Le dataset **HMEQ** (Home Equity) contient des informations sur des prêts immobiliers :
variables financières (montant du prêt, valeur du bien, ratio dette/revenu) et comportementales
(historique de paiement, incidents passés). Variable cible binaire : défaut (1) / remboursement (0).

---

## Installation Python

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
jupyter notebook CodePython.ipynb
```

---

## Stack technique

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-276DC3?style=flat-square&logo=r&logoColor=white)
![SAS](https://img.shields.io/badge/SAS-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-red?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

---

## Auteurs

**Emmanuel KOURAOGO**
[GitHub](https://github.com/EKOURAOGO) · [Email](mailto:ekouraogo73@gmail.com)
