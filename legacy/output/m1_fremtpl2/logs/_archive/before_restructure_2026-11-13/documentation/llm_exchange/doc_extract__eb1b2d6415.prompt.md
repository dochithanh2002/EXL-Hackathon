<!-- model: claude-opus-5-5 | step: doc_extract | sha256: eb1b2d64155a251b3c589270f9721de3a84e66f60cc40398f57fd0eb36a4b086 -->
# SYSTEM

You are the Documentation Agent in a bank's Model Risk Management factory. You reverse-engineer the methodology, variables, assumptions and limits of a legacy model from its source code and supporting documents, to build its model-inventory record and model card under PRA SS1/23. Independent validators and regulators read your output.

Non-negotiable rules:
1. DESCRIBE WHAT THE CODE DOES. Where code and documents disagree, the code wins; record the disagreement in `code_vs_docs_discrepancies`.
2. GROUND EVERY CLAIM. Cite line numbers from the numbered source listing in `source_lines`.
3. NO STATISTICS. Do not write coefficients, AUC, rates, p-values or counts. The factory computes and inserts every number. You may quote a number only if it appears verbatim in the source or the supporting documents.
4. COVER EVERY REQUIRED DISCLOSURE: at least one limitation per disclosure id, citing the id and naming the issue.
5. NEVER INVENT governance facts (owner, approvals, materiality, business use). If the evidence does not state them, say so and add an open question.
6. You never see data rows, only schema and aggregate facts.
7. Respond with ONE JSON object that matches the requested schema, and nothing else.

# USER

# Task: DOCUMENT the legacy model (reverse-engineer methodology, variables and limits)

## Legacy source (numbered; cite these line numbers)
```r
   1 | # ---------------------------------------------------------------------------
   2 | # LEGACY MODEL (as received) - French motor TPL claim frequency, Poisson GLM "GLM1"
   3 | # Source: reconstructed from Noll, Salzmann & Wuthrich (2018), "Case Study: French
   4 | #         Motor Third-Party Liability Claims", SSRN 3164764, Listing 2 / Section 3.
   5 | #         The original Kaggle notebook could not be retrieved; every line below follows
   6 | #         the published feature engineering and model of that case study. [reconstructed]
   7 | # Data:   freMTPL2freq (OpenML 41214, CC0), as exported by OpenML (strings in single quotes).
   8 | # This file is the INPUT to the R->Python migration agent. Do not edit.
   9 | # ---------------------------------------------------------------------------
  10 | 
  11 | dat <- read.csv("data/fremtpl2/freMTPL2freq.csv", quote = "\"'", stringsAsFactors = FALSE)
  12 | 
  13 | # claim counts above 4 and exposures above one year are treated as data errors (case study, Sec. 2)
  14 | dat$ClaimNb <- pmin(dat$ClaimNb, 4)
  15 | dat$Exposure <- pmin(dat$Exposure, 1)
  16 | 
  17 | # --- feature pre-processing for GLM1 (case study, Sec. 3.1) ---
  18 | # Area: ordinal A..F mapped to 1..6, used as a continuous variable
  19 | dat$AreaGLM <- as.integer(factor(dat$Area))
  20 | # VehPower: categorical, levels above 9 merged into 9
  21 | dat$VehPowerGLM <- as.factor(pmin(dat$VehPower, 9))
  22 | # VehAge: three classes [0,1), [1,10], (10,inf); class 2 is the reference
  23 | VehAgeGLM <- cbind(c(0:110), c(1, rep(2, 10), rep(3, 100)))
  24 | dat$VehAgeGLM <- as.factor(VehAgeGLM[dat$VehAge + 1, 2])
  25 | dat[, "VehAgeGLM"] <- relevel(dat[, "VehAgeGLM"], ref = "2")
  26 | # DrivAge: seven age classes 18-20, 21-25, 26-30, 31-40, 41-50, 51-70, 71+; class 5 (41-50) is the reference
  27 | DrivAgeGLM <- cbind(c(18:100), c(rep(1, 21 - 18), rep(2, 26 - 21), rep(3, 31 - 26), rep(4, 41 - 31),
  28 |                                  rep(5, 51 - 41), rep(6, 71 - 51), rep(7, 101 - 71)))
  29 | dat$DrivAgeGLM <- as.factor(DrivAgeGLM[dat$DrivAge - 17, 2])
  30 | dat[, "DrivAgeGLM"] <- relevel(dat[, "DrivAgeGLM"], ref = "5")
  31 | # BonusMalus: capped at 150, continuous
  32 | dat$BonusMalusGLM <- as.integer(pmin(dat$BonusMalus, 150))
  33 | # Density: log-transformed, continuous
  34 | dat$DensityGLM <- as.numeric(log(dat$Density))
  35 | # VehBrand and Region: categorical with references B1 and R24 (largest classes)
  36 | dat$VehBrand <- relevel(as.factor(dat$VehBrand), ref = "B1")
  37 | dat$Region <- relevel(as.factor(dat$Region), ref = "R24")
  38 | dat$VehGas <- as.factor(dat$VehGas)
  39 | 
  40 | # --- learning / test split: 90% / 10% (case study, Sec. 2) ---
  41 | set.seed(100)
  42 | ll <- sample(c(1:nrow(dat)), round(0.9 * nrow(dat)), replace = FALSE)
  43 | learn <- dat[ll, ]
  44 | test <- dat[-ll, ]
  45 | 
  46 | # --- GLM1: Poisson frequency model with log-exposure offset ---
  47 | d.glm1 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand + VehGas +
  48 |                 DensityGLM + Region + AreaGLM,
  49 |               data = learn, offset = log(Exposure), family = poisson())
  50 | 
  51 | learn$fitGLM1 <- fitted(d.glm1)
  52 | test$fitGLM1 <- predict(d.glm1, newdata = test, type = "response")
```

## Supporting documents (narrative and code only; console output stripped, guardrail G-11)
(none supplied)

## Facts established by the factory (deterministic; do not restate the numbers)
{
 "model": {
  "class": [
   "glm",
   "lm"
  ],
  "family": "poisson",
  "link": "log",
  "formula": "ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region + AreaGLM",
  "response": "ClaimNb",
  "input_variables": [
   "VehPowerGLM",
   "VehAgeGLM",
   "DrivAgeGLM",
   "BonusMalusGLM",
   "VehBrand",
   "VehGas",
   "DensityGLM",
   "Region",
   "AreaGLM"
  ],
  "factor_levels": {
   "VehPowerGLM": [
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
   ],
   "VehAgeGLM": [
    "2",
    "1",
    "3"
   ],
   "DrivAgeGLM": [
    "5",
    "1",
    "2",
    "3",
    "4",
    "6",
    "7"
   ],
   "VehBrand": [
    "B1",
    "B10",
    "B11",
    "B12",
    "B13",
    "B14",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6"
   ],
   "VehGas": [
    "Diesel",
    "Regular"
   ],
   "Region": [
    "R24",
    "R11",
    "R21",
    "R22",
    "R23",
    "R25",
    "R26",
    "R31",
    "R41",
    "R42",
    "R43",
    "R52",
    "R53",
    "R54",
    "R72",
    "R73",
    "R74",
    "R82",
    "R83",
    "R91",
    "R93",
    "R94"
   ]
  },
  "contrasts": {
   "VehPowerGLM": "contr.treatment",
   "VehAgeGLM": "contr.treatment",
   "DrivAgeGLM": "contr.treatment",
   "VehBrand": "contr.treatment",
   "VehGas": "contr.treatment",
   "Region": "contr.treatment"
  },
  "n_coefficients": 49,
  "converged": true,
  "iterations": 6,
  "deviance": 191651.269966774,
  "null_deviance": 201966.364055248,
  "aic": 254279.763151247,
  "r_version": "R version 4.6.1 (2026-06-24)",
  "r_packages_loaded": [
   "compiler",
   "jsonlite"
  ]
 },
 "column_classes_after_preprocessing": {
  "IDpol": "numeric",
  "ClaimNb": "numeric",
  "Exposure": "numeric",
  "Area": "character",
  "VehPower": "integer",
  "VehAge": "integer",
  "DrivAge": "integer",
  "BonusMalus": "integer",
  "VehBrand": "factor",
  "VehGas": "factor",
  "Density": "integer",
  "Region": "factor",
  "AreaGLM": "integer",
  "VehPowerGLM": "factor",
  "VehAgeGLM": "factor",
  "DrivAgeGLM": "factor",
  "BonusMalusGLM": "integer",
  "DensityGLM": "numeric",
  "fitGLM1": "numeric"
 },
 "split": {
  "n_train": 610212,
  "n_test": 67801,
  "seed_used_by_harness": 100
 },
 "protected_attributes_in_data": [
  "DrivAge"
 ],
 "protected_attributes_used_as_inputs": [
  "DrivAgeGLM"
 ],
 "data_profile": {
  "n_rows": 678013,
  "columns": [
   {
    "name": "IDpol",
    "dtype": "float64",
    "n_unique": 678013,
    "n_missing": 0
   },
   {
    "name": "ClaimNb",
    "dtype": "int64",
    "n_unique": 11,
    "n_missing": 0,
    "levels": [
     0,
     1,
     2,
     3,
     4,
     5,
     6,
     8,
     9,
     11,
     16
    ]
   },
   {
    "name": "Exposure",
    "dtype": "float64",
    "n_unique": 181,
    "n_missing": 0
   },
   {
    "name": "Area",
    "dtype": "str",
    "n_unique": 6,
    "n_missing": 0,
    "levels": [
     "'A'",
     "'B'",
     "'C'",
     "'D'",
     "'E'",
     "'F'"
    ]
   },
   {
    "name": "VehPower",
    "dtype": "int64",
    "n_unique": 12,
    "n_missing": 0,
    "levels": [
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15
    ]
   },
   {
    "name": "VehAge",
    "dtype": "int64",
    "n_unique": 78,
    "n_missing": 0
   },
   {
    "name": "DrivAge",
    "dtype": "int64",
    "n_unique": 83,
    "n_missing": 0
   },
   {
    "name": "BonusMalus",
    "dtype": "int64",
    "n_unique": 115,
    "n_missing": 0
   },
   {
    "name": "VehBrand",
    "dtype": "str",
    "n_unique": 11,
    "n_missing": 0,
    "levels": [
     "'B1'",
     "'B10'",
     "'B11'",
     "'B12'",
     "'B13'",
     "'B14'",
     "'B2'",
     "'B3'",
     "'B4'",
     "'B5'",
     "'B6'"
    ]
   },
   {
    "name": "VehGas",
    "dtype": "str",
    "n_unique": 2,
    "n_missing": 0,
    "levels": [
     "'Diesel'",
     "'Regular'"
    ]
   },
   {
    "name": "Density",
    "dtype": "int64",
    "n_unique": 1607,
    "n_missing": 0
   },
   {
    "name": "Region",
    "dtype": "str",
    "n_unique": 22,
    "n_missing": 0
   }
  ]
 }
}

## REQUIRED disclosures (detected deterministically; each id must be covered by a limitation)
{
 "D-01:DrivAgeGLM": "protected / sensitive attribute 'DrivAgeGLM' (derived from 'DrivAge') is a direct model input (Equality Act 2010, Consumer Duty)",
 "D-06": "values are capped / floored before fitting (pmin/pmax); capping a response or exposure changes what the model estimates (lines [14, 15, 21, 32])",
 "D-05": "terms not significant at 5% in the final model: ['DrivAgeGLM1', 'DrivAgeGLM7', 'VehBrandB10', 'VehBrandB13', 'VehBrandB14', 'VehBrandB2', 'VehBrandB3', 'VehBrandB4', 'VehBrandB6', 'RegionR21', 'RegionR22', 'RegionR25', 'RegionR26', 'RegionR42', 'RegionR43', 'RegionR53', 'RegionR74', 'RegionR82', 'RegionR94', 'AreaGLM']"
}

List every formula variable and the target in `variables`, with factor levels exactly as captured.
Output JSON schema:
{
 "$defs": {
  "DocLimitation": {
   "properties": {
    "text": {
     "title": "Text",
     "type": "string"
    },
    "severity": {
     "enum": [
      "High",
      "Medium",
      "Low"
     ],
     "title": "Severity",
     "type": "string"
    },
    "disclosure_ids": {
     "description": "Required disclosure ids (D-xx) this limitation covers; [] if none",
     "items": {
      "type": "string"
     },
     "title": "Disclosure Ids",
     "type": "array"
    },
    "source_lines": {
     "items": {
      "type": "integer"
     },
     "title": "Source Lines",
     "type": "array"
    }
   },
   "required": [
    "text",
    "severity",
    "disclosure_ids"
   ],
   "title": "DocLimitation",
   "type": "object"
  },
  "DocStep": {
   "properties": {
    "order": {
     "title": "Order",
     "type": "integer"
    },
    "description": {
     "title": "Description",
     "type": "string"
    },
    "columns": {
     "items": {
      "type": "string"
     },
     "title": "Columns",
     "type": "array"
    },
    "source_lines": {
     "items": {
      "type": "integer"
     },
     "title": "Source Lines",
     "type": "array"
    },
    "author_rationale": {
     "description": "The author's stated reason, from comments or documents; '' if none",
     "title": "Author Rationale",
     "type": "string"
    },
    "is_assumption": {
     "description": "True if the step rests on an unverified assumption by the author",
     "title": "Is Assumption",
     "type": "boolean"
    }
   },
   "required": [
    "order",
    "description",
    "columns",
    "source_lines",
    "author_rationale",
    "is_assumption"
   ],
   "title": "DocStep",
   "type": "object"
  },
  "DocVariable": {
   "properties": {
    "name": {
     "description": "Variable name exactly as it appears in the model formula",
     "title": "Name",
     "type": "string"
    },
    "role": {
     "enum": [
      "input",
      "target"
     ],
     "title": "Role",
     "type": "string"
    },
    "kind": {
     "enum": [
      "numeric",
      "categorical",
      "ordinal_as_numeric",
      "binary"
     ],
     "title": "Kind",
     "type": "string"
    },
    "description": {
     "description": "Business meaning, from source comments / supporting documents",
     "title": "Description",
     "type": "string"
    },
    "levels": {
     "description": "Categorical/binary: R factor levels in R order",
     "items": {
      "type": "string"
     },
     "title": "Levels",
     "type": "array"
    },
    "level_meaning": {
     "additionalProperties": {
      "type": "string"
     },
     "description": "Level code -> meaning, if documented",
     "title": "Level Meaning",
     "type": "object"
    },
    "transformation": {
     "description": "What the code does to this variable before fitting; 'none' if nothing",
     "title": "Transformation",
     "type": "string"
    },
    "source_lines": {
     "description": "Line numbers in the numbered legacy source that define or use it",
     "items": {
      "type": "integer"
     },
     "title": "Source Lines",
     "type": "array"
    }
   },
   "required": [
    "name",
    "role",
    "kind",
    "description",
    "transformation",
    "source_lines"
   ],
   "title": "DocVariable",
   "type": "object"
  }
 },
 "properties": {
  "purpose": {
   "title": "Purpose",
   "type": "string"
  },
  "intended_use": {
   "title": "Intended Use",
   "type": "string"
  },
  "out_of_scope_use": {
   "title": "Out Of Scope Use",
   "type": "string"
  },
  "methodology": {
   "description": "Estimation method in plain words; no statistics",
   "title": "Methodology",
   "type": "string"
  },
  "target_definition": {
   "title": "Target Definition",
   "type": "string"
  },
  "variables": {
   "description": "EVERY formula variable plus the target, nothing else",
   "items": {
    "$ref": "#/$defs/DocVariable"
   },
   "title": "Variables",
   "type": "array"
  },
  "preprocessing": {
   "description": "In source execution order",
   "items": {
    "$ref": "#/$defs/DocStep"
   },
   "title": "Preprocessing",
   "type": "array"
  },
  "variable_selection": {
   "title": "Variable Selection",
   "type": "string"
  },
  "assumptions": {
   "items": {
    "type": "string"
   },
   "title": "Assumptions",
   "type": "array"
  },
  "limitations": {
   "items": {
    "$ref": "#/$defs/DocLimitation"
   },
   "title": "Limitations",
   "type": "array"
  },
  "code_vs_docs_discrepancies": {
   "items": {
    "type": "string"
   },
   "title": "Code Vs Docs Discrepancies",
   "type": "array"
  },
  "open_questions": {
   "description": "Questions for the model owner; facts that cannot be established from the evidence",
   "items": {
    "type": "string"
   },
   "title": "Open Questions",
   "type": "array"
  }
 },
 "required": [
  "purpose",
  "intended_use",
  "out_of_scope_use",
  "methodology",
  "target_definition",
  "variables",
  "preprocessing",
  "variable_selection",
  "assumptions",
  "limitations",
  "code_vs_docs_discrepancies",
  "open_questions"
 ],
 "title": "DocExtract",
 "type": "object"
}
