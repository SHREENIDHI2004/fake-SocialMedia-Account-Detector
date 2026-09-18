# Project Methodology (Intended Use & Evaluation Protocols)

This document summarizes HOW the Suspicious Account Risk Assessment System was built.
It is referenced by the RAG retrieval layer when the GenAI assistant needs to explain methodology.

## 1. Project objective

Build and evaluate a **suspicious social-media account risk assessment system**.
The system outputs a probability of suspicion (0–1) and associated risk level (low/medium/high),
along with explainable evidence, to help a human reviewer decide whether to investigate further.

**This is NOT a definitive fake-account classifier.** The project explicitly uses the term
"risk assessment" because:
1. The model cannot perfectly distinguish fake from genuine accounts.
2. Ground truth availability is limited (only 696 labeled Instagram samples).
3. Risk-assessment outputs should ALWAYS trigger a human review, not an automated punitive action.

## 2. Modeling principles followed

### 2.1 Strict train/validation/test separation
- Test set = frozen held-out split downloaded as-is from Kaggle (`test.csv`, 120 rows), never modified,
  used **exactly once** for final evaluation after all model / threshold / calibration decisions are locked.
- **Test set is NOT used for:**
  - Feature selection
  - Threshold selection
  - Hyperparameter tuning
  - Calibration fitting
  - Model selection
  - SHAP visualization development (SHAP dev uses training CV samples only)

### 2.2 No data leakage
- All preprocessing statistics (scaler parameters, imputation medians/modes) are computed on training folds only inside CV.
- Synthetic data (if used) is never mixed into the frozen test set and is always clearly marked.

### 2.3 Evaluation metrics used
- Primary ranking metric: **F1 score** (harmonic mean of precision and recall).
- Secondary ranking: **PR-AUC** (more informative than ROC-AUC under imbalanced real-world priors).
- Also tracked: precision, recall, accuracy, ROC-AUC, Brier (calibration), FPR, FNR, confusion matrix.

### 2.4 Calibration
Predicted probabilities are tested for calibration reliability using Brier score and calibration curves.
Platt scaling (sigmoid) and isotonic regression are both evaluated, applied ONLY on out-of-fold training-CV predictions.
Calibration is retained only if it demonstrably improves Brier score.

### 2.5 Threshold policy
Default 0.5 threshold is NOT used blindly. Instead, threshold is selected on **training-CV OOF predictions**
by maximizing F1 subject to a hard cap FPR ≤ 0.15 (≤15% false-positive rate on genuine accounts in CV).
This reflects the project ethical norm: "Do not wrongly accuse real users of being fake more than 15% of the time."

### 2.6 Model selection policy
Models are ranked by cross-validated F1 on training splits only.
The simplest model (fewest dependencies, fewest parameters) whose F1 is within 2 percentage points of the
best model (at 95% confidence) is selected. We reject the assumption that "bigger = better" for tabular data
at n≈600.

## 3. GenAI usage boundary (architecturally enforced)

The LLM never makes the fake/genuine decision independently.
Flow order:
```
Account Data → Validation → Feature Engineering → ML Model → Calibrated Probability
     → SHAP/XAI → RAG Retrieval → LLM Investigation Assistant → Grounded Explanation
```

The LLM is ONLY allowed to:
1. Summarize *provided* ML evidence.
2. Retrieve relevant methodological knowledge from the RAG KB.
3. Recommend manual verification steps from the documented Investigation Guidance.
4. Explicitly list missing information (e.g., "account age was not available — verify manually").

The LLM is FORBIDDEN from:
- Inventing facts about the account not present in the evidence
- Stating definitively that the account is "fake" or "genuine"
- Modifying `risk_level` field independently

## 4. Regulatory / ethical considerations

This project adheres to the following constraints:
- **No protected attributes** (race, gender, religion, nationality, age-as-demographic, disability) are used.
- All features are platform-neutral profile signals that a user can inspect themselves.
- **No scraping policy violation:** Real API integration is limited to official SDKs and user-specified tokens.
- **No credential stuffing / password guessing whatsoever.**
- **No auto-ban or auto-suspend output:** outputs are advisory-only for human SOC reviewers.
- **GDPR rights:** If this system were ever applied to EU subjects, users would have rights under GDPR
  to request an explanation for decisions that significantly affect them. The SHAP + GenAI explanation pipeline
  was designed partially to satisfy this requirement.
