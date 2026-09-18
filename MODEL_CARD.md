# Model Card — Fake Social Media Account Detector `v1`

**Model ID:** `fake-account-detector/random_forest_isotonic_v1`
**Artifact directory:** `models/v1/`
**Release status:** Reference implementation / Portfolio quality. No production deployment warranty expressed or implied.

---

## 1. Model purpose

Produce a calibrated binary score for a social-media profile given only public-profile-level metadata (follower counts, bio text, username, profile-completeness flags). The score represents the model's estimated probability that the account is fake / bot / inauthentic rather than a legitimate human-operated account.

The output is explicitly designed to be surfaced to a **human investigator**, not to drive automated account-takedown. The API surface wraps the score with SHAP attributions, RAG investigation context, recommended actions, and (optionally) an LLM narrative so the human has everything needed for a decision.

## 2. Intended use

- **Triage aid** for safety, trust-and-safety, and brand-protection teams who need to score lists of candidate accounts prior to manual review.
- **Demonstration and educational use** to show a complete ML pipeline: dataset ingestion → engineered features → multi-model comparison → calibration → threshold selection → XAI → RAG → GenAI wrapper.
- **Offline / batch scoring** of JSONL account records where operators collect their own data and feed records to the `/detect`, `/explain`, or `/report` endpoints.
- **Operating-point tuning by teams** via the `DETECTION_THRESHOLD`, `RISK_LEVEL_HIGH`, and `RISK_LEVEL_MEDIUM` environment variables to suit their own tolerance for false positives vs. false negatives.

## 3. Non-intended use

- **Automated account suspension / banning.** The model has not been audited for regulatory or contractual fairness and should never be the sole input to any decision that materially affects a person. A human-in-the-loop review is mandatory for any enforcement action.
- **Law-enforcement or employment screening.** No suitability is claimed for screening job candidates, tenants, or persons of interest.
- **Cross-platform deployment without re-evaluation.** The model was trained on Instagram-style labeled rows. Applying it to Twitter/X, TikTok, LinkedIn, Bluesky or closed DM networks without retraining / evaluation is an extrapolation and will almost certainly degrade.
- **Real-time detection for highly adversarial 2025-era fakes.** Modern bot farms (LLM-generated bios, avatar farms, aged residential IPs) were not represented in the 2019-2022 era training set. Performance is unknown and likely degraded for modern attacks.
- **High-volume inference per second without additional caching / load-testing.** The SHAP explainer and sentence-transformers RAG have cold-start latency.

## 4. Training data

- **Rows available:** 696 (576 tuning / training, 120 frozen held-out test).
- **Columns at input time (Kaggle naming convention):** `#followers`, `#follows`, `#posts`, `profile pic` (bool), `external URL` (bool), `private` (bool), `description length` (int char count after processing), `nums/length username` (float 0..1), `fullname words` (int), `nums/length fullname` (float 0..1), `name==username` (bool), plus a class label.
- **Class balance on full data after split:**
  - Training 576: balanced (≈50/50) per RandomizedSearchCV stratification.
  - Test 120: exactly 60 fake / 60 legit to give stable FPR/FNR estimates.
- **Missing values:** None at training time; Kaggle rows were complete. Feature engineering and the inference pipeline still impute defensively because real-world POST bodies come with many missing fields.

## 5. Data provenance

- **Primary dataset:** Kaggle — *Bakhshandeh Instagram fake and real accounts dataset*. Originally published in the context of binary fake-account classification papers / competitions. Download link is documented in `data/README.md`.
- **Labeling:** Labels were provided with the dataset release. Anecdotally they were crowd-labeled with review; the exact labeling protocol is not re-verified here. Operators should assume label noise at a few-percent level.
- **Collection date of data:** Not explicitly recorded by the release; inferred to be ~2022-era Instagram.
- **No additional features from API live calls were used for training.** The Kaggle snapshot is the sole supervision source for v1.
- **Secondary reference files** (`data/influencers_reference.csv`, `data/existing_samples.csv`) are not in the training set; they are reference / example-only files and the classifier never sees them at train or inference.
- **No human subject data beyond what Kaggle released is stored in this repo.** All committed CSVs are aggregate columns with no PII beyond the handle itself.

## 6. Labels

| Label value | Semantics                       | Count (train / test) |
|-------------|---------------------------------|----------------------|
| 1 / `fake`  | Fake / bot / inauthentic account | ≈288 train / 60 test |
| 0 / `legit` | Legitimate human-operated account | ≈288 train / 60 test |

The API responses expose a string enum `predicted_label` in `{"fake","legit"}` derived from thresholding the probability against `DETECTION_THRESHOLD` (default 0.50).

## 7. Features

The 12 raw Kaggle columns become ~40 engineered features in four groups. Feature groups are fully enumerated in `README.md §6` and implemented in `ml/features/engineering.py`:

1. **Ratio features (div-by-zero-safe):** `follower_following_ratio`, `following_follower_ratio`, `log1p_followers`, `log1p_following`, `log1p_posts`, `posts_per_follower`, `followers_per_day_age`, `posts_per_day_age`, `log1p_account_age_days`.
2. **Text features:** `username_digit_ratio`, `fullname_digit_ratio`, `name_equals_username` (0/1), `bio_has_url` (0/1), `bio_spam_keyword_count` (0..N). Spam lexicon: 18 phrases (see README §6).
3. **Profile flags:** `has_profile_pic`, `has_external_url`, `is_private`, `is_verified` (nullable booleans, preserved as NaN for imputation).
4. **Rule signals:** binary indicators for very/moderately low follower/following ratio, zero-posts-with-nonzero-followers, missing profile picture, very young (<30d) / young (<90d) account, very high following (>10k), bio spam-keyword count ≥ 2.

## 8. Preprocessing

Implemented by `ml/preprocessing/pipeline.py:build_preprocessing_pipeline`, which returns an sklearn `Pipeline` wrapping a `ColumnTransformer`:

1. **`ColumnSelector`** — projects an input DataFrame down to the schema the model expects; missing columns are added as NaN so inputs with missing fields still classify (defensive; not exercised during training since the Kaggle split was complete).
2. **`Log1pClip`** — log1p + upper-tail clipping on extreme count columns (defensive against absurd inputs).
3. **Per-type column transformer:**
   - Numeric columns → `SimpleImputer(strategy="median", add_indicator=True)` → `RobustScaler`. Missing-indicator columns are appended, so *which-fields-were-missing* is itself a signal.
   - Categorical / flag columns → `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")`.
   - If RobustScaler variant is requested unavailable the code falls back to StandardScaler with the same column plan.

4. **Feature name recovery** — `get_feature_names_after_preprocessing(preproc)` returns the column order, which is required by SHAP to label explanations.

Training-time note: imputer `fit()` is called on the 576-row training split only. The 120-row frozen test set is `transform()`ed only; no data leakage from test into fit.

## 9. Model architecture

- **Base estimator:** `sklearn.ensemble.RandomForestClassifier`.
- **Hyperparameters (selected by RandomizedSearchCV):**
  - `n_estimators=300`
  - `min_samples_leaf=10`
  - `max_features="log2"`
  - `max_depth=20`
  - `class_weight="balanced_subsample"`
  - `random_state=42`
- **Wrapper:** `sklearn.calibration.CalibratedClassifierCV(estimator=base_pipeline, method="isotonic", cv="prefit")` — isotonic regression calibrator fit on the 576-row training split. The CalibratedClassifierCV output is what is serialized to `models/v1/calibrated_pipeline.joblib` and what the API serves.
- **Inference shape:** input DataFrame N rows × P features → `predict_proba` returns N×2, class 1 probability is the score reported.
- **Compute requirement:** CPU-only. SHAP uses `shap.TreeExplainer` (fast for tree ensembles). No GPU needed.

Five model families were compared in the 5-fold CV sweep. Random Forest won on mean F1. Model comparison table: `README.md §8`; raw CSV: `reports/model_comparison_cv.csv`.

## 10. Evaluation methodology

**Distinction between verified (strong) performance and synthetic / weakly-labeled experiments:**

- **VERIFIED PERFORMANCE — frozen 120-row held-out test set.** This is the strong result. The 120 rows were partitioned before any preprocessing fit, feature engineering selection, hyperparameter search, model selection, or threshold selection. Metrics reported here are as close to unbiased as the small dataset permits.
- **REFERENCE-ONLY PERFORMANCE — cross-validation and OOF sweep.** These numbers come from the 576-row training split under 5-fold CV (CV means) or from OOF predictions (threshold selection OOF). They are useful internally for model pick and threshold pick but they overestimate true performance (CV optimism, leakage from repeated use); they must not be presented as the production number. The README and this model card lead with the held-out test numbers for that reason.

Protocol:

1. RandomizedSearchCV n_iter=30 × 5 folds on training split only → select best Random Forest by mean F1.
2. Refit best hyperparams on full 576 rows → fit isotonic calibrator → save both pipelines.
3. Threshold sweep over OOF CV predictions on the 576 rows: maximize F1 subject to FPR ≤ 0.15 → pick threshold = 0.50. OOF at 0.50: F1 = 0.9299, FPR = 0.087.
4. Apply calibrated model exactly once to the frozen 120-row split at threshold = 0.50 → report §11 metrics.
5. No second pass over the test set. No threshold re-adjustment after seeing test FPR/FNR.

## 11. Metrics

**Verified — Frozen 120-row held-out set (60 fake / 60 legit). Threshold = 0.50.**

| Metric       | Value |
|--------------|-------|
| Accuracy     | 0.875 |
| Precision    | 0.959 |
| Recall       | 0.783 |
| **F1**       | **0.8624** |
| ROC-AUC      | 0.9607 |
| PR-AUC       | 0.9634 |
| Brier score  | 0.0962 |
| FPR          | 0.033 (2 / 60 legit flagged) |
| FNR          | 0.217 (13 / 60 fake missed) |

Confusion matrix (threshold 0.50):

|                   | Pred Fake | Pred Legit |
|-------------------|-----------|------------|
| Actually Fake (60)| TP = 47   | FN = 13    |
| Actually Legit(60)| FP = 2    | TN = 58    |

**Reference-only (do NOT advertise as production performance):**

| Metric                         | OOF CV at threshold selection | 5-fold CV mean (Random Forest) |
|--------------------------------|-------------------------------|--------------------------------|
| F1                             | 0.9299                        | 0.9245                         |
| ROC-AUC                        | —                             | 0.9742                         |
| FPR                            | 0.087                         | —                              |

## 12. Threshold

- **Default threshold: 0.50**
- **Selection method:** `max(F1) subject to FPR ≤ 0.15` over the 5-fold OOF CV predictions on the 576 training rows.
- **OOF operating point at 0.50:** F1 = 0.9299, FPR = 0.087.
- **Held-out result at 0.50:** FPR = 0.033 (better OOB), FNR = 0.217 (worse than OOF — recall regressed ~6 pt).
- **Tuning guidance for users:**
  - If your queue can tolerate more false positives to catch more fakes, lower `DETECTION_THRESHOLD` to 0.35–0.40.
  - If you need almost-zero false positives (e.g. a ban-pending queue), raise to 0.70–0.80 and accept higher FNR.
- The model exposes `prediction.threshold` and `prediction.score` in every `/detect` response so operators can write their own downstream rules.

## 13. Calibration

- **Method:** Isotonic regression via `CalibratedClassifierCV(method="isotonic", cv="prefit")`.
- **Rationale:** Tree ensembles with 300 trees and heavy regularization tend to be reasonably well-calibrated, but isotonic calibration is cheap and materially improved Brier score (0.096 post-calibration on the held-out set). Platt scaling was considered and rejected because a held-out calibration split was not saved.
- **Interpretation of probability after calibration:** A score of 0.70 means the model expects ~70% of similarly-scoring examples in the test-distribution to actually be fake.
- **Brier score (lower=better):** 0.0962 (held-out)
- **Visual check:** Calibration curve in `reports/calibration_curve.png`. Calibration was checked visually; no major miscalibration regions at medium-to-high scores.
- **Limitation:** Isotonic calibration can be wiggly at extreme tails with only 576 training rows. Very-low and very-high probabilities should be treated ordinal rather than literal.

## 14. Limitations

1. **Small, single-platform data (n=576 train).** Statistical power is low; error bars on FPR/FNR are wide. A differently-sized 120-row test set could give materially different results. Confidence intervals not computed here.
2. **Instagram-only.** No representation of X/Twitter, TikTok, LinkedIn, Bluesky, Telegram, Discord, closed forums, or accounts with very different surface features.
3. **Label quality unknown.** Dataset was obtained via Kaggle release; no in-house re-labeling audit done.
4. **No content features.** No posting cadence, no post text NLP, no image/video/audio embeddings, no DM/comment corpus, no hashtag graphs. Known fake-detection signals from these modalities are completely absent.
5. **No graph signals.** Follower-followee graph, mutual-network triangle counts, share cascades are unavailable.
6. **Temporal drift.** Training data is several years old. Modern bots age their accounts, use synthetic avatars, write LLM bios, and post UGC-style content; v1 has not been reevaluated.
7. **Not fairness-audited.** Protected attributes (location, language, demographic proxies) were not checked for equalized odds, demographic parity, or calibration-by-group. Using the model against user populations from cultures / locales where English-language spam-kws don't work may degrade recall or shift FPR.
8. **Missing fields imputation at inference.** Optional fields (`account_age_days`, `is_verified`) that are present at train time are frequently missing in submitted JSON. Missing-indicator columns partially compensate, but performance on minimal-input calls is unmeasured beyond a few isolated unit tests.

## 15. Possible biases

- **Lexicon bias.** Spam keyword list is English only. Accounts spamming in Arabic, Spanish, Portuguese, Chinese, Hindi, etc. will have zero hits on the bio-spam signal, reducing the model's evidence mass available for those locales.
- **Digit ratio bias.** Username digit-ratio heuristic treats non-Arabic numerals (Devanagari, Bengali, CJK digits, etc.) as zero digits, which may advantage or disadvantage certain locale populations.
- **Platform-culture bias.** Low follower/following ratio is used as a fake signal, but some legitimate cultures (e.g. new-user cohorts on some platforms, creators building audience for the first time) naturally follow many more than follow them.
- **Verification badge skew.** Very few Kaggle training rows had `is_verified=true`. The model tends to strongly reduce risk on verified badges, but real-world verified-badge abuse can bypass that cue.
- **Age feature skew.** Very young account age (< 30 / < 90 days) is a strong fake signal, but new legitimate users will also be concentrated here and will see elevated risk scores.

## 16. Failure cases

Held-out 120-row set errors were harvested by `ml/evaluation/error_analysis.py` into `reports/error_analysis/false_positives.csv` and `false_negatives.csv`. Patterns observed:

1. **False positives (n = 2 on the test set at threshold 0.50):** Usually legitimate users at the low end of follower counts who happen to copy their username as their display name and/or have a newly-created (< 90-day old) account with few posts.
2. **False negatives (n = 13):** Stealth fakes who:
   - have a profile picture uploaded;
   - have follower/following ratios closer to 1.0 (managed / aged accounts);
   - have bios without spam lexicon hits;
   - have nonzero post counts (some engagement-bot farms post schedule).

3. **Extreme-input failure:** Accounts with JSON fields that are nonsense-strings (followers="not-a-number"). Mitigated by `_safe_int` after the bugfix in the finalization session. Tests in `tests/test_model.py` cover this.
4. **Cold-start latency:** First `/explain` call builds the SHAP explainer and sentence-transformer embedding. Failure case under timeout is "explain not ready yet"; no retry queuing implemented in v1.

## 17. Explainability

Per-request SHAP:
- **Service module:** `app/services/xai_service.py`.
- **Strategy:** TreeExplainer preferred (exact for Random Forest). Fallback: KernelExplainer with 10-row background sample of preprocessed training rows.
- **Caching:** Explainer singleton built on first explain call, then reused.
- **Output:** Up to 8 ranked risk factors, each signed by direction (increases_risk / decreases_risk), each with a human-readable description.
- **Training-phase visuals:** SHAP beeswarm summary at `reports/shap_summary_beeswarm.png`; per-sample attributions in `reports/shap_sample_explanations.json`.
- **Known SHAP limitation:** When numeric fields are missing at request time and SimpleImputer fills the median, the SHAP value for that feature reflects the imputed-median value, not the "absence of evidence" semantic directly; operators can cross-check with the missing-indicator feature's SHAP value.

## 18. GenAI limitations

The narrative assistant (§README 13) is a UX layer, not part of scoring:

- Only `GENAI_PROVIDER=openai` wired; only one model-id default (`gpt-4o-mini`). No other providers implemented in v1.
- The model prompt is deterministic template-assembled from evidence + recommendations + RAG snippets. No conversation history, no tool calls, no iterative refinement.
- Prompt-injection risk: User-controlled fields (`bio`, `username`) appear in the prompt context. Treat output as untrusted prose. This is why the API surfaces a separate `prediction.score` alongside `risk_summary`; decisions must use the score/SHAP, not the summary.
- Timeout / network / outages: bounded. `llm_available: false` appears, the heuristic `_summarize_without_llm` returns a rule-composed English summary instead.
- **Confabulation risk:** LLM output may invent citations, misinterpret risk factors, or invent next-steps. Recommended actions returned under `recommended_actions` are rule-derived and are the authoritative list; the prose summary of them is the part that can confabulate.

## 19. RAG limitations

- Small hand-built corpus (8 Markdown files, several hundred words each). Coverage is intentionally opinionated about what to teach investigators; it is NOT a general corpus about OSINT, bot taxonomy, or platform ToS enforcement.
- `all-MiniLM-L6-v2` embedding model: good for sentence-level semantics, limited for long-context reasoning. Query encoding and document encoding share the same model; no reranker stage.
- TF-IDF fallback backend operates on bag-of-words and fails for synonym queries; on the other hand it has zero cold-start latency.
- Updating KB: right now, editing `genai/rag/kb_docs/*.md` and restarting the server re-indexes. No API-available rebuild, no versioning, no changelog for the KB corpus in v1.

## 20. Security considerations

- **Secrets loading:** All keys via env-var read in `config.py` only; no hardcoded values. Empty string defaults mean the app starts safely with zero secrets configured.
- **.env must not be committed.** `.env` is gitignored; `.env.example` is the committed template.
- **No pickle of untrusted bytes.** `joblib.load` is only called on files written by the project's training code under `models/v1/`. Operators should never place a joblib file obtained from a third party into that directory.
- **Rate limiting (in-memory, 30 / min / IP):** Defends the SHAP and sentence-transformers endpoints. Not a substitute for an upstream WAF or API gateway in production.
- **`MAX_CONTENT_LENGTH = 2 MB`** on Flask. Prevents oversized POST bodies.
- **Validation stack:** Pydantic v2 schemas are preferred when installed; a fallback attribute-dict shim is provided if Pydantic is uninstalled. `_safe_int` is used inside feature engineering as a last line of defense for string / NaN / None / inf inputs.
- **TLS / auth out of scope for this repo.** Deploy behind a reverse proxy (nginx / PaaS edge) with TLS and client-IP forwarding. Add API-key auth middleware if exposing the service to untrusted networks.
- **Logging:** Application code never logs config['genai']['api_key'] or any secret. Operators should verify their gunicorn / nginx log format does not echo Authorization headers or body contents of `/explain` POSTs.

## 21. Reproducibility checklist

- [x] `random_state=42` on model, CV splitter, RandomizedSearch.
- [x] Explicit hyperparameters persisted to `models/v1/metadata.json`.
- [x] Threshold persisted to `models/v1/threshold.json`.
- [x] Frozen test set is 120 fixed rows; no test→train leakage in preproc pipeline.
- [x] requirements.txt pins minimum versions (not free-floating unpinned).
- [x] Training outputs (metrics CSVs, confusion matrix, ROC/PR/calibration curves, SHAP pngs) saved to `reports/`.
- [x] All project paths are relative to `config.py`; no machine-local absolute paths in code.
- [ ] ⚠️ Training script re-runs produce bit-identical models? Not verified here (joblib caching + sklearn small non-determinism in multithreaded RF). Expected to be statistically identical but not byte-identical.

## 22. Changelog / versioning

- **v1 — initial release.** Random Forest + Isotonic calibration. API v1 endpoints `/health`, `/model`, `/detect`, `/explain`, `/report`, UI at `/`. SHAP, RAG, and optional GenAI narrative.
- Future versions should bump `MODEL_VERSION` env var, add new subdir under `models/`, retain old artifacts for rollback, and update this model card with deltas.
