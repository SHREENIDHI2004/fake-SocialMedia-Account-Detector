# Model Limitations

This project has important limitations that MUST be communicated to any end-user or downstream consumer
of the risk scores. Any GenAI explanation should reference this document to avoid overclaiming.

## 1. Training data limitations

### 1.1 Dataset platform focus
The classifier was trained on **Instagram-only account profiles** (Bakhshandeh 2019, 696 labeled accounts).
Performance on **Twitter/X, Facebook, TikTok, LinkedIn, or other platforms has NOT been evaluated** and
may be significantly worse. Cross-platform use is for research purposes only.

### 1.2 Dataset time period
Training data was **collected in March 2019**. Social-media fake-account behavior has evolved since then.
2024-2025 fake accounts often use AI-generated profile pictures, more realistic bios, and behavior patterns
not present in a 2019 dataset. The model should be expected to degrade over time and should be periodically
re-evaluated against newer labeled benchmarks.

### 1.3 Label noise
The dataset creator stated: *"the dataset has high level of accuracy though there might be a couple of
misidentified accounts in the spammers list as well."* We assume label noise in the range of ~2-5% is present.

### 1.4 Class distribution skew
The dataset is intentionally balanced (50/50) for easier training. In real-world platform traffic, the
true base rate of fake accounts among newly created accounts is typically in the 1-15% range depending on platform.
This means the raw probability must be considered **well-calibrated with respect to the training distribution**,
not the real-world base rate. At inference, apply domain-specific prior corrections if you know the expected
prevalence in your population.

## 2. Feature limitations

### 2.1 No behavioral temporal data
The model uses profile-level features ONLY. It has no access to:
- Posting time-series
- Comment / like / engagement patterns
- Follower / following growth curves
- Post content / caption text / images (except profile picture binary indicator and bio length)
- IP / device / location signals
- DM / interaction graphs

These behavioral signals are critical for detecting modern sophisticated bots.

### 2.2 is_verified feature availability
Per project specification, `is_verified` is only used when genuinely provided by an API. It is NOT
derived from follower counts. If verification is missing (as it is in the core training dataset), the model
uses a missing-indicator column. Expect reduced precision for high-follower accounts where verification data is missing.

### 2.3 Account age availability
Many APIs do not expose account creation date. When `account_age_days` is missing, the model uses an
imputation + missing-indicator approach. Interpret predictions with missing age with lower confidence.

### 2.4 No OCR / Computer Vision
The model does NOT analyze the content of the profile picture — only checks whether a picture exists.
Catfishing (stolen) images are not detected by the ML model itself. Manual reverse-image search (documented)
is required for borderline cases.

## 3. Operational limitations

### 3.1 No defense against adversarial evasion
Fake account farm operators actively adjust to evade detectors. Any public feature threshold can be reverse-engineered.
Do not expose raw feature values, thresholds, or model internals to untrusted users.

### 3.2 Calibration drift
Calibration (Platt / isotonic) is valid only on the distribution on which it was fit. As user populations
shift over time, the predicted probabilities can become uncalibrated. Re-calibrate every 3-6 months with
newly labeled data if available.

### 3.3 Do NOT use as sole decision factor
This tool is meant to flag accounts for **human review**. Never auto-suspend, auto-ban, or auto-punish a user
solely based on this risk score without human review.

The UI explicitly displays:
> "Risk assessment only — this result is not definitive proof that an account is fake."

## 4. Known failure cases (non-exhaustive)

1. **New legitimate users with low follower count:** High false-positive rate for newly signed-up users
   in their first week who legitimately follow many people.
2. **Follow-for-follow micro-influencers:** Real humans who do S4S / follow-back may score high false positive.
3. **Aged / compromised accounts:** A 5-year-old genuine account that was hacked and used for spam has
   profile features of a real account → false negative.
4. **Cross-platform:** Twitter/X / Facebook / TikTok profiles may have very different norms → use at own risk.
5. **Language drift:** Spam keyword lists become stale.
