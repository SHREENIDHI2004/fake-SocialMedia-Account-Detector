# Project Data Directory

This directory contains datasets used by the Suspicious Account Risk Assessment System.
None of the raw labeled training data is committed to git.

## Files Shipped in Repository

| File | Description | Labeled? | Source |
|------|-------------|--------|--------|
| `existing_samples.csv` | 10 sample accounts carried over from original project's sample/ (was `dataset.csv`) | **NO — no `is_fake` column; used exclusively as manual UI integration test inputs | Original project, author: SHREENIDHI2004 |
| `influencers_reference.csv` | 1,000 top Instagram influencers (Dec 2022) | NO — all presumed genuine, used POST-HOC for distribution sanity check only | HypeAuditor via original project author |

## PRIMARY TRAINING DATA (Must be downloaded — NOT committed)

### Kaggle: `kaggle_bakhshandeh/`

**Source:** https://www.kaggle.com/free4ever1/instagram-fake-spammer-genuine-accounts
**Uploader:** Bardiya Bakhshandeh (Kaggle user `free4ever1`)
**Date collected by author:** 15–19 March 2019 (crawler)
**License:** [Creative Commons Attribution 3.0 Unported (CC BY 3.0)
  — https://creativecommons.org/licenses/by/3.0/
**Redistribution:** Permitted WITH attribution. We do NOT redistribute the raw
  dataset files; the user of this repository downloads it locally following the steps below.
**Expected files after download:**
  - `train.csv` (576 rows, 12 columns + label)
  - `test.csv`  (120 rows, 12 columns + label — FROZEN held-out test set)

### Schema of Bakhshandeh Dataset Columns

Original columns (10 features + 1 label):

| Original Column Name | Renamed in our pipeline | Description |
|---|---|---|
| `profile pic` | `has_profile_pic` | Binary, 0/1 |
| `nums/length username` | `username_digit_ratio` | 0.0–1.0 |
| `fullname words` | `fullname_words` | integer count of tokens in display name |
| `nums/length fullname` | `fullname_digit_ratio` | 0.0–1.0 |
| `name==username` | `name_equals_username` | binary 0/1 |
| `description length` | `bio_length` | integer character count of bio |
| `external URL` | `has_external_url` | binary 0/1 |
| `private` | `is_private` | binary 0/1 |
| `#posts` | `posts` | integer |
| `#followers` | `followers` | integer |
| `#follows` | `following` | integer |
| `fake` | `is_fake` | **Label** binary 0 = genuine / 1 = fake/spammer |

Label definition (from the dataset's Data Card):

> *"I have personally identified the spammer/fake accounts included in this dataset after carefully examining each instance and as such the dataset has high level of accuracy though there might be a couple of misidentified accounts in the spammers list as well."* — B. Bakhshandeh, 2019.

In this project:

- The dataset's original `train.csv` is used for stratified 5-fold CV training, hyperparameter tuning, calibration fitting, and threshold optimization.
- The dataset's original `test.csv` is the **FROZEN held-out test set**. It is used exactly ONCE for the final evaluation report, and is not used for any model/feature/hyperparameter/threshold selection.

## How to Download

### Option A — Kaggle CLI (Recommended
```
pip install kaggle
kaggle datasets download -d free4ever1/instagram-fake-spammer-genuine-accounts -p data/kaggle_bakhshandeh --unzip
```
After downloading, verify file contents of `train.csv` + `test.csv` are in `data/kaggle_bakhshandeh/`.

### Option B — Web Download
Go to the URL above, sign in to Kaggle → "Download" button → extract zip → move both CSVs to `data/kaggle_bakhshandeh/`.

## Attribution required when reporting results (CC BY 3.0):

> *This project uses the "Instagram fake spammer genuine accounts" dataset, collected and published by Bardiya Bakhshandeh, available on Kaggle under CC BY 3.0. We thank the contributor for making this resource available to the research community.

## Citation of academic papers that use the same dataset (for traceability of prior results):

- Dey A., Reddy H., Manjistha D., Sinha N. (2019). Detection of Fake Accounts in Instagram using Machine Learning. IJCSIT Vol 11(5).
  https://aircconline.com/ijcsit/V11N5/11519ijcsit07.pdf
- Swetha B. (2018). Recognition of Serious Issue Haunting the Social Media Platforms to Detect Fake Accounts Using Machine Learning. IRJIET Vol.
- Azer et al. Multi platforms fake accounts detection based on federated learning (2024) — cites this dataset as Dataset 1 (IG fake: 693; real: 94).
