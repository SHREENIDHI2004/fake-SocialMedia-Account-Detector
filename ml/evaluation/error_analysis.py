import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.data_loader import load_primary_dataset
from ml.training.train import _extract_target


def analyze_errors(test_df: pd.DataFrame, X_test_df: pd.DataFrame,
                   y_test, proba_test, y_pred_test, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = test_df.reset_index(drop=True).copy()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()

    y_test_arr = np.asarray(y_test)
    proba_arr = np.asarray(proba_test)
    y_pred_arr = np.asarray(y_pred_test)

    test_df['y_true'] = y_test_arr
    test_df['y_pred'] = y_pred_arr
    test_df['proba'] = proba_arr
    test_df['abs_error'] = np.abs(y_test_arr - proba_arr)

    fp_mask = (y_test_arr == 0) & (y_pred_arr == 1)
    fn_mask = (y_test_arr == 1) & (y_pred_arr == 0)
    correct_mask = y_test_arr == y_pred_arr

    fp_df = test_df[fp_mask].sort_values('proba', ascending=False).copy()
    fn_df = test_df[fn_mask].sort_values('proba', ascending=True).copy()
    hard_mask = (test_df['abs_error'] > 0.45) & ~fp_mask & ~fn_mask
    hard_df = test_df[hard_mask].sort_values('abs_error', ascending=False).copy()

    summary = {
        'n_total': int(len(y_test)),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'fp_rate_among_genuine': float(fp / max(tn + fp, 1)),
        'fn_rate_among_fake': float(fn / max(tp + fn, 1)),
    }

    with open(output_dir / 'error_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    fp_df.to_csv(output_dir / 'false_positives.csv', index=False)
    fn_df.to_csv(output_dir / 'false_negatives.csv', index=False)
    hard_df.to_csv(output_dir / 'hard_cases.csv', index=False)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ERROR ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nCounts: TP={tp} TN={tn} FP={fp} FN={fn}")
    report_lines.append(f"FPR (of genuine): {summary['fp_rate_among_genuine']:.2%}")
    report_lines.append(f"FNR (of fake):    {summary['fn_rate_among_fake']:.2%}")
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("FALSE POSITIVES (genuine accounts predicted as fake)")
    report_lines.append("-" * 80)
    if len(fp_df):
        for _, row in fp_df.iterrows():
            report_lines.append(f"  prob={row['proba']:.3f} | username={row.get('username', 'N/A')} "
                                f"followers={row.get('followers','?')} following={row.get('following','?')} "
                                f"posts={row.get('posts','?')} pic={row.get('has_profile_pic','?')}")
        report_lines.append(f"\n  Possible causes: extreme follower/following ratio, missing bio, unverified small accounts")
    else:
        report_lines.append("  (none)")

    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("FALSE NEGATIVES (fake accounts predicted as genuine)")
    report_lines.append("-" * 80)
    if len(fn_df):
        for _, row in fn_df.iterrows():
            report_lines.append(f"  prob={row['proba']:.3f} | username={row.get('username', 'N/A')} "
                                f"followers={row.get('followers','?')} following={row.get('following','?')} "
                                f"posts={row.get('posts','?')} pic={row.get('has_profile_pic','?')}")
        report_lines.append(f"\n  Possible causes: fake accounts with normal profile/follower counts, no bio spam text")
    else:
        report_lines.append("  (none)")

    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("MODEL IMPROVEMENT RECOMMENDATIONS (derived)")
    report_lines.append("-" * 80)
    if fp > 0:
        report_lines.append(f"  * Consider higher threshold or higher FPR cap to reduce false positives (currently {fp})")
        report_lines.append(f"  * Add profile-verification / historical-activity features if available to reduce legitimate low-profile false alarms.")
    if fn > 0:
        report_lines.append(f"  * Consider collecting more text/NLP features (bio word-level TF-IDF, username n-grams) to catch fakes that mimic genuine profile stats.")
        report_lines.append(f"  * Consider synthetic fake augmentation if ablation shows real-test improvement.")
    if fp == 0 and fn == 0:
        report_lines.append(f"  * No errors on this test split — verify via cross-platform influencers-reference sanity check.")
    report_lines.append("  * Always manually investigate borderline cases (prob 0.40-0.60); see hard_cases.csv")

    report_txt = "\n".join(report_lines)
    with open(output_dir / 'error_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_txt)
    print(report_txt)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Error analysis after final evaluation.")
    parser.add_argument('--reports-dir', type=str, default=str(PROJECT_ROOT / 'reports'))
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / 'models' / 'v1'

    train_df, test_df, info = load_primary_dataset()
    if not info['available']:
        print("Cannot run error analysis: Kaggle dataset not downloaded.")
        return 1

    X_test_df, y_test = _extract_target(test_df)

    import joblib
    model_path = models_dir / 'model_pipeline.joblib'
    cal_path = models_dir / 'calibrated_pipeline.joblib'
    threshold_path = models_dir / 'threshold.json'

    if not model_path.exists() or not threshold_path.exists():
        print("Cannot run error analysis: model/threshold not saved. Run evaluate.py first.")
        return 1

    pipeline = joblib.load(cal_path if cal_path.exists() else model_path)
    with open(threshold_path, 'r', encoding='utf-8') as f:
        threshold_info = json.load(f)
    threshold = float(threshold_info['threshold'])

    proba = pipeline.predict_proba(X_test_df)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    analyze_errors(test_df, X_test_df, y_test, proba, y_pred, reports_dir / 'error_analysis')
    return 0


if __name__ == '__main__':
    sys.exit(main())
