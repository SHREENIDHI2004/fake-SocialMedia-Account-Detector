import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore', category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.engineering import build_feature_frame
from ml.training.data_loader import (
    load_primary_dataset,
    summarize_dataset,
)
from ml.training.train import _extract_target


def plot_confusion_matrix(y_true, y_pred, output_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Genuine (0)', 'Fake (1)'],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6, label='Baseline (AUC=0.5)')
    ax.legend(loc='lower right')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true, y_proba, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name='Model')
    baseline = y_true.mean()
    ax.axhline(baseline, linestyle='--', color='gray', alpha=0.6, label=f'Baseline (prevalence={baseline:.2f})')
    ax.legend(loc='lower left')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(y_true, y_proba, output_path: Path, title: str, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax.plot(mean_pred, frac_pos, marker='o', label='Model')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(title)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(feature_names, importance, output_path: Path, title: str, top_n: int = 20):
    order = np.argsort(importance)[::-1]
    order_top = order[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(order_top)), importance[order_top], color='steelblue')
    ax.set_yticks(range(len(order_top)))
    ax.set_yticklabels([feature_names[i] for i in order_top])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def sweep_thresholds(y_true, y_proba, threshold_grid=None, fpr_cap=0.15):
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.05, 1.0, 0.05), 3)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    rows = []
    for t in threshold_grid:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fpr = fp / max(n_neg, 1)
        fnr = fn / max(n_pos, 1)
        rows.append({
            'threshold': float(t),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'fpr': float(fpr),
            'fnr': float(fnr),
        })
    df = pd.DataFrame(rows)
    df_valid = df[df['fpr'] <= fpr_cap]
    if len(df_valid) == 0:
        idx = df['fpr'].idxmin()
    else:
        idx = df_valid.loc[df_valid['f1'].idxmax()].name
    selected = df.loc[idx].to_dict()
    return df, selected


def run_cv_out_of_fold_predictions(estimator_pipeline, X_df, y, n_splits: int = 5, random_state: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(y), dtype=float)
    fold_models = []
    for train_idx, val_idx in skf.split(X_df, y):
        X_tr = X_df.iloc[train_idx] if isinstance(X_df, pd.DataFrame) else X_df[train_idx]
        y_tr = y[train_idx]
        X_val = X_df.iloc[val_idx] if isinstance(X_df, pd.DataFrame) else X_df[val_idx]
        import copy
        m = copy.deepcopy(estimator_pipeline)
        m.fit(X_tr, y_tr)
        fold_models.append(m)
        proba = m.predict_proba(X_val)
        if proba.ndim == 2:
            proba = proba[:, 1]
        oof_proba[val_idx] = proba
    return oof_proba, fold_models


def fit_calibrator(base_pipeline, X_df, y, method: str = 'isotonic', n_splits: int = 5, random_state: int = 42):
    calibrator = CalibratedClassifierCV(
        base_pipeline,
        method=method,
        cv=n_splits,
        n_jobs=-1,
    )
    calibrator.fit(X_df, y)
    return calibrator


def main():
    parser = argparse.ArgumentParser(description="Calibration, threshold optimization, and final evaluation (held-out test).")
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--reports-dir', type=str, default=str(PROJECT_ROOT / 'reports'))
    parser.add_argument('--calibration-method', choices=['isotonic', 'sigmoid', 'none'], default='isotonic')
    parser.add_argument('--fpr-cap', type=float, default=0.15)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / 'models' / 'v1'
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, info = load_primary_dataset()
    print(summarize_dataset(train_df, test_df))

    if not info['available']:
        print("Cannot run evaluation: Kaggle dataset not downloaded. See data/README.md.")
        return 1

    X_train_df, y_train = _extract_target(train_df)
    X_test_df, y_test = _extract_target(test_df)

    import joblib
    model_path = models_dir / 'model_pipeline.joblib'
    if not model_path.exists():
        print("No tuned model found. Running tuning first (or baseline training)...")
        from ml.training.hyperparameter_tune import main as tune_main
        import shlex
        sys.argv = ['hyperparameter_tune.py', '--models', 'random_forest', 'hist_gradient_boosting',
                    '--n-iter', '15']
        try:
            tune_main()
        except SystemExit:
            pass
        sys.argv = ['evaluate.py']
    base_pipeline = joblib.load(model_path)

    print("\n=== OOF predictions on TRAINING data for threshold + calibration ===")
    oof_proba, _ = run_cv_out_of_fold_predictions(base_pipeline, X_train_df, y_train,
                                                  n_splits=args.n_splits,
                                                  random_state=args.random_state)

    print("\n=== Threshold sweep on OOF (training-CV) predictions ONLY ===")
    threshold_sweep, selected = sweep_thresholds(y_train, oof_proba, fpr_cap=args.fpr_cap)
    threshold_sweep.to_csv(reports_dir / 'threshold_sweep_oof.csv', index=False)
    print(f"Selected threshold = {selected['threshold']:.2f} (FPR cap = {args.fpr_cap})")
    print(f"  At threshold: P={selected['precision']:.4f} R={selected['recall']:.4f} "
          f"F1={selected['f1']:.4f} FPR={selected['fpr']:.4f} FNR={selected['fnr']:.4f}")

    print("\n=== Calibration fit on TRAINING split only (using CV inside CalibratedClassifierCV) ===")
    brier_raw = brier_score_loss(y_train, oof_proba)
    print(f"Brier score (raw, OOF train): {brier_raw:.6f}")

    calibrator = None
    if args.calibration_method != 'none':
        calibrator = fit_calibrator(base_pipeline, X_train_df, y_train,
                                     method=args.calibration_method,
                                     n_splits=args.n_splits,
                                     random_state=args.random_state)
        cal_oof_proba = calibrator.predict_proba(X_train_df)[:, 1]
        brier_cal = brier_score_loss(y_train, cal_oof_proba)
        print(f"Brier score ({args.calibration_method}, fit+predict train): {brier_cal:.6f}")
        use_calibration = brier_cal < brier_raw
        if use_calibration:
            print(f"Using calibrated probabilities (Brier {brier_raw:.4f} -> {brier_cal:.4f})")
        else:
            print(f"NOT using calibration (Brier did not improve: {brier_raw:.4f} -> {brier_cal:.4f})")
            calibrator = None
    else:
        use_calibration = False

    print("\n=== Final evaluation on FROZEN held-out TEST SET (used exactly once) ===")
    if calibrator is not None:
        proba_test = calibrator.predict_proba(X_test_df)[:, 1]
    else:
        proba_test = base_pipeline.predict_proba(X_test_df)[:, 1]

    threshold = float(selected['threshold'])
    y_pred_test = (proba_test >= threshold).astype(int)

    n_pos = int((y_test == 1).sum())
    n_neg = int((y_test == 0).sum())
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()
    test_metrics = {
        'n_samples_test': int(len(y_test)),
        'n_positive_test': n_pos,
        'n_negative_test': n_neg,
        'threshold': threshold,
        'accuracy': float(accuracy_score(y_test, y_pred_test)),
        'precision': float(precision_score(y_test, y_pred_test, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_test, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_test, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, proba_test)),
        'pr_auc': float(average_precision_score(y_test, proba_test)),
        'brier_score': float(brier_score_loss(y_test, proba_test)),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'fpr': float(fp / max(n_neg, 1)),
        'fnr': float(fn / max(n_pos, 1)),
        'calibration_used': str(args.calibration_method if calibrator is not None else 'none'),
    }
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred_test, target_names=['Genuine', 'Fake'], zero_division=0))

    plot_confusion_matrix(y_test, y_pred_test, reports_dir / 'confusion_matrix.png',
                          title=f'Confusion Matrix (test, t={threshold:.2f})')
    plot_roc_curve(y_test, proba_test, reports_dir / 'roc_curve.png', title='ROC Curve (test)')
    plot_pr_curve(y_test, proba_test, reports_dir / 'precision_recall_curve.png', title='Precision-Recall Curve (test)')
    plot_calibration_curve(y_test, proba_test, reports_dir / 'calibration_curve.png',
                           title=f'Calibration Curve (test, {args.calibration_method})')

    try:
        clf = base_pipeline.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            preprocessor = base_pipeline.named_steps['preprocessor']
            from ml.preprocessing.pipeline import get_feature_names_after_preprocessing
            feature_names = get_feature_names_after_preprocessing(preprocessor, list(X_train_df.columns))
            importances = np.asarray(clf.feature_importances_, dtype=float)
            if len(importances) == len(feature_names):
                plot_feature_importance(feature_names, importances,
                                        reports_dir / 'feature_importance.png',
                                        title='Feature Importance (model-native, test-held-out model)')
                pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances,
                }).sort_values('importance', ascending=False).to_csv(reports_dir / 'feature_importance.csv', index=False)
    except Exception as e:
        print(f"Feature importance plot skipped: {e}")

    with open(reports_dir / 'final_test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2)

    with open(reports_dir / 'classification_report_test.txt', 'w', encoding='utf-8') as f:
        f.write(classification_report(y_test, y_pred_test, target_names=['Genuine', 'Fake'], zero_division=0))

    if calibrator is not None:
        joblib.dump(calibrator, models_dir / 'calibrated_pipeline.joblib')

    final_metadata = {
        'selected_threshold': threshold,
        'calibration_method': args.calibration_method if calibrator is not None else 'none',
        'threshold_selection_method': f'OOF CV max F1 under FPR cap {args.fpr_cap}',
        'threshold_oof_metrics': selected,
        'test_metrics': test_metrics,
    }
    meta_path = models_dir / 'metadata.json'
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    metadata.update(final_metadata)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(models_dir / 'threshold.json', 'w', encoding='utf-8') as f:
        json.dump({'threshold': threshold}, f, indent=2)

    print(f"\nAll evaluation outputs saved to: {reports_dir}")
    print(f"Final calibrated pipeline + threshold saved to: {models_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
