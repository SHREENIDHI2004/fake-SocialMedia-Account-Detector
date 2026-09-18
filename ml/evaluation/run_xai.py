import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.data_loader import load_primary_dataset
from ml.training.train import _extract_target
from ml.preprocessing.pipeline import get_feature_names_after_preprocessing
from app.services.xai_service import (
    shap_available,
    generate_shap_summary_plot,
    explain_account,
)
from ml.features.engineering import build_feature_frame


def save_native_feature_importance(base_pipeline, X_train_df, reports_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    clf = base_pipeline.named_steps['clf']
    preprocessor = base_pipeline.named_steps['preprocessor']
    if not hasattr(clf, 'feature_importances_'):
        print("Model has no native feature_importances_; skipping.")
        return None
    feature_names = get_feature_names_after_preprocessing(preprocessor, list(X_train_df.columns))
    importances = np.asarray(clf.feature_importances_, dtype=float)
    if len(importances) != len(feature_names):
        print(f"Length mismatch: importances={len(importances)} names={len(feature_names)}")
        feature_names = [f'f{i}' for i in range(len(importances))]
    order = np.argsort(importances)[::-1]
    order_top = order[:30][::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(order_top)), importances[order_top], color='steelblue')
    ax.set_yticks(range(len(order_top)))
    ax.set_yticklabels([feature_names[i] for i in order_top])
    ax.set_xlabel('Gini Importance (Mean Decrease Impurity)')
    ax.set_title('Random Forest — Native Feature Importance (Top 30)')
    fig.tight_layout()
    out_png = reports_dir / 'feature_importance.png'
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    out_csv = reports_dir / 'feature_importance.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved native feature importance -> {out_png} , {out_csv}")
    print(f"Top 10 features:")
    for _, row in df.head(10).iterrows():
        print(f"  {row['feature']:<40s} {row['importance']:.4f}")
    return df


def main():
    reports_dir = PROJECT_ROOT / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / 'models' / 'v1'

    import joblib
    base_path = models_dir / 'model_pipeline.joblib'
    if not base_path.exists():
        print("No model pipeline found. Run evaluate.py first.")
        sys.exit(1)
    base_pipeline = joblib.load(base_path)

    train_df, test_df, info = load_primary_dataset()
    if not info['available']:
        print("Dataset not available")
        sys.exit(1)
    X_train_df, y_train = _extract_target(train_df)
    X_test_df, y_test = _extract_target(test_df)

    print("=" * 60)
    print("1. Native feature importance (Random Forest Gini)")
    print("=" * 60)
    save_native_feature_importance(base_pipeline, X_train_df, reports_dir)
    print()

    print("=" * 60)
    print(f"2. SHAP availability: {shap_available()}")
    print("=" * 60)
    if shap_available():
        out_summary = reports_dir / 'shap_summary_beeswarm.png'
        ok = generate_shap_summary_plot(base_pipeline, X_train_df, out_summary, max_samples=150)
        if ok:
            print(f"Saved SHAP summary -> {out_summary}")
        else:
            print("SHAP summary plot failed silently.")
        print()

        print("=" * 60)
        print("3. Per-account SHAP explanation (test samples)")
        print("=" * 60)
        preprocessor = base_pipeline.named_steps['preprocessor']
        feature_names = get_feature_names_after_preprocessing(preprocessor, list(X_train_df.columns))

        samples_shown = 0
        for i in range(len(X_test_df)):
            label = int(y_test[i])
            row = X_test_df.iloc[i:i + 1]
            exp = explain_account(base_pipeline, X_train_df, row, feature_names=feature_names, top_n=5)
            if not exp['available']:
                continue
            tag = 'GENUINE (0)' if label == 0 else 'FAKE (1)'
            print(f"\n  --- test_{i} {tag} ---")
            print(f"  Base value: {exp['shap_base_value']}")
            if exp['risk_factors']:
                print(f"  Top risk factors (increase fake prob):")
                for r in exp['risk_factors']:
                    print(f"    +{r['shap_value']:+.4f} {r['feature']}")
            if exp['positive_signals']:
                print(f"  Top positive signals (decrease fake prob):")
                for r in exp['positive_signals']:
                    print(f"    {r['shap_value']:+.4f} {r['feature']}")
            samples_shown += 1
            if samples_shown >= 4:
                break

        xai_test_path = reports_dir / 'shap_sample_explanations.json'
        with open(xai_test_path, 'w', encoding='utf-8') as f:
            json.dump({
                'explainer_available': exp['available'],
                'first_fake_explanation': exp,
            }, f, indent=2, default=str)
        print(f"\nSaved sample SHAP test -> {xai_test_path}")
    else:
        print("SHAP not installed; skipping SHAP plots. Install with: pip install shap")
    print("\nDONE.")


if __name__ == '__main__':
    main()
