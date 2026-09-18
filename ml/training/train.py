import argparse
import json
import sys
import os
from pathlib import Path
from pprint import pprint
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.engineering import build_feature_frame
from ml.preprocessing.pipeline import build_preprocessing_pipeline
from ml.training.data_loader import (
    load_primary_dataset,
    summarize_dataset,
)


SCORING = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0),
    'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba'),
    'pr_auc': make_scorer(average_precision_score, response_method='predict_proba'),
}


def build_model_catalog(random_state: int = 42, class_weight='balanced'):
    models = OrderedDict()
    models['logistic_regression'] = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight=class_weight,
        random_state=random_state,
        solver='liblinear',
    )
    models['random_forest'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    models['extra_trees'] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    models['hist_gradient_boosting'] = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
    )
    models['gradient_boosting'] = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        random_state=random_state,
    )
    return models


def _extract_target(df: pd.DataFrame):
    y = pd.to_numeric(df['is_fake'], errors='coerce').astype(int).values
    records = df.drop(columns=['is_fake', 'data_source', 'dataset_split'], errors='ignore').to_dict('records')
    X_features = build_feature_frame(records)
    return X_features, y


def compare_models_cv(train_df: pd.DataFrame,
                      n_splits: int = 5,
                      random_state: int = 42,
                      include_optional_features: bool = True):
    if train_df is None or len(train_df) == 0:
        raise RuntimeError(
            "Training dataset is empty.\n"
            "Please download the Kaggle Bakhshandeh dataset:\n"
            "  1. pip install kaggle\n"
            "  2. kaggle datasets download -d free4ever1/instagram-fake-spammer-genuine-accounts "
            "-p data/kaggle_bakhshandeh --unzip\n"
            "  3. Ensure data/kaggle_bakhshandeh/ contains train.csv and test.csv\n"
            "See data/README.md for more information."
        )

    X_df, y = _extract_target(train_df)

    preprocessor = build_preprocessing_pipeline(include_optional_features=include_optional_features)
    models = build_model_catalog(random_state=random_state)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []
    progress = tqdm(models.items(), desc="Model comparison CV", total=len(models))
    for model_name, model in progress:
        progress.set_postfix_str(model_name)
        pipe = SkPipeline([
            ('preprocessor', preprocessor),
            ('clf', model),
        ])
        cv = cross_validate(
            pipe,
            X_df,
            y,
            cv=skf,
            scoring=SCORING,
            return_train_score=False,
            return_estimator=False,
            error_score='raise',
            n_jobs=1,
        )
        row = {
            'model': model_name,
        }
        for metric_name in SCORING.keys():
            vals = cv[f'test_{metric_name}']
            row[f'{metric_name}_mean'] = float(np.mean(vals))
            row[f'{metric_name}_std'] = float(np.std(vals))
            row[f'{metric_name}_min'] = float(np.min(vals))
            row[f'{metric_name}_max'] = float(np.max(vals))
        row['fit_time_mean_sec'] = float(np.mean(cv.get('fit_time', [0.0])))
        results.append(row)

    results_df = pd.DataFrame(results)
    return results_df, X_df, y, preprocessor


def print_results(results_df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("BASELINE MODEL COMPARISON — STRATIFIED 5-FOLD CV (TRAINING DATA ONLY)")
    print("=" * 100)
    cols = ['model',
            'f1_mean', 'f1_std',
            'pr_auc_mean', 'pr_auc_std',
            'roc_auc_mean', 'roc_auc_std',
            'precision_mean', 'precision_std',
            'recall_mean', 'recall_std',
            'accuracy_mean', 'accuracy_std']
    view = results_df[cols].copy()
    for c in cols[1:]:
        view[c] = view[c].map(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v)
    print(view.to_string(index=False))
    print()

    print("Ranked by F1 score (primary metric):")
    ranked = results_df.sort_values('f1_mean', ascending=False).reset_index(drop=True)
    for i, row in ranked.iterrows():
        print(
            f"  {i+1}. {row['model']:<32s} F1={row['f1_mean']:.4f}±{row['f1_std']:.4f} | "
            f"PR-AUC={row['pr_auc_mean']:.4f}±{row['pr_auc_std']:.4f} | "
            f"P={row['precision_mean']:.4f} R={row['recall_mean']:.4f}"
        )
    best = ranked.iloc[0]
    print(f"\nBest model by F1: {best['model']} (F1={best['f1_mean']:.4f})")
    return ranked


def save_comparison(results_df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nModel comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train + compare baseline ML models via stratified CV.")
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--reports-dir', type=str, default=str(PROJECT_ROOT / 'reports'))
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading primary dataset...")
    train_df, test_df, info = load_primary_dataset()
    print(summarize_dataset(train_df, test_df))
    print()
    print("Dataset availability info:")
    pprint(info)

    if not info['available']:
        print("\n" + "!" * 100)
        print("WARNING: Kaggle dataset not found locally.")
        print("You can still review the pipeline and code, but training cannot proceed.")
        print("Follow data/README.md to download the dataset and rerun this script.")
        print("!" * 100)
        print()
        print("Writing dataset_status.json...")
        with open(reports_dir / 'dataset_status.json', 'w', encoding='utf-8') as f:
            json.dump({
                'available': False,
                'message': 'Kaggle dataset not downloaded. See data/README.md.',
                'info': info,
            }, f, indent=2)
        return 1

    print("\nStarting 5-fold stratified CV baseline comparison...\n")
    results_df, X_df, y, preprocessor = compare_models_cv(
        train_df,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    ranked = print_results(results_df)
    save_comparison(results_df, reports_dir / 'model_comparison_cv.csv')

    status = {
        'available': True,
        'n_train_rows': len(train_df),
        'n_test_rows_frozen': len(test_df),
        'class_balance_train': info['train_class_distribution'],
        'best_model_by_f1': ranked.iloc[0]['model'],
        'best_f1_mean': float(ranked.iloc[0]['f1_mean']),
        'best_f1_std': float(ranked.iloc[0]['f1_std']),
        'all_models': results_df.to_dict(orient='records'),
    }
    with open(reports_dir / 'model_comparison_status.json', 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)

    print(f"\nStatus written to: {reports_dir / 'model_comparison_status.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
