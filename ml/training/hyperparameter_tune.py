import argparse
import json
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.engineering import build_feature_frame
from ml.preprocessing.pipeline import build_preprocessing_pipeline
from ml.training.data_loader import load_primary_dataset
from ml.training.train import build_model_catalog, SCORING, compare_models_cv


def build_param_distributions(model_name: str, random_state: int = 42):
    dist = OrderedDict()
    if model_name == 'logistic_regression':
        dist = {
            'clf__C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            'clf__class_weight': [None, 'balanced'],
            'clf__solver': ['liblinear', 'lbfgs'],
        }
    elif model_name == 'random_forest':
        dist = {
            'clf__n_estimators': [200, 300, 500],
            'clf__max_depth': [None, 6, 10, 15, 20],
            'clf__min_samples_leaf': [1, 3, 5, 10],
            'clf__class_weight': [None, 'balanced', 'balanced_subsample'],
            'clf__max_features': ['sqrt', 'log2', None],
        }
    elif model_name == 'extra_trees':
        dist = {
            'clf__n_estimators': [200, 300, 500],
            'clf__max_depth': [None, 8, 12, 16],
            'clf__min_samples_leaf': [1, 3, 5, 8],
            'clf__class_weight': [None, 'balanced'],
        }
    elif model_name == 'hist_gradient_boosting':
        dist = {
            'clf__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'clf__max_depth': [3, 5, 7, 9, 12],
            'clf__max_iter': [200, 300, 500],
            'clf__min_samples_leaf': [10, 20, 30],
            'clf__l2_regularization': [0.0, 0.1, 1.0, 5.0, 10.0],
        }
    elif model_name == 'gradient_boosting':
        dist = {
            'clf__n_estimators': [150, 250, 400],
            'clf__learning_rate': [0.02, 0.05, 0.1, 0.15],
            'clf__max_depth': [2, 3, 4, 6],
            'clf__subsample': [0.8, 0.9, 1.0],
        }
    return dist


def tune_model_with_validation(X_df, y,
                               base_preprocessor,
                               model_name: str,
                               model_instance,
                               n_iter: int = 50,
                               n_splits: int = 5,
                               random_state: int = 42):
    pipe = SkPipeline([
        ('preprocessor', base_preprocessor),
        ('clf', model_instance),
    ])

    param_distributions = build_param_distributions(model_name, random_state)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='f1',
        cv=skf,
        refit='f1',
        verbose=0,
        random_state=random_state,
        n_jobs=-1,
        return_train_score=False,
        error_score='raise',
    )

    search.fit(X_df, y)
    return search


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning via RandomizedSearchCV on training data only.")
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--n-iter', type=int, default=60)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--models', nargs='*', default=[],
                        help="Which model names to tune (default: top 2 from model_comparison_cv.csv)")
    parser.add_argument('--reports-dir', type=str, default=str(PROJECT_ROOT / 'reports'))
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / 'models' / 'v1'
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, info = load_primary_dataset()
    if not info['available']:
        print("Kaggle dataset not available. Skipping tuning.")
        return 1

    from ml.training.train import _extract_target
    X_df, y = _extract_target(train_df)
    preprocessor = build_preprocessing_pipeline()

    models_to_tune = list(args.models) if args.models else []
    if not models_to_tune:
        cv_csv = reports_dir / 'model_comparison_cv.csv'
        if cv_csv.exists():
            comp = pd.read_csv(cv_csv)
            top = comp.sort_values('f1_mean', ascending=False).head(2)
            models_to_tune = list(top['model'].values)
        else:
            print("No prior model comparison found. Running comparison first...")
            comp_df, _, _, _ = compare_models_cv(train_df,
                                                  n_splits=args.n_splits,
                                                  random_state=args.random_state)
            comp_df.to_csv(reports_dir / 'model_comparison_cv.csv', index=False)
            top = comp_df.sort_values('f1_mean', ascending=False).head(2)
            models_to_tune = list(top['model'].values)

    print(f"Tuning models: {models_to_tune}")

    catalog = build_model_catalog(random_state=args.random_state)
    tuning_rows = []
    best_overall = None
    best_model_name = None

    for model_name in tqdm(models_to_tune, desc="Tuning"):
        print(f"\n=== Tuning {model_name} ===")
        model_instance = catalog[model_name]
        search = tune_model_with_validation(
            X_df=X_df,
            y=y,
            base_preprocessor=preprocessor,
            model_name=model_name,
            model_instance=model_instance,
            n_iter=args.n_iter,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        print(f"  Best F1 CV: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")
        tuning_rows.append({
            'model': model_name,
            'best_cv_f1': float(search.best_score_),
            'best_params': json.dumps(search.best_params_, sort_keys=True),
        })
        if best_overall is None or search.best_score_ > best_overall.best_score_:
            best_overall = search
            best_model_name = model_name

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(reports_dir / 'tuning_results.csv', index=False)
    print("\nTuning summary:")
    print(tuning_df.to_string(index=False))

    print(f"\nBest model after tuning: {best_model_name} (CV F1 = {best_overall.best_score_:.4f})")
    print(f"Params: {best_overall.best_params_}")

    import joblib
    best_pipeline = best_overall.best_estimator_
    joblib.dump(best_pipeline, models_dir / 'model_pipeline.joblib')
    metadata = {
        'model_name': best_model_name,
        'best_cv_f1': float(best_overall.best_score_),
        'best_params': best_overall.best_params_,
        'preprocessor_params': {},
        'random_state': args.random_state,
        'tuned_on_rows': len(train_df),
        'tuning_method': f'RandomizedSearchCV n_iter={args.n_iter} n_splits={args.n_splits} on training split only',
        'frozen_test_rows': len(test_df),
        'note': 'Frozen test set was NOT used for any tuning or selection.',
    }
    with open(models_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nSaved best model + metadata to {models_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
