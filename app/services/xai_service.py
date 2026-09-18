import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


_SHAP_AVAILABLE = False
try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False


def shap_available() -> bool:
    return _SHAP_AVAILABLE


def create_explainer(pipeline, X_train_df: pd.DataFrame):
    if not shap_available():
        return None, None
    preprocessor = pipeline.named_steps.get('preprocessor')
    clf = pipeline.named_steps.get('clf')
    X_train_preprocessed = preprocessor.transform(X_train_df)
    if hasattr(clf, 'predict_proba'):
        def model_fun(X_arr):
            p = clf.predict_proba(X_arr)
            if p.ndim == 2:
                return p[:, 1]
            return p
        try:
            explainer = shap.Explainer(model_fun, X_train_preprocessed, seed=42)
            return explainer, preprocessor
        except Exception:
            try:
                explainer = shap.KernelExplainer(model_fun, shap.sample(X_train_preprocessed, min(50, len(X_train_preprocessed)), seed=42))
                return explainer, preprocessor
            except Exception:
                return None, preprocessor
    return None, preprocessor


def explain_account(pipeline, X_train_df: pd.DataFrame, X_single_df: pd.DataFrame,
                    feature_names: List[str] = None, top_n: int = 5) -> Dict[str, Any]:
    result = {
        'available': False,
        'risk_factors': [],
        'positive_signals': [],
        'shap_base_value': None,
    }
    if not shap_available():
        return result
    explainer, preprocessor = create_explainer(pipeline, X_train_df)
    if explainer is None or preprocessor is None:
        return result

    X_single_preprocessed = preprocessor.transform(X_single_df)
    try:
        shap_values = explainer.shap_values(X_single_preprocessed)
    except Exception:
        return result

    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1]).reshape(-1)
    else:
        sv = np.asarray(shap_values).reshape(-1)

    if feature_names is None:
        from ml.preprocessing.pipeline import get_feature_names_after_preprocessing
        feature_names = get_feature_names_after_preprocessing(preprocessor, list(X_train_df.columns), X_ref=X_train_df)

    values = sv
    if len(feature_names) != len(values):
        feature_names = [f'f{i}' for i in range(len(values))]

    order = np.argsort(-np.abs(values))
    risk = []
    pos = []
    for idx in order[:2 * top_n]:
        item = {
            'feature': feature_names[idx],
            'shap_value': float(values[idx]),
            'direction': 'increases_risk' if values[idx] > 0 else 'decreases_risk',
        }
        if values[idx] > 0:
            risk.append(item)
        else:
            pos.append(item)
    result['available'] = True
    result['risk_factors'] = sorted(risk, key=lambda r: r['shap_value'], reverse=True)[:top_n]
    result['positive_signals'] = sorted(pos, key=lambda r: r['shap_value'])[:top_n]
    try:
        result['shap_base_value'] = float(getattr(explainer, 'expected_value', 0.0))
    except Exception:
        result['shap_base_value'] = None
    return result


def generate_shap_summary_plot(pipeline, X_train_df: pd.DataFrame, output_path: Path,
                               max_samples: int = 200):
    if not shap_available():
        return False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    explainer, preprocessor = create_explainer(pipeline, X_train_df)
    if explainer is None:
        return False
    sample_size = min(max_samples, len(X_train_df))
    sample = X_train_df.sample(sample_size, random_state=42) if sample_size < len(X_train_df) else X_train_df.copy()
    X_pre = preprocessor.transform(sample)
    try:
        sv = explainer.shap_values(X_pre)
        if isinstance(sv, list):
            sv_arr = np.asarray(sv[1])
        else:
            sv_arr = np.asarray(sv)
        from ml.preprocessing.pipeline import get_feature_names_after_preprocessing
        fn = get_feature_names_after_preprocessing(preprocessor, list(X_train_df.columns))
        fig, ax = plt.subplots(figsize=(11, 7))
        shap.summary_plot(sv_arr, X_pre, feature_names=fn, max_display=20, show=False)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"SHAP summary skipped: {e}")
        return False
