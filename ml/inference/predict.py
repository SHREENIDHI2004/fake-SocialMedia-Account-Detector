import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.engineering import build_account_features

_MODELS_DIR = PROJECT_ROOT / 'models' / 'v1'

_cache: Dict[str, Any] = {}


@dataclass
class ModelBundle:
    available: bool
    base_pipeline: Any = None
    calibrated_pipeline: Any = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None


def _load_pipeline():
    if 'pipeline' in _cache:
        return _cache['pipeline']
    import joblib
    cal_path = _MODELS_DIR / 'calibrated_pipeline.joblib'
    base_path = _MODELS_DIR / 'model_pipeline.joblib'
    path = cal_path if cal_path.exists() else base_path
    if not path.exists():
        return None
    pipeline = joblib.load(path)
    _cache['pipeline'] = pipeline
    return pipeline


def _load_threshold() -> float:
    if 'threshold' in _cache:
        return float(_cache['threshold'])
    p = _MODELS_DIR / 'threshold.json'
    default = 0.5
    if not p.exists():
        return default
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    t = float(data.get('threshold', default))
    _cache['threshold'] = t
    return t


def _load_metadata() -> dict:
    if 'metadata' in _cache:
        return _cache['metadata']
    p = _MODELS_DIR / 'metadata.json'
    if not p.exists():
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    _cache['metadata'] = data
    return data


def risk_level_from_probability(p: float, medium: float = 0.4, high: float = 0.7) -> str:
    if p >= high:
        return 'high'
    if p >= medium:
        return 'medium'
    return 'low'


def load_model_bundle(models_dir: Optional[Path] = None) -> ModelBundle:
    models_dir = Path(models_dir) if models_dir else _MODELS_DIR
    import joblib
    base_path = models_dir / 'model_pipeline.joblib'
    cal_path = models_dir / 'calibrated_pipeline.joblib'
    threshold_path = models_dir / 'threshold.json'
    meta_path = models_dir / 'metadata.json'

    if not base_path.exists() and not cal_path.exists():
        return ModelBundle(
            available=False,
            reason=f"No pipeline joblib found at {base_path} or {cal_path}",
        )
    base_pipeline = None
    if base_path.exists():
        try:
            base_pipeline = joblib.load(base_path)
        except Exception as e:
            return ModelBundle(available=False, reason=f"base pipeline load failed: {e}")
    cal_pipeline = None
    if cal_path.exists():
        try:
            cal_pipeline = joblib.load(cal_path)
        except Exception as e:
            return ModelBundle(available=False, reason=f"calibrated pipeline load failed: {e}")
    threshold = 0.5
    if threshold_path.exists():
        try:
            with open(threshold_path, 'r', encoding='utf-8') as f:
                threshold = float(json.load(f).get('threshold', 0.5))
        except Exception:
            threshold = 0.5
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    return ModelBundle(
        available=True,
        base_pipeline=base_pipeline,
        calibrated_pipeline=cal_pipeline,
        threshold=threshold,
        metadata=meta,
    )


def predict_probability(X_df: pd.DataFrame, bundle: ModelBundle) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X = X_df
    if bundle.base_pipeline is None and bundle.calibrated_pipeline is None:
        n = len(X_df)
        return np.full(n, np.nan), None
    raw_proba: Optional[np.ndarray] = None
    if bundle.base_pipeline is not None:
        try:
            raw_proba = np.asarray(bundle.base_pipeline.predict_proba(X))[:, 1].astype(float)
        except Exception:
            raw_proba = None
    cal_proba: Optional[np.ndarray] = None
    if bundle.calibrated_pipeline is not None:
        try:
            cal_proba = np.asarray(bundle.calibrated_pipeline.predict_proba(X))[:, 1].astype(float)
        except Exception:
            cal_proba = None
    if raw_proba is None:
        raw_proba = cal_proba if cal_proba is not None else np.full(len(X_df), np.nan)
    return raw_proba, cal_proba


def account_dict_to_features(account: Dict[str, Any]) -> pd.DataFrame:
    feats = build_account_features(
        username=account.get('username', ''),
        display_name=account.get('display_name') or account.get('name') or account.get('username', ''),
        bio=account.get('bio', '') or account.get('description', ''),
        followers=account.get('followers', 0),
        following=account.get('following', 0),
        posts=account.get('posts', 0),
        has_profile_pic=account.get('has_profile_pic', False),
        account_age_days=account.get('account_age_days'),
        is_verified=account.get('is_verified'),
        is_private=account.get('is_private'),
        has_external_url=account.get('has_external_url'),
    )
    return pd.DataFrame([feats])


def predict_account(account: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = _load_pipeline()
    threshold = _load_threshold()
    metadata = _load_metadata()

    result = {
        'model_version': metadata.get('model_version') or 'v1-unavailable',
        'model_available': pipeline is not None,
    }
    feats_df = account_dict_to_features(account)
    feats_dict = feats_df.iloc[0].replace({pd.NA: None}).to_dict()

    if pipeline is None:
        from ml.features.engineering import compute_rule_signals
        rs = compute_rule_signals(
            followers=account.get('followers', 0) or 0,
            following=account.get('following', 0) or 0,
            posts=account.get('posts', 0) or 0,
            has_profile_pic=bool(account.get('has_profile_pic', False)),
            account_age_days=account.get('account_age_days'),
            bio=account.get('bio', '') or '',
            is_verified=bool(account.get('is_verified', False)),
        )
        point_scale = {
            'rule_low_ratio_very': 30,
            'rule_low_ratio_mod': 15,
            'rule_no_posts_with_followers': 25,
            'rule_extreme_posts_per_follower': 20,
            'rule_bio_spam_keyword': 25,
            'rule_no_profile_pic': 15,
            'rule_very_young_account': 20,
            'rule_young_account': 10,
            'rule_very_low_followers': 10,
            'rule_very_high_following': 15,
        }
        score = sum(int(rs.get(k, 0)) * v for k, v in point_scale.items())
        heuristic_prob = float(min(score, 100)) / 100.0
        result.update({
            'risk_probability': heuristic_prob,
            'risk_level': risk_level_from_probability(heuristic_prob),
            'engine': 'heuristic_rule_based',
            'rule_signals': sorted([k for k, v in rs.items() if v]),
            'top_features': [],
            'positive_signals': [],
            'shap_values': None,
            'feature_values': feats_dict,
            'is_flagged': heuristic_prob >= float(threshold),
            'threshold': float(threshold),
        })
        return result

    proba = float(pipeline.predict_proba(feats_df)[0, 1])
    y_pred = int(proba >= threshold)
    level = risk_level_from_probability(proba)
    top_features, positive_signals = _interpret_from_importance(feats_dict, proba)

    rule_signals_triggered = []
    for k, v in feats_dict.items():
        if str(k).startswith('rule_') and v == 1:
            rule_signals_triggered.append(str(k))

    result.update({
        'risk_probability': proba,
        'risk_level': level,
        'predicted_label': y_pred,
        'threshold': threshold,
        'engine': 'ml_trained' + ('_calibrated' if (_MODELS_DIR / 'calibrated_pipeline.joblib').exists() else ''),
        'rule_signals': rule_signals_triggered,
        'top_features': top_features,
        'positive_signals': positive_signals,
        'feature_values': feats_dict,
        'shap_values': None,
        'is_flagged': proba >= float(threshold),
    })
    return result


def _interpret_from_importance(feats_dict: dict, proba: float, top_n: int = 5) -> Tuple[List[dict], List[dict]]:
    risk_markers = [
        ('rule_no_profile_pic', 'No profile picture'),
        ('rule_low_ratio_very', 'Extreme following/follower ratio'),
        ('rule_low_ratio_mod', 'Unbalanced following/follower ratio'),
        ('rule_very_young_account', 'Very new account (< 1 week)'),
        ('rule_young_account', 'Relatively new account (< 30 days)'),
        ('rule_bio_spam_keyword', 'Bio contains spam keywords'),
        ('rule_no_posts_with_followers', 'No posts but has followers'),
        ('rule_extreme_posts_per_follower', 'Extremely high posts / follower ratio'),
        ('rule_very_low_followers', 'Very low follower count (< 10)'),
        ('rule_very_high_following', 'Extremely high following count (> 3000)'),
        ('username_digit_ratio', None),
        ('fullname_digit_ratio', None),
        ('following_follower_ratio', None),
        ('bio_spam_keyword_hit', 'Bio spam keyword matched'),
    ]
    positive_markers = [
        ('has_profile_pic', 'Has profile picture'),
        ('is_verified', 'Account is verified'),
        ('log1p_account_age_days', None),
        ('follower_following_ratio', None),
        ('log1p_posts', None),
    ]

    risk_list = []
    for key, label in risk_markers:
        if key not in feats_dict:
            continue
        val = feats_dict[key]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if 'ratio' in key:
            if key == 'following_follower_ratio' and float(val) > 10.0:
                risk_list.append({
                    'feature': key,
                    'value': float(val),
                    'explanation': label or f'Following/follower ratio = {float(val):.2f} (high indicates suspicious follow-churn)',
                    'weight': 1.0,
                })
            continue
        if key in ('username_digit_ratio', 'fullname_digit_ratio'):
            if float(val) > 0.3:
                risk_list.append({
                    'feature': key,
                    'value': float(val),
                    'explanation': f'High digit ratio in {key.split("_")[0]} ({float(val):.2f}) suggests generated name',
                    'weight': 0.7,
                })
            continue
        if bool(val):
            risk_list.append({
                'feature': key,
                'value': bool(val),
                'explanation': label or key,
                'weight': 0.9,
            })
    risk_list.sort(key=lambda r: r['weight'], reverse=True)

    pos_list = []
    for key, label in positive_markers:
        if key not in feats_dict:
            continue
        val = feats_dict[key]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if key == 'follower_following_ratio':
            if float(val) >= 1.0:
                pos_list.append({
                    'feature': key,
                    'value': float(val),
                    'explanation': f'Balanced follower/following ratio ({float(val):.2f}) suggests a real user',
                    'weight': 0.8,
                })
            continue
        if key == 'log1p_account_age_days' and float(val) > np.log1p(90):
            pos_list.append({
                'feature': 'account_age_days',
                'value': float(np.expm1(float(val))),
                'explanation': f'Older account ({int(np.expm1(float(val)))} days) supports legitimacy',
                'weight': 0.7,
            })
            continue
        if key == 'log1p_posts' and float(val) > np.log1p(30):
            pos_list.append({
                'feature': 'posts',
                'value': float(np.expm1(float(val))),
                'explanation': 'Substantial posting history suggests real user behavior',
                'weight': 0.6,
            })
            continue
        if bool(val):
            pos_list.append({
                'feature': key,
                'value': bool(val),
                'explanation': label,
                'weight': 0.9 if key == 'is_verified' else 0.5,
            })
    pos_list.sort(key=lambda r: r['weight'], reverse=True)
    return risk_list[:top_n], pos_list[:top_n]
