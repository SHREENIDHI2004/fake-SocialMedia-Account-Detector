import pandas as pd
import numpy as np
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


STRATEGY_PROFILE = 'include'  # 'include' (with indicator) | 'exclude' | 'drop_feature'


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self._columns = list(self.columns)
        return self

    def transform(self, X):
        cols = getattr(self, '_columns', list(self.columns))
        if isinstance(X, pd.DataFrame):
            present = [c for c in cols if c in X.columns]
            missing = [c for c in cols if c not in X.columns]
            df = X[present].copy()
            for c in cols:
                if c in missing:
                    df[c] = np.nan
            df = df[cols]
            return df
        return X


class Log1pClip(BaseEstimator, TransformerMixin):
    def __init__(self, columns, clip_percentile: float = 0.995):
        self.columns = columns
        self.clip_percentile = clip_percentile

    def fit(self, X, y=None):
        self._columns = list(self.columns)
        self.clips_ = {}
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._columns)
        for c in self._columns:
            if c in X.columns:
                vals = pd.to_numeric(X[c], errors='coerce').fillna(0.0)
                q = float(np.quantile(vals.values, self.clip_percentile)) if len(vals) else 0.0
                self.clips_[c] = max(q, 1.0)
        return self

    def transform(self, X):
        cols = getattr(self, '_columns', list(self.columns))
        clips = getattr(self, 'clips_', {})
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=cols)
        out = pd.DataFrame(index=df.index)
        for c in cols:
            vals = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype(float)
            clip = clips.get(c, None)
            if clip is not None:
                vals = vals.clip(upper=clip)
            out[c] = np.log1p(vals.values)
        return out.values if not isinstance(X, pd.DataFrame) else out


def build_preprocessing_pipeline(include_optional_features: bool = True,
                                 use_robust_scaler: bool = True,
                                 strategy_profile: str = STRATEGY_PROFILE):
    numeric_skewed = [
        'log1p_followers', 'log1p_following', 'log1p_posts',
        'log1p_account_age_days',
    ]

    numeric_standard = [
        'username_length', 'username_digit_ratio', 'username_special_ratio',
        'fullname_words', 'fullname_length', 'fullname_digit_ratio',
        'name_equals_username', 'bio_length', 'bio_has_url',
        'bio_spam_keyword_count', 'bio_spam_keyword_hit',
        'followers', 'following', 'posts',
        'follower_following_ratio', 'following_follower_ratio', 'posts_per_follower',
        'followers_per_day_age', 'posts_per_day_age',
    ]

    rule_signal_cols = [
        'rule_low_ratio_very', 'rule_low_ratio_mod',
        'rule_no_posts_with_followers', 'rule_extreme_posts_per_follower',
        'rule_bio_spam_keyword', 'rule_no_profile_pic',
        'rule_young_account', 'rule_very_young_account',
        'rule_very_low_followers', 'rule_very_high_following',
        'rule_not_verified',
    ]

    binary_profile_cols = [
        'has_profile_pic', 'is_verified', 'is_private', 'has_external_url',
    ]

    numeric_skewed_present = [c for c in numeric_skewed]
    numeric_standard_present = [c for c in numeric_standard]
    rule_signal_present = [c for c in rule_signal_cols]
    binary_profile_present = [c for c in binary_profile_cols]
    if not include_optional_features:
        binary_profile_present = [c for c in binary_profile_cols if c in ('has_profile_pic',)]

    transformers = []

    if numeric_standard_present:
        scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        transformers.append(
            ('numeric_standard', Pipeline([
                ('select', ColumnSelector(numeric_standard_present)),
                ('impute', SimpleImputer(strategy='median', add_indicator=True)),
                ('scale', scaler),
            ]), numeric_standard_present)
        )

    if numeric_skewed_present:
        scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        transformers.append(
            ('numeric_skewed', Pipeline([
                ('select', ColumnSelector(numeric_skewed_present)),
                ('impute', SimpleImputer(strategy='median', add_indicator=True)),
                ('scale', scaler),
            ]), numeric_skewed_present)
        )

    if rule_signal_present:
        transformers.append(
            ('rule_signals', Pipeline([
                ('select', ColumnSelector(rule_signal_present)),
                ('impute', SimpleImputer(strategy='most_frequent', add_indicator=True)),
            ]), rule_signal_present)
        )

    if binary_profile_present and strategy_profile != 'exclude':
        transformers.append(
            ('profile_flags', Pipeline([
                ('select', ColumnSelector(binary_profile_present)),
                ('impute', SimpleImputer(strategy='most_frequent', add_indicator=True)),
            ]), binary_profile_present)
        )

    preprocessor = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
    return preprocessor


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer,
                                          input_columns: list,
                                          X_ref: Optional[pd.DataFrame] = None) -> list:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'remainder':
            continue
        if isinstance(trans, Pipeline):
            imputer = None
            for step_name, step in trans.steps:
                if isinstance(step, SimpleImputer):
                    imputer = step
            base_cols = list(cols)
            kept_base = list(base_cols)
            if imputer is not None and hasattr(imputer, 'statistics_'):
                stats = np.asarray(imputer.statistics_)
                kept_mask = ~np.isnan(stats)
                kept_base = [c for c, keep in zip(base_cols, kept_mask) if keep]
            names.extend(kept_base)
            if imputer is not None and getattr(imputer, 'indicator_', None) is not None:
                for missing_idx in imputer.indicator_.features_:
                    orig_pos = int(missing_idx)
                    if orig_pos < len(kept_base):
                        names.append(f'missing_{kept_base[orig_pos]}')
                    else:
                        names.append(f'missing_{name}_col_{orig_pos}')
        else:
            names.extend(list(cols))
    if X_ref is not None:
        try:
            out = preprocessor.transform(X_ref)
            actual_n = out.shape[1]
            while len(names) < actual_n:
                names.append(f'f{len(names)}')
            if len(names) > actual_n:
                names = names[:actual_n]
        except Exception:
            pass
    return names
