import sys
import os
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.preprocessing.pipeline import (
    ColumnSelector,
    Log1pClip,
    build_preprocessing_pipeline,
    get_feature_names_after_preprocessing,
)
from ml.features.engineering import build_feature_frame


@pytest.fixture(scope="module")
def sample_df():
    recs = [
        {
            "username": f"user_{i}",
            "fullname": f"User {i}",
            "bio": f"Bio text for user {i} with different length content",
            "followers": int(10 ** (i / 5)) if i < 15 else int(5 * (i + 1)),
            "following": 200 - i,
            "posts": i * 3 + 1,
            "has_profile_pic": i % 2 == 0,
            "is_verified": i % 5 == 0,
            "is_private": False,
            "has_external_url": i % 3 == 0,
            "account_age_days": 30 + i * 10,
        }
        for i in range(20)
    ]
    return build_feature_frame(recs)


class TestColumnSelector:
    def test_select_present_cols(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        sel = ColumnSelector(["a", "b"])
        out = sel.fit_transform(df)
        assert list(out.columns) == ["a", "b"]

    def test_select_missing_cols_filled_nan(self):
        df = pd.DataFrame({"a": [1, 2]})
        sel = ColumnSelector(["a", "z"])
        out = sel.fit_transform(df)
        assert "z" in out.columns
        assert out["z"].isna().all()


class TestLog1pClip:
    def test_log1p_transform_shape(self):
        df = pd.DataFrame({"x": [0.0, 1.0, 100.0]})
        t = Log1pClip(["x"], clip_percentile=1.0)
        out = t.fit_transform(df)
        vals = out["x"].values
        assert abs(vals[0] - 0.0) < 1e-6
        assert abs(vals[1] - np.log1p(1.0)) < 1e-6

    def test_clip_is_applied(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 1000]})
        t = Log1pClip(["x"], clip_percentile=0.8)
        out = t.fit_transform(df)
        max_val = float(out["x"].max())
        assert max_val < np.log1p(1000)


class TestBuildPreprocessingPipeline:
    def test_pipeline_fit_transform_shape(self, sample_df):
        pipe = build_preprocessing_pipeline()
        out = pipe.fit_transform(sample_df)
        assert out.shape[0] == len(sample_df)
        assert out.shape[1] > 0

    def test_pipeline_no_optional_profile(self, sample_df):
        pipe = build_preprocessing_pipeline(include_optional_features=False)
        out = pipe.fit_transform(sample_df)
        assert out.shape[0] == len(sample_df)

    def test_pipeline_scaler_variants(self, sample_df):
        pipe_robust = build_preprocessing_pipeline(use_robust_scaler=True)
        pipe_standard = build_preprocessing_pipeline(use_robust_scaler=False)
        out_r = pipe_robust.fit_transform(sample_df)
        out_s = pipe_standard.fit_transform(sample_df)
        assert out_r.shape[0] == out_s.shape[0]


class TestGetFeatureNamesAfterPreprocessing:
    def test_names_match_width(self, sample_df):
        pipe = build_preprocessing_pipeline()
        pipe.fit(sample_df)
        names = get_feature_names_after_preprocessing(
            pipe, list(sample_df.columns), X_ref=sample_df
        )
        out = pipe.transform(sample_df)
        assert len(names) == out.shape[1]

    def test_names_non_empty(self, sample_df):
        pipe = build_preprocessing_pipeline()
        pipe.fit(sample_df)
        names = get_feature_names_after_preprocessing(
            pipe, list(sample_df.columns), X_ref=sample_df
        )
        assert len([n for n in names if n]) > 0
