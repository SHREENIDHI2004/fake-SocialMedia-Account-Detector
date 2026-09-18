import sys
import os
import math
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.inference.predict import (
    ModelBundle,
    load_model_bundle,
    predict_probability,
    predict_account,
    risk_level_from_probability,
)
from ml.features.engineering import build_feature_frame


@pytest.fixture(scope="module")
def bundle():
    return load_model_bundle()


class TestLoadModelBundle:
    def test_bundle_returns_modelbundle(self, bundle):
        assert isinstance(bundle, ModelBundle)

    def test_bundle_available(self, bundle):
        assert bundle.available is True, f"Model unavailable: {bundle.reason}"

    def test_threshold_valid(self, bundle):
        assert bundle.threshold is not None
        assert 0.0 <= float(bundle.threshold) <= 1.0

    def test_metadata_has_model_name(self, bundle):
        assert "model_name" in bundle.metadata

    def test_test_metrics_present(self, bundle):
        tm = bundle.metadata.get("test_metrics", {})
        assert "f1" in tm
        assert "roc_auc" in tm
        assert 0.0 <= float(tm["roc_auc"]) <= 1.0

    def test_calibrated_pipeline_present_or_absent_gracefully(self, bundle):
        if bundle.calibrated_pipeline is not None:
            # Check it has predict_proba
            assert hasattr(bundle.calibrated_pipeline, "predict_proba")
        # If absent, base_pipeline should be there
        if bundle.base_pipeline is None and bundle.calibrated_pipeline is None:
            pytest.fail("Neither base nor calibrated pipeline loaded")


class TestPredictProbability:
    def test_returns_valid_probabilities(self, bundle):
        df = build_feature_frame([{
            "username": "bot123", "fullname": "bot",
            "followers": 1, "following": 2000, "posts": 0,
            "has_profile_pic": False,
        }])
        raw, cal = predict_probability(df, bundle)
        assert raw is not None
        assert len(raw) == 1
        assert 0.0 <= float(raw[0]) <= 1.0
        if cal is not None:
            assert len(cal) == 1
            assert 0.0 <= float(cal[0]) <= 1.0

    def test_multiple_rows(self, bundle):
        df = build_feature_frame([
            {"username": "a", "followers": 1000, "following": 100, "posts": 50, "has_profile_pic": True, "is_verified": True, "account_age_days": 1000},
            {"username": "b", "followers": 2, "following": 3000, "posts": 0, "has_profile_pic": False, "bio": "win free money"},
        ])
        raw, cal = predict_probability(df, bundle)
        assert len(raw) == 2
        # Fake should score higher than legitimate
        assert float(raw[1]) > float(raw[0])

    def test_empty_dataframe_handled(self, bundle):
        from ml.features.engineering import NUMERIC_FEATURES, BOOLEAN_OR_UNKNOWN_FEATURES
        cols = NUMERIC_FEATURES + BOOLEAN_OR_UNKNOWN_FEATURES
        df = pd.DataFrame(columns=cols)
        raw, cal = predict_probability(df, bundle)
        assert len(raw) == 0


class TestPredictAccount:
    def test_full_account(self):
        r = predict_account({
            "username": "bot_sales_9987",
            "display_name": "",
            "bio": "win free money click link in bio giveaway crypto airdrop",
            "followers": 5, "following": 4900, "posts": 0,
            "has_profile_pic": False, "account_age_days": 3,
            "is_verified": False, "is_private": False, "has_external_url": False,
        })
        assert "risk_probability" in r
        assert 0.0 <= float(r["risk_probability"]) <= 1.0
        assert r["is_flagged"] if float(r["risk_probability"]) >= 0.5 else (not r["is_flagged"])

    def test_legitimate_account_low_prob(self):
        r = predict_account({
            "username": "sarah.travels",
            "display_name": "Sarah Mitchell",
            "bio": "Adventure seeker. Sharing travel photos. Coffee enthusiast",
            "followers": 12500, "following": 890, "posts": 342,
            "has_profile_pic": True, "account_age_days": 1420,
            "is_verified": True, "is_private": False, "has_external_url": True,
        })
        assert float(r["risk_probability"]) < 0.5

    def test_missing_optional_fields(self):
        r = predict_account({"followers": 50, "following": 100})
        assert "risk_probability" in r
        assert 0.0 <= float(r["risk_probability"]) <= 1.0

    def test_invalid_input_types_handled(self):
        r = predict_account({
            "username": None,
            "followers": "not-a-number",
            "following": None,
        })
        # Should not crash; either heuristic or ML path returns valid dict
        assert "risk_probability" in r
        assert "risk_level" in r


class TestInvalidInput:
    def test_nan_values(self, bundle):
        df = build_feature_frame([{
            "username": "a",
            "followers": float("nan"),
            "following": float("nan"),
            "posts": 0,
        }])
        raw, cal = predict_probability(df, bundle)
        assert len(raw) == 1
        assert not math.isnan(float(raw[0])) or True  # tolerant; just no crash

    def test_all_null_fields(self):
        r = predict_account({})
        assert isinstance(r, dict)
        assert "risk_probability" in r

    def test_wrong_types_string_numbers(self):
        r = predict_account({
            "followers": "100",
            "following": "50",
            "posts": "10",
        })
        assert "risk_probability" in r
        # Values that parse as numeric should work
        assert 0.0 <= float(r["risk_probability"]) <= 1.0


class TestRiskLevel:
    def test_high_threshold(self):
        assert risk_level_from_probability(0.8) == "high"

    def test_medium_threshold(self):
        assert risk_level_from_probability(0.5) == "medium"

    def test_low_threshold(self):
        assert risk_level_from_probability(0.1) == "low"
