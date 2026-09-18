import sys
import os
import math
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.features.engineering import (
    SPAM_KEYWORDS,
    compute_text_features,
    compute_ratio_features,
    compute_profile_flags,
    compute_rule_signals,
    build_account_features,
    build_feature_frame,
)


class TestTextFeatures:
    def test_empty_strings_return_zeroes(self):
        f = compute_text_features(username="", display_name="", bio="")
        assert f["username_length"] == 0
        assert f["username_digit_ratio"] == 0.0
        assert f["fullname_words"] == 0
        assert f["bio_length"] == 0
        assert f["bio_spam_keyword_hit"] == 0

    def test_none_inputs_are_coerced(self):
        f = compute_text_features(username=None, display_name=None, bio=None)
        assert f["username_length"] == 0
        assert f["bio_spam_keyword_hit"] == 0

    def test_username_digit_ratio_computed(self):
        f = compute_text_features(username="user1234", display_name="", bio="")
        assert 0.5 <= f["username_digit_ratio"] <= 0.6

    def test_bio_url_detected(self):
        f = compute_text_features(
            username="u", display_name="d", bio="Check https://example.com for more"
        )
        assert f["bio_has_url"] == 1

    def test_bio_url_absent(self):
        f = compute_text_features(username="u", display_name="d", bio="No links here")
        assert f["bio_has_url"] == 0

    def test_spam_keyword_hit_present(self):
        f = compute_text_features(
            username="u", display_name="d", bio="win free giveaway click here"
        )
        assert f["bio_spam_keyword_hit"] == 1
        assert f["bio_spam_keyword_count"] >= 3

    def test_spam_keyword_hit_absent(self):
        f = compute_text_features(username="u", display_name="d", bio="hello world")
        assert f["bio_spam_keyword_hit"] == 0
        assert f["bio_spam_keyword_count"] == 0

    def test_name_equals_username_flag(self):
        f = compute_text_features(
            username="johnsmith", display_name="JohnSmith", bio=""
        )
        assert f["name_equals_username"] == 1

    def test_name_not_equals_username(self):
        f = compute_text_features(
            username="js123", display_name="John Smith", bio=""
        )
        assert f["name_equals_username"] == 0


class TestRatioFeatures:
    def test_basic_ratio_computation(self):
        f = compute_ratio_features(followers=100, following=200, posts=50)
        assert abs(f["follower_following_ratio"] - 0.5) < 1e-3
        assert f["log1p_followers"] == math.log1p(100)

    def test_zero_following_avoids_div_by_zero(self):
        f = compute_ratio_features(followers=100, following=0, posts=0)
        assert not math.isnan(f["follower_following_ratio"])
        assert not math.isinf(f["follower_following_ratio"])

    def test_none_inputs_default_to_zero(self):
        f = compute_ratio_features(followers=None, following=None, posts=None)
        assert f["followers"] == 0
        assert f["following"] == 0
        assert f["posts"] == 0

    def test_negative_inputs_clamped(self):
        f = compute_ratio_features(followers=-10, following=-5, posts=-1)
        assert f["followers"] == 0
        assert f["following"] == 0
        assert f["posts"] == 0

    def test_account_age_features_produced(self):
        f = compute_ratio_features(
            followers=100, following=50, posts=10, account_age_days=30
        )
        assert f["account_age_days"] == 30
        assert not math.isnan(f["followers_per_day_age"])
        assert not math.isnan(f["log1p_account_age_days"])

    def test_account_age_none_features_nan(self):
        f = compute_ratio_features(followers=100, following=50, posts=10)
        assert math.isnan(f["account_age_days"])
        assert math.isnan(f["followers_per_day_age"])


class TestProfileFlags:
    def test_all_flags_set(self):
        f = compute_profile_flags(
            has_profile_pic=True,
            is_verified=True,
            is_private=True,
            has_external_url=True,
        )
        assert f["has_profile_pic"] == 1
        assert f["is_verified"] == 1
        assert f["is_private"] == 1
        assert f["has_external_url"] == 1

    def test_all_false(self):
        f = compute_profile_flags(
            has_profile_pic=False,
            is_verified=False,
            is_private=False,
            has_external_url=False,
        )
        assert f["has_profile_pic"] == 0
        assert f["is_verified"] == 0
        assert f["is_private"] == 0
        assert f["has_external_url"] == 0

    def test_optional_none_becomes_nan(self):
        f = compute_profile_flags(
            has_profile_pic=True, is_verified=False,
            is_private=None, has_external_url=None,
        )
        assert f["has_profile_pic"] == 1
        assert math.isnan(f["is_private"])
        assert math.isnan(f["has_external_url"])


class TestRuleSignals:
    def test_low_ratio_very_triggered(self):
        s = compute_rule_signals(followers=5, following=200, posts=0)
        assert s["rule_low_ratio_very"] == 1

    def test_low_ratio_mod_triggered(self):
        s = compute_rule_signals(followers=60, following=200, posts=0)
        assert s["rule_low_ratio_very"] == 0
        assert s["rule_low_ratio_mod"] == 1

    def test_no_posts_with_followers(self):
        s = compute_rule_signals(followers=100, following=50, posts=0)
        assert s["rule_no_posts_with_followers"] == 1

    def test_no_profile_pic_flag(self):
        s = compute_rule_signals(
            followers=0, following=0, posts=0, has_profile_pic=False
        )
        assert s["rule_no_profile_pic"] == 1

    def test_has_profile_pic_no_flag(self):
        s = compute_rule_signals(
            followers=0, following=0, posts=0, has_profile_pic=True
        )
        assert s["rule_no_profile_pic"] == 0

    def test_very_young_account(self):
        s = compute_rule_signals(followers=0, following=0, posts=0, account_age_days=3)
        assert s["rule_very_young_account"] == 1
        assert s["rule_young_account"] == 1

    def test_young_account_not_very_young(self):
        s = compute_rule_signals(followers=0, following=0, posts=0, account_age_days=15)
        assert s["rule_very_young_account"] == 0
        assert s["rule_young_account"] == 1

    def test_old_account_no_age_flags(self):
        s = compute_rule_signals(followers=0, following=0, posts=0, account_age_days=365)
        assert s["rule_very_young_account"] == 0
        assert s["rule_young_account"] == 0

    def test_age_none_no_crash(self):
        s = compute_rule_signals(followers=0, following=0, posts=0)
        assert s["rule_very_young_account"] == 0
        assert s["rule_young_account"] == 0

    def test_very_high_following(self):
        s = compute_rule_signals(followers=0, following=5000, posts=0)
        assert s["rule_very_high_following"] == 1

    def test_bio_spam_keyword(self):
        s = compute_rule_signals(followers=0, following=0, posts=0, bio="giveaway win free")
        assert s["rule_bio_spam_keyword"] == 1


class TestBuildAccountFeatures:
    def test_returns_expected_keys(self):
        f = build_account_features(
            username="user", display_name="User Name", bio="hello",
            followers=10, following=20, posts=5, has_profile_pic=True,
        )
        assert "follower_following_ratio" in f
        assert "username_digit_ratio" in f
        assert "bio_spam_keyword_hit" in f
        assert "rule_no_profile_pic" in f

    def test_bijective_input_output_non_null_ratios(self):
        f = build_account_features(
            username="alice", display_name="Alice Wonderland",
            bio="Travel photos and daily life", followers=5000, following=500,
            posts=200, has_profile_pic=True, account_age_days=730,
            is_verified=True, is_private=False, has_external_url=True,
        )
        assert f["follower_following_ratio"] > 1.0
        assert f["is_verified"] == 1
        assert f["log1p_account_age_days"] > 0


class TestBuildFeatureFrame:
    def test_empty_list_returns_df(self):
        df = build_feature_frame([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_single_record(self):
        df = build_feature_frame([{
            "username": "x", "fullname": "Y", "bio": "hi",
            "followers": 10, "following": 20, "posts": 0,
        }])
        assert len(df) == 1
        assert "follower_following_ratio" in df.columns

    def test_multiple_records(self):
        recs = [
            {"username": "a", "followers": 100, "following": 50},
            {"username": "b", "followers": 5, "following": 5000},
        ]
        df = build_feature_frame(recs)
        assert len(df) == 2

    def test_alternate_legacy_column_names(self):
        df = build_feature_frame([{
            "profile pic": 1, "external URL": 0, "private": 0,
            "#followers": 10, "#follows": 20, "#posts": 5,
        }])
        assert len(df) == 1
        assert df["followers"].iloc[0] == 10
