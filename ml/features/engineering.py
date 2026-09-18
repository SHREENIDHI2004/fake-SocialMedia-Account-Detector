import re
import math
import pandas as pd
import numpy as np


SPAM_KEYWORDS = [
    'giveaway', 'win', 'free', 'click here', 'follow for follow',
    'make money', 'easy money', 'get rich', 'spam', 'bot',
    'f4f', 'l4l', 'follow me', 'dm me', 'link in bio',
    'click link', 'follow back', 'instant follow', 'get followers',
    'buy followers', 'crypto giveaway', 'airdrop', 'investment',
    'earn money', 'guaranteed', 'winner', 'prize',
]


def _safe_int(v, default: int = 0) -> int:
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, np.integer)):
        try:
            if math.isnan(float(v)):
                return default
        except Exception:
            pass
        return int(v)
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return default
        return int(v)
    s = str(v).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def _safe_div(a, b, eps: float = 1e-6):
    if b is None:
        return 0.0
    b = float(b)
    if math.isnan(b) or abs(b) < eps:
        return 0.0
    return float(a) / b


def _digit_ratio(s) -> float:
    if s is None:
        return 0.0
    s = str(s)
    if len(s) == 0:
        return 0.0
    digits = sum(1 for c in s if c.isdigit())
    return digits / len(s)


def _special_char_ratio(s) -> float:
    if s is None:
        return 0.0
    s = str(s)
    if len(s) == 0:
        return 0.0
    special = sum(1 for c in s if not c.isalnum() and not c.isspace())
    return special / len(s)


def compute_text_features(username: str = '', display_name: str = '', bio: str = '') -> dict:
    username = '' if username is None else str(username).strip()
    display_name = '' if display_name is None else str(display_name).strip()
    bio = '' if bio is None else str(bio).strip()

    username_lower = username.lower()
    display_name_lower = display_name.lower()

    features = {
        'username_length': len(username),
        'username_digit_ratio': _digit_ratio(username),
        'username_special_ratio': _special_char_ratio(username),
        'fullname_words': len([w for w in re.split(r'\s+', display_name) if w]) if display_name else 0,
        'fullname_length': len(display_name),
        'fullname_digit_ratio': _digit_ratio(display_name),
        'name_equals_username': 1 if (username and username_lower == display_name_lower) else 0,
        'bio_length': len(bio),
        'bio_has_url': 1 if re.search(r'https?://|www\.', bio) is not None else 0,
        'bio_spam_keyword_count': sum(1 for kw in SPAM_KEYWORDS if kw in bio.lower()),
        'bio_spam_keyword_hit': 1 if any(kw in bio.lower() for kw in SPAM_KEYWORDS) else 0,
    }
    return features


def compute_ratio_features(followers: int, following: int, posts: int,
                           account_age_days: int = None) -> dict:
    followers = _safe_int(followers, 0)
    following = _safe_int(following, 0)
    posts = _safe_int(posts, 0)

    followers = max(0, followers)
    following = max(0, following)
    posts = max(0, posts)

    ffr = _safe_div(followers, following)
    ffr_inv = _safe_div(following, followers)
    ppf = _safe_div(posts, followers)

    feats = {
        'followers': followers,
        'following': following,
        'posts': posts,
        'follower_following_ratio': ffr,
        'following_follower_ratio': ffr_inv,
        'posts_per_follower': ppf,
        'log1p_followers': math.log1p(followers),
        'log1p_following': math.log1p(following),
        'log1p_posts': math.log1p(posts),
        'account_age_days': np.nan,
        'followers_per_day_age': np.nan,
        'posts_per_day_age': np.nan,
        'log1p_account_age_days': np.nan,
    }

    if account_age_days is not None and not (isinstance(account_age_days, float) and math.isnan(account_age_days)):
        age = max(0, int(account_age_days))
        age_clamped = max(age, 1)
        feats.update({
            'account_age_days': age,
            'followers_per_day_age': _safe_div(followers, age_clamped),
            'posts_per_day_age': _safe_div(posts, age_clamped),
            'log1p_account_age_days': math.log1p(age),
        })

    return feats


def compute_profile_flags(has_profile_pic: bool, is_verified: bool,
                          is_private: bool = None, has_external_url: bool = None,
                          bio: str = None) -> dict:
    flags = {
        'has_profile_pic': int(bool(has_profile_pic)) if has_profile_pic is not None else np.nan,
        'is_verified': int(bool(is_verified)) if is_verified is not None else np.nan,
    }
    if is_private is not None:
        flags['is_private'] = int(bool(is_private))
    else:
        flags['is_private'] = np.nan

    if has_external_url is not None:
        flags['has_external_url'] = int(bool(has_external_url))
    else:
        bio_has_url = None
        if bio:
            bio_has_url = 1 if re.search(r'https?://|www\.', str(bio)) is not None else 0
        flags['has_external_url'] = bio_has_url if bio_has_url is not None else np.nan

    return flags


def compute_rule_signals(followers: int = 0, following: int = 0, posts: int = 0,
                         has_profile_pic: bool = False, account_age_days: int = None,
                         bio: str = '', is_verified: bool = False) -> dict:
    followers = _safe_int(followers, 0)
    following = _safe_int(following, 0)
    posts = _safe_int(posts, 0)
    bio = '' if bio is None else str(bio).lower()
    bio_spam_hit = any(kw in bio for kw in SPAM_KEYWORDS)

    ffr = _safe_div(followers, following) if following > 0 else float('inf')

    signals = {}
    signals['rule_low_ratio_very'] = 1 if (following > 0 and ffr < 0.1) else 0
    signals['rule_low_ratio_mod'] = 1 if (following > 0 and 0.1 <= ffr < 0.5) else 0
    signals['rule_no_posts_with_followers'] = 1 if (posts == 0 and followers > 0) else 0
    signals['rule_extreme_posts_per_follower'] = 1 if (followers > 0 and _safe_div(posts, followers) > 10) else 0
    signals['rule_bio_spam_keyword'] = 1 if bio_spam_hit else 0
    signals['rule_no_profile_pic'] = 0 if has_profile_pic else 1
    signals['rule_young_account'] = 0
    signals['rule_very_young_account'] = 0
    if account_age_days is not None and not (isinstance(account_age_days, float) and math.isnan(account_age_days)):
        age = int(account_age_days)
        if age < 30:
            signals['rule_young_account'] = 1
        if age < 7:
            signals['rule_very_young_account'] = 1
    signals['rule_very_low_followers'] = 1 if followers < 10 else 0
    signals['rule_very_high_following'] = 1 if following > 3000 else 0
    signals['rule_not_verified'] = 0 if is_verified else 1
    return signals


def build_account_features(username: str = '', display_name: str = '', bio: str = '',
                           followers: int = 0, following: int = 0, posts: int = 0,
                           has_profile_pic: bool = False, account_age_days: int = None,
                           is_verified: bool = None, is_private: bool = None,
                           has_external_url: bool = None) -> dict:
    feats = {}
    feats.update(compute_text_features(username=username, display_name=display_name, bio=bio))
    feats.update(compute_ratio_features(followers=followers, following=following,
                                        posts=posts, account_age_days=account_age_days))
    feats.update(compute_profile_flags(has_profile_pic=has_profile_pic,
                                       is_verified=is_verified,
                                       is_private=is_private,
                                       has_external_url=has_external_url,
                                       bio=bio))
    feats.update(compute_rule_signals(followers=followers, following=following, posts=posts,
                                      has_profile_pic=has_profile_pic,
                                      account_age_days=account_age_days, bio=bio,
                                      is_verified=bool(is_verified) if is_verified is not None else False))
    return feats


def build_feature_frame(records: list) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = build_account_features(
            username=rec.get('username', ''),
            display_name=rec.get('display_name') or rec.get('name') or rec.get('fullname') or rec.get('username', ''),
            bio=rec.get('bio', '') or rec.get('description', ''),
            followers=rec.get('followers', 0) or rec.get('#followers', 0),
            following=rec.get('following', 0) or rec.get('#follows', 0),
            posts=rec.get('posts', 0) or rec.get('#posts', 0),
            has_profile_pic=rec.get('has_profile_pic', False) if rec.get('has_profile_pic') is not None else bool(rec.get('profile pic', 0)),
            account_age_days=rec.get('account_age_days'),
            is_verified=rec.get('is_verified'),
            is_private=rec.get('is_private') if rec.get('is_private') is not None else (int(rec['private']) if 'private' in rec and rec['private'] is not None else None),
            has_external_url=rec.get('has_external_url') if rec.get('has_external_url') is not None else (int(rec['external URL']) if 'external URL' in rec and rec['external URL'] is not None else None),
        )
        if 'is_fake' in rec:
            row['is_fake'] = rec['is_fake']
        if 'data_source' in rec:
            row['data_source'] = rec['data_source']
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


NUMERIC_FEATURES = [
    'username_length', 'username_digit_ratio', 'username_special_ratio',
    'fullname_words', 'fullname_length', 'fullname_digit_ratio',
    'name_equals_username', 'bio_length', 'bio_has_url',
    'bio_spam_keyword_count', 'bio_spam_keyword_hit',
    'followers', 'following', 'posts',
    'follower_following_ratio', 'following_follower_ratio', 'posts_per_follower',
    'log1p_followers', 'log1p_following', 'log1p_posts',
    'followers_per_day_age', 'posts_per_day_age', 'log1p_account_age_days',
    'rule_low_ratio_very', 'rule_low_ratio_mod',
    'rule_no_posts_with_followers', 'rule_extreme_posts_per_follower',
    'rule_bio_spam_keyword', 'rule_no_profile_pic',
    'rule_young_account', 'rule_very_young_account',
    'rule_very_low_followers', 'rule_very_high_following',
    'rule_not_verified',
]

BOOLEAN_OR_UNKNOWN_FEATURES = [
    'has_profile_pic', 'is_verified', 'is_private', 'has_external_url',
    'account_age_days',
]
