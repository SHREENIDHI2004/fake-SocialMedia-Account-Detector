import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
KAGGLE_DIR = DATA_DIR / 'kaggle_bakhshandeh'


def rename_kaggle_columns(df: pd.DataFrame) -> pd.DataFrame:
    renames = {
        'profile pic': 'has_profile_pic',
        'nums/length username': 'username_digit_ratio',
        'fullname words': 'fullname_words',
        'nums/length fullname': 'fullname_digit_ratio',
        'name==username': 'name_equals_username',
        'description length': 'bio_length',
        'external URL': 'has_external_url',
        'private': 'is_private',
        '#posts': 'posts',
        '#followers': 'followers',
        '#follows': 'following',
        'fake': 'is_fake',
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    return df


def _standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['has_profile_pic', 'name_equals_username', 'has_external_url',
                'is_private', 'is_fake']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    for col in ['username_digit_ratio', 'fullname_digit_ratio']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['fullname_words', 'bio_length', 'posts', 'followers', 'following']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df


def _add_dataset_columns(df: pd.DataFrame, data_source: str, split: str) -> pd.DataFrame:
    df = df.copy()
    df['data_source'] = data_source
    df['dataset_split'] = split
    if 'username' not in df.columns:
        df.insert(0, 'username', [f'{split}_{i}' for i in range(len(df))])
    if 'display_name' not in df.columns:
        df['display_name'] = df['username']
    if 'bio' not in df.columns:
        df['bio'] = ''
    if 'account_age_days' not in df.columns:
        df['account_age_days'] = pd.NA
    if 'is_verified' not in df.columns:
        df['is_verified'] = pd.NA
    return df


def check_account_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
    if 'username' not in train_df.columns or 'username' not in test_df.columns:
        return 0
    train_usernames = set(str(u).lower() for u in train_df['username'].dropna().unique())
    test_usernames = set(str(u).lower() for u in test_df['username'].dropna().unique())
    overlap = len(train_usernames & test_usernames)
    return overlap


def load_existing_samples() -> pd.DataFrame:
    path = DATA_DIR / 'existing_samples.csv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['data_source'] = 'existing_samples'
    df['dataset_split'] = 'integration_test_only'
    df = df.rename(columns={
        'has_profile_pic': 'has_profile_pic',
        'account_age_days': 'account_age_days',
        'is_verified': 'is_verified',
    })
    if 'display_name' not in df.columns:
        df['display_name'] = df['username']
    return df


def load_influencers_reference() -> pd.DataFrame:
    path = DATA_DIR / 'influencers_reference.csv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    def parse_followers(val):
        if pd.isna(val):
            return 0
        s = str(val).strip().upper()
        try:
            if 'M' in s:
                return int(float(s.replace('M', '')) * 1_000_000)
            if 'K' in s:
                return int(float(s.replace('K', '')) * 1_000)
            return int(float(s))
        except Exception:
            return 0

    df['followers'] = df['followers'].apply(parse_followers)
    df['data_source'] = 'influencers_reference'
    df['dataset_split'] = 'post_training_sanity_only'
    if 'username' not in df.columns:
        df['username'] = df['name'].astype(str)
    if 'display_name' not in df.columns:
        df['display_name'] = df['instagram name'].fillna(df['name']).astype(str)
    if 'bio' not in df.columns:
        parts = [df['Category_1'].fillna(''), df['Category_2'].fillna(''), df['country'].fillna('')]
        df['bio'] = [' | '.join([str(p) for p in row if str(p)]) for row in zip(*parts)]
    if 'posts' not in df.columns:
        df['posts'] = pd.NA
    if 'following' not in df.columns:
        df['following'] = pd.NA
    if 'has_profile_pic' not in df.columns:
        df['has_profile_pic'] = 1
    if 'is_verified' not in df.columns:
        df['is_verified'] = (df['followers'] >= 1_000_000).astype(int)
        df.loc[df['followers'] < 1_000_000, 'is_verified'] = pd.NA
    if 'account_age_days' not in df.columns:
        df['account_age_days'] = pd.NA
    return df


def load_kaggle_split(path_csv: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    df = rename_kaggle_columns(df)
    df = _standardize_dtypes(df)
    df = _add_dataset_columns(df, data_source='kaggle_bakhshandeh', split=split)
    return df


def load_primary_dataset():
    train_path = KAGGLE_DIR / 'train.csv'
    test_path = KAGGLE_DIR / 'test.csv'

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    available = False

    if train_path.exists() and test_path.exists():
        train_df = load_kaggle_split(train_path, 'train')
        test_df = load_kaggle_split(test_path, 'test')
        available = True

    overlap = check_account_overlap(train_df, test_df)

    info = {
        'available': available,
        'kaggle_dir': str(KAGGLE_DIR),
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_test_username_overlap': overlap,
        'train_class_distribution': train_df['is_fake'].value_counts().to_dict() if available and 'is_fake' in train_df.columns else {},
        'test_class_distribution': test_df['is_fake'].value_counts().to_dict() if available and 'is_fake' in test_df.columns else {},
        'duplicates_train': int(train_df.duplicated().sum()) if not train_df.empty else 0,
        'duplicates_test': int(test_df.duplicated().sum()) if not test_df.empty else 0,
        'missing_values_train': train_df.isnull().sum().to_dict() if not train_df.empty else {},
        'missing_values_test': test_df.isnull().sum().to_dict() if not test_df.empty else {},
    }
    return train_df, test_df, info


def summarize_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("DATASET SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Train rows: {len(train_df)}")
    lines.append(f"Test rows:  {len(test_df)}")
    if 'is_fake' in train_df.columns:
        lines.append("Train class distribution:")
        vc = train_df['is_fake'].value_counts(dropna=False)
        for k, v in vc.items():
            pct = 100.0 * v / len(train_df) if len(train_df) else 0
            lines.append(f"  is_fake={k}: {v} ({pct:.1f}%)")
    if 'is_fake' in test_df.columns:
        lines.append("Test class distribution:")
        vc = test_df['is_fake'].value_counts(dropna=False)
        for k, v in vc.items():
            pct = 100.0 * v / len(test_df) if len(test_df) else 0
            lines.append(f"  is_fake={k}: {v} ({pct:.1f}%)")
    lines.append(f"Duplicates in train: {int(train_df.duplicated().sum())}")
    lines.append(f"Duplicates in test:  {int(test_df.duplicated().sum())}")
    overlap = check_account_overlap(train_df, test_df)
    lines.append(f"Username overlap (train intersect test): {overlap}")
    lines.append("Missing values (train):")
    ms = train_df.isnull().sum()
    for col in train_df.columns:
        if ms[col] > 0:
            lines.append(f"  {col}: {ms[col]}")
    return "\n".join(lines)
