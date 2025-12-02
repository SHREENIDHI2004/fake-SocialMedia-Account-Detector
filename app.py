from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
import requests
import json

# Try to import config, if not available use defaults
try:
    from config import (
        INSTAGRAM_GRAPH_API, 
        INSTAGRAM_BASIC_DISPLAY, 
        INSTALOADER_CONFIG,
        FACEBOOK_GRAPH_API,
        API_METHOD,
        USE_REAL_API
    )
except ImportError:
    # Default configuration if config.py doesn't exist
    INSTAGRAM_GRAPH_API = {'access_token': '', 'user_id': ''}
    INSTAGRAM_BASIC_DISPLAY = {'client_id': '', 'client_secret': '', 'access_token': ''}
    INSTALOADER_CONFIG = {'username': '', 'password': '', 'use_session_file': True}
    FACEBOOK_GRAPH_API = {'access_token': '', 'app_id': '', 'app_secret': ''}
    API_METHOD = 'instaloader'
    USE_REAL_API = False

app = Flask(__name__)

# Sample dataset for demonstration (in production, use real APIs or larger datasets)
SAMPLE_DATASET = {
    'accounts': [
        {
            'username': 'fake_account_1',
            'platform': 'instagram',
            'followers': 25,
            'following': 4900,
            'posts': 2,
            'bio': 'Follow me for giveaway win free money click here',
            'has_profile_pic': False,
            'account_age_days': 5,
            'is_verified': False
        },
        {
            'username': 'genuine_user',
            'platform': 'instagram',
            'followers': 8500,
            'following': 450,
            'posts': 234,
            'bio': 'Photographer | Travel enthusiast | Coffee lover',
            'has_profile_pic': True,
            'account_age_days': 1200,
            'is_verified': False
        },
        {
            'username': 'spam_bot',
            'platform': 'twitter',
            'followers': 10,
            'following': 5000,
            'posts': 15000,
            'bio': 'Make money fast! Click link! Follow for follow!',
            'has_profile_pic': False,
            'account_age_days': 30,
            'is_verified': False
        }
    ]
}

def parse_followers(followers_str):
    """Parse followers string like '409.8M' to integer"""
    if pd.isna(followers_str) or not isinstance(followers_str, str):
        return 0
    
    followers_str = str(followers_str).strip().upper()
    if 'M' in followers_str:
        # Millions
        num = float(followers_str.replace('M', ''))
        return int(num * 1_000_000)
    elif 'K' in followers_str:
        # Thousands
        num = float(followers_str.replace('K', ''))
        return int(num * 1_000)
    else:
        # Try to parse as integer
        try:
            return int(float(followers_str))
        except:
            return 0

def load_dataset():
    """Load Instagram influencers dataset"""
    dataset_path = 'social media influencers-INSTAGRAM - -DEC 2022.csv'
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            # Parse followers column
            df['followers_parsed'] = df['followers'].apply(parse_followers)
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    return None

def check_spam_keywords(bio):
    """Check for spam keywords in bio"""
    spam_keywords = [
        'giveaway', 'win', 'free', 'click here', 'follow for follow',
        'make money', 'easy money', 'get rich', 'spam', 'bot',
        'f4f', 'l4l', 'follow me', 'dm me', 'link in bio spam'
    ]
    if not bio:
        return False
    bio_lower = bio.lower()
    return any(keyword in bio_lower for keyword in spam_keywords)

def calculate_fake_probability(account_data):
    """
    Calculate fake probability based on multiple factors
    Returns probability (0-100) and detected issues
    """
    issues = []
    risk_score = 0
    
    # Factor 1: Follower/Following Ratio
    if account_data.get('following', 0) > 0:
        ratio = account_data.get('followers', 0) / account_data.get('following', 1)
        if ratio < 0.1:  # Following way more than followers
            risk_score += 30
            issues.append(f"Very low follower-to-following ratio ({ratio:.2f})")
        elif ratio < 0.5:
            risk_score += 15
            issues.append(f"Low follower-to-following ratio ({ratio:.2f})")
    
    # Factor 2: Post count
    posts = account_data.get('posts', 0)
    followers = account_data.get('followers', 0)
    if posts == 0 and followers > 0:
        risk_score += 25
        issues.append("No posts but has followers")
    elif posts > 0 and followers > 0:
        posts_per_follower = posts / followers
        if posts_per_follower > 10:  # Suspiciously high post rate
            risk_score += 20
            issues.append("Extremely high post-to-follower ratio")
    
    # Factor 3: Spam keywords in bio
    bio = account_data.get('bio', '')
    if check_spam_keywords(bio):
        risk_score += 25
        issues.append("Bio contains spam keywords")
    
    # Factor 4: Profile picture
    if not account_data.get('has_profile_pic', False):
        risk_score += 15
        issues.append("No profile picture")
    
    # Factor 5: Account age
    account_age = account_data.get('account_age_days', 0)
    if account_age < 30:
        risk_score += 10
        if account_age < 7:
            risk_score += 10
            issues.append(f"Very new account ({account_age} days old)")
    
    # Factor 6: Follower count (too low or suspicious)
    if account_data.get('followers', 0) < 10:
        risk_score += 10
        issues.append("Very low follower count")
    
    # Factor 7: Following count (too high)
    if account_data.get('following', 0) > 3000:
        risk_score += 15
        issues.append(f"Following too many accounts ({account_data.get('following', 0)})")
    
    # Cap at 100
    fake_probability = min(risk_score, 100)
    
    # Determine suggested action
    if fake_probability >= 70:
        action = "Report or Verify"
    elif fake_probability >= 40:
        action = "Review Carefully"
    else:
        action = "Likely Genuine"
    
    return fake_probability, issues, action

def fetch_instagram_graph_api(username):
    """
    Fetch Instagram account data using Instagram Graph API
    Requires: Access token with instagram_basic or instagram_graph_user_profile permissions
    """
    try:
        access_token = INSTAGRAM_GRAPH_API.get('access_token', '').strip()
        if not access_token:
            return None
        
        # First, get the Instagram Business Account ID from username
        # Note: This requires the account to be linked to a Facebook Page
        url = f"https://graph.facebook.com/v18.0/{username}"
        params = {
            'fields': 'id,username,account_type',
            'access_token': access_token
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            user_data = response.json()
            user_id = user_data.get('id')
            
            if user_id:
                # Get detailed profile information
                profile_url = f"https://graph.facebook.com/v18.0/{user_id}"
                profile_params = {
                    'fields': 'username,biography,followers_count,follows_count,media_count,profile_picture_url,is_verified',
                    'access_token': access_token
                }
                
                profile_response = requests.get(profile_url, params=profile_params, timeout=10)
                
                if profile_response.status_code == 200:
                    profile = profile_response.json()
                    
                    # Calculate account age (if available in API response)
                    # Note: Account creation date might not be available in Graph API
                    account_age_days = 0
                    
                    return {
                        'username': profile.get('username', username),
                        'platform': 'instagram',
                        'followers': profile.get('followers_count', 0),
                        'following': profile.get('follows_count', 0),
                        'posts': profile.get('media_count', 0),
                        'bio': profile.get('biography', ''),
                        'has_profile_pic': profile.get('profile_picture_url') is not None,
                        'account_age_days': account_age_days,  # Not available in Graph API
                        'is_verified': profile.get('is_verified', False)
                    }
        
        return None
    except Exception as e:
        print(f"Instagram Graph API Error: {e}")
        return None

def fetch_instagram_instaloader(username):
    """
    Fetch Instagram account data using instaloader (scraping library)
    This doesn't require API keys, but may require login for private accounts
    """
    try:
        import instaloader
        
        # Create Instaloader instance
        L = instaloader.Instaloader()
        
        # Try to load session if available
        session_file = 'instagram_session'
        if INSTALOADER_CONFIG.get('use_session_file', True) and os.path.exists(session_file):
            try:
                L.load_session_from_file(INSTALOADER_CONFIG.get('username', ''), session_file)
            except:
                pass
        
        # Login if credentials are provided (optional, only needed for private accounts)
        if INSTALOADER_CONFIG.get('username') and INSTALOADER_CONFIG.get('password'):
            try:
                L.login(INSTALOADER_CONFIG['username'], INSTALOADER_CONFIG['password'])
                # Save session for future use
                if INSTALOADER_CONFIG.get('use_session_file', True):
                    L.save_session_to_file(session_file)
            except Exception as login_error:
                print(f"Instaloader login error (may still work for public accounts): {login_error}")
        
        # Fetch profile
        profile = instaloader.Profile.from_username(L.context, username)
        
        # Calculate account age
        account_age_days = 0
        try:
            # Try to get joined_date (may not be available for all profiles)
            if hasattr(profile, 'joined_date') and profile.joined_date:
                account_age_days = (datetime.now() - profile.joined_date).days
        except:
            # Account age not available
            pass
        
        return {
            'username': profile.username,
            'platform': 'instagram',
            'followers': profile.followers,
            'following': profile.followees,
            'posts': profile.mediacount,
            'bio': profile.biography or '',
            'has_profile_pic': profile.profile_pic_url is not None,
            'account_age_days': account_age_days,
            'is_verified': profile.is_verified
        }
    except Exception as e:
        error_str = str(e).lower()
        if 'does not exist' in error_str or 'profilenotexists' in error_str:
            print(f"Instagram profile '{username}' does not exist")
        elif 'private' in error_str or 'privatenotfollowed' in error_str:
            print(f"Instagram profile '{username}' is private and not followed")
        else:
            print(f"Instaloader Error: {e}")
        return None

def fetch_facebook_graph_api(username):
    """
    Fetch Facebook account data using Facebook Graph API
    Requires: Access token with public_profile, user_friends permissions
    Note: Facebook API has strict privacy policies - only public data available
    """
    try:
        access_token = FACEBOOK_GRAPH_API.get('access_token', '').strip()
        if not access_token:
            return None
        
        # Facebook Graph API - get user profile by username or user ID
        # Note: Facebook usernames are different from Instagram
        # You can use user ID or username (if available)
        
        # Try to get user ID first (if username is provided)
        url = f"https://graph.facebook.com/v18.0/{username}"
        params = {
            'fields': 'id,name,username,about,link,friends,posts',
            'access_token': access_token
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            user_data = response.json()
            
            # Get detailed information
            user_id = user_data.get('id') or username
            
            # Get friends count (if available)
            friends_count = 0
            if 'friends' in user_data:
                if isinstance(user_data['friends'], dict) and 'summary' in user_data['friends']:
                    friends_count = user_data['friends']['summary'].get('total_count', 0)
                elif isinstance(user_data['friends'], list):
                    friends_count = len(user_data['friends'])
            
            # Get posts count (if available)
            posts_count = 0
            if 'posts' in user_data:
                if isinstance(user_data['posts'], dict) and 'data' in user_data['posts']:
                    posts_count = len(user_data['posts']['data'])
                elif isinstance(user_data['posts'], list):
                    posts_count = len(user_data['posts'])
            
            # Try to get more detailed profile info
            profile_url = f"https://graph.facebook.com/v18.0/{user_id}"
            profile_params = {
                'fields': 'id,name,username,about,link,picture,verified,friends.limit(1).summary(true),posts.limit(1)',
                'access_token': access_token
            }
            
            profile_response = requests.get(profile_url, params=profile_params, timeout=10)
            
            if profile_response.status_code == 200:
                profile = profile_response.json()
                
                # Extract friends count from summary
                if 'friends' in profile and 'summary' in profile['friends']:
                    friends_count = profile['friends']['summary'].get('total_count', 0)
                
                # Try to get followers count (for Pages)
                followers_count = 0
                if 'fan_count' in profile:  # For Pages
                    followers_count = profile.get('fan_count', 0)
                
                # Account age - Facebook doesn't provide this directly
                account_age_days = 0
                
                # Check if profile picture exists
                has_profile_pic = profile.get('picture') is not None or 'picture' in profile
                
                return {
                    'username': profile.get('username') or profile.get('name', username),
                    'platform': 'facebook',
                    'followers': followers_count,  # May be 0 for personal profiles
                    'following': friends_count,  # Friends count as proxy
                    'posts': posts_count,  # Limited by API
                    'bio': profile.get('about', '') or profile.get('name', ''),
                    'has_profile_pic': has_profile_pic,
                    'account_age_days': account_age_days,  # Not available in Graph API
                    'is_verified': profile.get('verified', False)
                }
            else:
                # Fallback to basic data
                return {
                    'username': user_data.get('username') or user_data.get('name', username),
                    'platform': 'facebook',
                    'followers': 0,
                    'following': friends_count,
                    'posts': posts_count,
                    'bio': user_data.get('about', '') or user_data.get('name', ''),
                    'has_profile_pic': 'picture' in user_data,
                    'account_age_days': 0,
                    'is_verified': False
                }
        
        return None
    except Exception as e:
        print(f"Facebook Graph API Error: {e}")
        return None

def fetch_account_data(username, platform):
    """
    Fetch account data from real APIs, then fallback to dataset/mock data
    Priority: Real API > Dataset > Mock Data
    """
    # Check if username exists in sample dataset first
    for account in SAMPLE_DATASET['accounts']:
        if account['username'].lower() == username.lower() and account['platform'] == platform.lower():
            return account
    
    # Try real Instagram API if enabled and platform is Instagram
    if USE_REAL_API and platform.lower() == 'instagram':
        result = None
        
        # Try Graph API first if configured
        if API_METHOD == 'graph_api' and INSTAGRAM_GRAPH_API.get('access_token'):
            result = fetch_instagram_graph_api(username)
            if result:
                return result
        
        # Try Instaloader (works for public accounts without API keys)
        if API_METHOD == 'instaloader' or not result:
            try:
                result = fetch_instagram_instaloader(username)
                if result:
                    return result
            except ImportError:
                print("Instaloader not installed. Install with: pip install instaloader")
            except Exception as e:
                print(f"Instaloader failed: {e}")
    
    # Try real Facebook API if enabled and platform is Facebook
    if USE_REAL_API and platform.lower() == 'facebook':
        result = None
        
        # Try Facebook Graph API if configured
        if FACEBOOK_GRAPH_API.get('access_token'):
            result = fetch_facebook_graph_api(username)
            if result:
                return result
    
    # Try loading from Instagram influencers dataset
    df = load_dataset()
    if df is not None and platform.lower() == 'instagram':
        # Search in 'name' column (username) and 'instagram name' column
        username_lower = username.lower()
        filtered = df[
            (df['name'].str.lower() == username_lower) | 
            (df['instagram name'].str.lower().str.contains(username_lower, na=False))
        ]
        
        if not filtered.empty:
            row = filtered.iloc[0]
            
            # Get followers (already parsed)
            followers = int(row.get('followers_parsed', 0))
            
            # Estimate following based on followers (influencers typically follow 0.5-5% of their followers)
            # For large accounts, following is usually much lower
            if followers > 10_000_000:  # 10M+
                following = int(followers * 0.001)  # 0.1% of followers
            elif followers > 1_000_000:  # 1M+
                following = int(followers * 0.01)  # 1% of followers
            else:
                following = int(followers * 0.05)  # 5% of followers
            
            # Estimate posts based on followers (more followers = more posts typically)
            # Influencers usually have 100-5000 posts
            if followers > 50_000_000:  # 50M+
                posts = int(2000 + (followers / 1_000_000) * 100)
            elif followers > 10_000_000:  # 10M+
                posts = int(1000 + (followers / 1_000_000) * 50)
            else:
                posts = int(500 + (followers / 1_000_000) * 20)
            
            # Create bio from available data
            category = str(row.get('Category_1', ''))
            category2 = str(row.get('Category_2', ''))
            country = str(row.get('country', ''))
            bio_parts = []
            if category and category != 'nan':
                bio_parts.append(category)
            if category2 and category2 != 'nan':
                bio_parts.append(category2)
            if country and country != 'nan':
                bio_parts.append(country)
            bio = ' | '.join(bio_parts) if bio_parts else 'Instagram Influencer'
            
            # These are real influencers, so they likely have profile pics and are verified
            has_profile_pic = True
            is_verified = followers > 1_000_000  # Accounts with 1M+ are often verified
            
            # Account age - estimate based on follower count (older accounts have more followers)
            # This is a rough estimate
            if followers > 100_000_000:
                account_age_days = 3000  # ~8 years
            elif followers > 50_000_000:
                account_age_days = 2500  # ~7 years
            elif followers > 10_000_000:
                account_age_days = 2000  # ~5.5 years
            else:
                account_age_days = 1500  # ~4 years
            
            return {
                'username': str(row.get('name', username)),
                'platform': 'instagram',
                'followers': followers,
                'following': following,
                'posts': posts,
                'bio': bio,
                'has_profile_pic': has_profile_pic,
                'account_age_days': account_age_days,
                'is_verified': is_verified
            }
    
    # Generate mock data as last resort (only if real API fails and dataset doesn't have the account)
    import random
    return {
        'username': username,
        'platform': platform.lower(),
        'followers': random.randint(10, 10000),
        'following': random.randint(50, 5000),
        'posts': random.randint(0, 500),
        'bio': 'Sample bio text',
        'has_profile_pic': random.choice([True, False]),
        'account_age_days': random.randint(1, 2000),
        'is_verified': random.choice([True, False])
    }

@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_account():
    """Analyze account and return prediction"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        platform = data.get('platform', 'instagram').lower()
        
        if not username:
            return jsonify({
                'error': 'Username is required',
                'fake_probability': 0,
                'issues': [],
                'action': 'Error'
            }), 400
        
        # Remove @ symbol if present
        username = username.lstrip('@')
        
        # Fetch account data
        account_data = fetch_account_data(username, platform)
        
        # Calculate fake probability
        fake_probability, issues, action = calculate_fake_probability(account_data)
        
        # Prepare response
        result = {
            'success': True,
            'username': username,
            'platform': platform,
            'fake_probability': round(fake_probability, 2),
            'issues': issues,
            'action': action,
            'account_data': {
                'followers': account_data.get('followers', 0),
                'following': account_data.get('following', 0),
                'posts': account_data.get('posts', 0),
                'bio': account_data.get('bio', ''),
                'has_profile_pic': account_data.get('has_profile_pic', False),
                'account_age_days': account_data.get('account_age_days', 0),
                'is_verified': account_data.get('is_verified', False)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'fake_probability': 0,
            'issues': [],
            'action': 'Error'
        }), 500

@app.route('/report', methods=['POST'])
def report_account():
    """Report account to central system (simulated)"""
    try:
        data = request.get_json()
        username = data.get('username', '')
        platform = data.get('platform', '')
        reason = data.get('reason', 'Suspected fake account')
        
        # Simulate saving to a log file or database
        report_entry = {
            'timestamp': datetime.now().isoformat(),
            'username': username,
            'platform': platform,
            'reason': reason
        }
        
        # Append to reports log file
        with open('reports.log', 'a') as f:
            f.write(f"{report_entry}\n")
        
        return jsonify({
            'success': True,
            'message': f'Report for @{username} has been submitted successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)


