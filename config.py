"""
Configuration file for Instagram API credentials
Replace with your actual API keys/credentials
"""

# Instagram Graph API Configuration
# Get these from: https://developers.facebook.com/apps/
INSTAGRAM_GRAPH_API = {
    'access_token': '',  # Your Instagram Graph API access token
    'user_id': '',  # Your Instagram Business Account User ID (optional)
}

# Instagram Basic Display API (Alternative method)
# Get these from: https://developers.facebook.com/apps/
INSTAGRAM_BASIC_DISPLAY = {
    'client_id': '',  # Your App ID
    'client_secret': '',  # Your App Secret
    'access_token': '',  # User access token
}

# Instaloader Configuration (Fallback method - uses login, not API keys)
# This method doesn't use API keys, it uses your Instagram login
INSTALOADER_CONFIG = {
    'username': 'shrinidhi8765',  # Your Instagram username (optional, only if needed)
    'password': 'sssscgK9@',  # Your Instagram password (optional, only if needed)
    'use_session_file': True,  # Save session to avoid repeated logins
}

# Facebook Graph API Configuration
# Get these from: https://developers.facebook.com/apps/
FACEBOOK_GRAPH_API = {
    'access_token': '',  # Your Facebook Graph API access token
    'app_id': '',  # Your Facebook App ID (optional)
    'app_secret': '',  # Your Facebook App Secret (optional)
}

# API Method Priority (try in this order)
# Options: 'graph_api', 'basic_display', 'instaloader', 'mock'
API_METHOD = 'instaloader'  # Change to your preferred method

# Enable real API fetching
USE_REAL_API = True  # Set to True to enable real API calls, False for mock data

