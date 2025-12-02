"""
Quick test script to verify Instagram API integration
Run this to test if your configuration works
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from config import USE_REAL_API, API_METHOD, INSTALOADER_CONFIG
    from app import fetch_account_data
    
    print("=" * 60)
    print("Testing Instagram API Integration")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  USE_REAL_API: {USE_REAL_API}")
    print(f"  API_METHOD: {API_METHOD}")
    if API_METHOD == 'instaloader':
        print(f"  Username: {INSTALOADER_CONFIG.get('username', 'Not set')}")
        print(f"  Password: {'*' * len(INSTALOADER_CONFIG.get('password', '')) if INSTALOADER_CONFIG.get('password') else 'Not set'}")
    
    print("\n" + "=" * 60)
    print("Testing with a real Instagram account...")
    print("=" * 60)
    
    # Test with a well-known public account
    test_username = "instagram"  # Official Instagram account (public)
    print(f"\nFetching data for: @{test_username}")
    print("(This may take a few seconds...)\n")
    
    try:
        account_data = fetch_account_data(test_username, 'instagram')
        
        if account_data:
            print("[SUCCESS] Real data fetched:")
            print(f"  Username: {account_data.get('username')}")
            print(f"  Followers: {account_data.get('followers'):,}")
            print(f"  Following: {account_data.get('following'):,}")
            print(f"  Posts: {account_data.get('posts'):,}")
            bio = account_data.get('bio', '')
            try:
                print(f"  Bio: {bio[:100]}...")
            except UnicodeEncodeError:
                print(f"  Bio: (contains special characters, length: {len(bio)} chars)")
            print(f"  Has Profile Pic: {account_data.get('has_profile_pic')}")
            print(f"  Account Age: {account_data.get('account_age_days')} days")
            print(f"  Verified: {account_data.get('is_verified')}")
            
            print("\n" + "=" * 60)
            print("[SUCCESS] Setup is working correctly!")
            print("=" * 60)
            print("\nYou can now run the Flask app:")
            print("  python app.py")
            print("\nThen open http://localhost:5000 in your browser")
        else:
            print("[ERROR] Failed to fetch data. Check your configuration.")
            
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nPossible issues:")
        print("  1. Check your internet connection")
        print("  2. Verify your Instagram credentials in config.py")
        print("  3. Instagram may be rate-limiting (wait a few minutes)")
        print("  4. Check if instaloader is installed: pip install instaloader")

except ImportError as e:
    print(f"[ERROR] Import Error: {e}")
    print("\nMake sure you're in the project directory and all dependencies are installed:")
    print("  pip install -r requirements.txt")
except Exception as e:
    print(f"[ERROR] Unexpected Error: {e}")
    import traceback
    traceback.print_exc()

