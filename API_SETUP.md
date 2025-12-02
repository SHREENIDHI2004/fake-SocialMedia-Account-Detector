# 📡 Instagram API Setup Guide

This guide will help you configure the Instagram API integration to fetch real account data for fake account detection.

## 🔑 Available Methods

The system supports **three methods** to fetch Instagram data:

### 1. **Instagram Graph API** (Recommended for Business Accounts)
- ✅ Official API from Meta/Facebook
- ✅ Most reliable and stable
- ❌ Requires Instagram Business Account linked to Facebook Page
- ❌ Needs access token with proper permissions

### 2. **Instaloader** (Recommended for Public Accounts)
- ✅ Works without API keys for public accounts
- ✅ Can access account age (join date)
- ✅ No account linking required
- ⚠️ May require login for private accounts
- ⚠️ Uses scraping (could be rate-limited)

### 3. **Fallback to Dataset/Mock Data**
- ✅ Always available
- ❌ Not real-time data

---

## 🚀 Quick Setup

### Step 1: Edit `config.py`

Open `config.py` and configure your preferred method:

```python
# Enable real API fetching
USE_REAL_API = True

# Choose your method: 'graph_api' or 'instaloader'
API_METHOD = 'instaloader'  # Change to 'graph_api' if you have Graph API credentials
```

---

## 📋 Method 1: Instagram Graph API Setup

### Prerequisites:
- Instagram Business Account
- Facebook Page linked to your Instagram account
- Facebook Developer Account

### Steps:

1. **Create a Facebook App**
   - Go to https://developers.facebook.com/apps/
   - Click "Create App" → Choose "Business" type
   - Fill in app details

2. **Add Instagram Basic Display or Instagram Graph API**
   - In your app dashboard, go to "Add Product"
   - Add "Instagram Basic Display" or "Instagram Graph API"

3. **Get Access Token**
   
   **Option A: User Access Token (Instagram Basic Display)**
   - Go to "Basic Display" → Settings
   - Add your Instagram account as a test user
   - Get OAuth URL and authorize
   - Copy the access token

   **Option B: Page Access Token (Instagram Graph API)**
   - Go to "Instagram Graph API" → Settings
   - Link your Instagram Business Account to a Facebook Page
   - Get Page Access Token from Facebook Graph API Explorer
   - Token must have: `instagram_basic`, `instagram_graph_user_profile`, `pages_show_list`

4. **Update `config.py`**

```python
INSTAGRAM_GRAPH_API = {
    'access_token': 'YOUR_ACCESS_TOKEN_HERE',
    'user_id': '',  # Optional: Your Instagram Business Account User ID
}

API_METHOD = 'graph_api'
USE_REAL_API = True
```

### Limitations:
- Only works with Instagram Business/Creator accounts
- Account must be linked to a Facebook Page
- Access tokens expire (usually 60 days for user tokens)

---

## 📋 Method 2: Instaloader Setup (Easiest)

### No API Keys Required for Public Accounts!

1. **Install Instaloader** (if not already installed)
   ```bash
   pip install instaloader
   ```

2. **For Public Accounts** (No configuration needed!)
   
   Just set in `config.py`:
   ```python
   API_METHOD = 'instaloader'
   USE_REAL_API = True
   INSTALOADER_CONFIG = {
       'username': '',  # Leave empty for public accounts
       'password': '',  # Leave empty for public accounts
       'use_session_file': True,
   }
   ```
   
   That's it! It will work for public Instagram accounts.

3. **For Private Accounts** (Optional)

   If you need to access private accounts, add your login credentials:
   ```python
   INSTALOADER_CONFIG = {
       'username': 'your_instagram_username',
       'password': 'your_instagram_password',
       'use_session_file': True,  # Saves session to avoid repeated logins
   }
   ```

### Advantages:
- ✅ Works immediately for public accounts
- ✅ No API setup required
- ✅ Can fetch account age (join date)
- ✅ More accessible than Graph API

### Disadvantages:
- ⚠️ Uses web scraping (may be rate-limited)
- ⚠️ Instagram might block frequent requests
- ⚠️ Less reliable than official API

---

## 🧪 Testing Your Setup

1. **Start the Flask app:**
   ```bash
   python app.py
   ```

2. **Open the web interface:**
   - Go to http://localhost:5000

3. **Test with a real Instagram username:**
   - Enter any public Instagram username (e.g., `natgeo`, `instagram`)
   - Click "Analyze Account"
   - Check if real data is fetched

4. **Check console output:**
   - If using Graph API: Look for "Instagram Graph API Error" messages
   - If using Instaloader: Look for "Instaloader Error" messages
   - Success means no errors and real data is shown

---

## 🔧 Troubleshooting

### Graph API Issues:

**Error: "Invalid access token"**
- Access token expired (regenerate it)
- Token doesn't have required permissions
- Solution: Get a new token with proper permissions

**Error: "User not found"**
- Account must be Instagram Business/Creator account
- Account must be linked to a Facebook Page
- Solution: Convert account to Business and link it

**Error: "Rate limit exceeded"**
- Too many API calls
- Solution: Wait before making more requests

### Instaloader Issues:

**Error: "Profile does not exist"**
- Username is incorrect
- Account was deleted
- Solution: Verify the username

**Error: "Private profile not followed"**
- Account is private and you're not logged in
- Solution: Add login credentials to `INSTALOADER_CONFIG` or use Graph API

**Error: "Login required"**
- Instagram detected automated access
- Solution: Add your credentials, enable 2FA, or wait before retrying

**Rate Limiting:**
- Instagram may temporarily block requests
- Solution: Wait a few minutes and reduce request frequency

---

## 🎯 Recommended Setup

**For Most Users:**
1. Use **Instaloader** with no credentials (public accounts only)
2. Set `USE_REAL_API = True` and `API_METHOD = 'instaloader'`
3. Test with public accounts first

**For Business/Production:**
1. Set up **Instagram Graph API** with proper credentials
2. Use `API_METHOD = 'graph_api'`
3. Implement proper token refresh mechanism
4. Add rate limiting to respect API limits

---

## 📝 Security Notes

⚠️ **Important:**
- **NEVER** commit `config.py` with real credentials to version control
- Add `config.py` to `.gitignore`
- Use environment variables for production
- Keep access tokens secure and rotate them regularly
- Don't share your API credentials

---

## 📚 Additional Resources

- Instagram Graph API Docs: https://developers.facebook.com/docs/instagram-api
- Instaloader Docs: https://instaloader.github.io/
- Facebook Developers: https://developers.facebook.com/

---

## ✅ Checklist

- [ ] Edited `config.py`
- [ ] Set `USE_REAL_API = True`
- [ ] Chosen API method (`graph_api` or `instaloader`)
- [ ] Added API credentials (if using Graph API)
- [ ] Installed instaloader: `pip install instaloader`
- [ ] Tested with a real Instagram username
- [ ] Verified real data is being fetched
- [ ] Added `config.py` to `.gitignore`

---

## 🆘 Still Having Issues?

1. Check console output for specific error messages
2. Verify your API credentials are correct
3. Test with a well-known public account (e.g., `instagram`, `natgeo`)
4. Check your internet connection
5. Verify instaloader is installed: `pip show instaloader`

---

**Happy Fake Account Detecting! 🚀**

