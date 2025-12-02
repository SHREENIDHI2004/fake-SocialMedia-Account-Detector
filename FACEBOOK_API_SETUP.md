# 📡 Facebook API Setup Guide

This guide will help you configure the Facebook Graph API integration to fetch real account data for fake account detection.

## 🔑 Facebook Graph API

The system uses **Facebook Graph API** (official API from Meta/Facebook) to fetch real Facebook profile data.

### ✅ Features:
- Official API from Meta/Facebook
- Most reliable and stable
- Access to public profile data
- Friend counts, posts, bio information

### ⚠️ Limitations:
- Requires Facebook Developer account
- Access token needed with proper permissions
- Privacy policies limit access to public data only
- User access tokens expire (usually 60 days)

---

## 🚀 Quick Setup

### Step 1: Create Facebook App

1. **Go to Facebook Developers**
   - Visit: https://developers.facebook.com/apps/
   - Click "Create App"
   - Choose app type: "Business" or "Other"

2. **Get App Credentials**
   - Note down your **App ID** and **App Secret**
   - These are in your app dashboard under "Settings" → "Basic"

### Step 2: Get Access Token

You have **two options**:

#### Option A: User Access Token (Recommended for Personal Use)

1. **Add Facebook Login Product**
   - In your app dashboard, go to "Add Product"
   - Find "Facebook Login" and click "Set Up"

2. **Configure OAuth Settings**
   - Go to Facebook Login → Settings
   - Add "Valid OAuth Redirect URIs": `http://localhost:5000`
   - Add your domain in production

3. **Get User Access Token**
   - Go to: https://developers.facebook.com/tools/explorer/
   - Select your app from the dropdown
   - Click "Get Token" → "Get User Access Token"
   - Select permissions:
     - `public_profile` (required)
     - `user_friends` (optional, for friend counts)
     - `user_posts` (optional, for posts data)
   - Authorize and copy the token

#### Option B: Page Access Token (For Public Pages/Profiles)

1. **Add Page**
   - Link a Facebook Page to your app
   - Go to "Products" → "Facebook Login" → "Settings"
   - Add Page ID

2. **Get Page Access Token**
   - Use Graph API Explorer
   - Select your app and page
   - Get Page Access Token with permissions

### Step 3: Update `config.py`

Open `config.py` and add your credentials:

```python
FACEBOOK_GRAPH_API = {
    'access_token': 'YOUR_ACCESS_TOKEN_HERE',
    'app_id': 'YOUR_APP_ID',  # Optional
    'app_secret': 'YOUR_APP_SECRET',  # Optional
}
```

**Important**: Keep your access token secure! Don't share it.

### Step 4: Enable Real API

Make sure in `config.py`:

```python
USE_REAL_API = True
```

---

## 🧪 Testing Your Setup

### Test with a Facebook Username or ID

1. **Find a Facebook Username or ID**
   - You can use a username (if available) or user ID
   - To find user ID: https://findmyfbid.com/

2. **Run your Flask app:**
   ```bash
   python app.py
   ```

3. **Test in browser:**
   - Go to http://localhost:5000
   - Select "Facebook" as platform
   - Enter a Facebook username or user ID
   - Click "Analyze Account"

### Example Test Accounts:

- **Facebook Username**: If the profile has a custom username (e.g., `zuck`, `mark.zuckerberg`)
- **Facebook User ID**: Numeric ID (e.g., `4` for Mark Zuckerberg's profile)
- **Public Pages**: Page usernames or Page IDs

---

## 🔧 Troubleshooting

### Error: "Invalid access token"
- **Cause**: Access token expired or invalid
- **Solution**: 
  - Generate a new access token from Graph API Explorer
  - Update `config.py` with the new token
  - User access tokens expire after ~60 days

### Error: "Insufficient permissions"
- **Cause**: Access token doesn't have required permissions
- **Solution**:
  - Go to Graph API Explorer
  - Select your app
  - Click "Get Token" → "Get User Access Token"
  - Add required permissions: `public_profile`, `user_friends`, `user_posts`
  - Generate new token and update config

### Error: "User not found" or "Unsupported operation"
- **Cause**: 
  - Username doesn't exist
  - Profile is private
  - Using wrong ID format
- **Solution**:
  - Verify the username/ID exists
  - Make sure you're using a public profile
  - Try using numeric user ID instead of username

### Error: "Rate limit exceeded"
- **Cause**: Too many API calls
- **Solution**: Wait before making more requests

### Limited Data Returned
- **Cause**: Facebook privacy policies limit accessible data
- **Solution**: 
  - This is normal - Facebook only allows public data
  - Friend counts may not be available for all profiles
  - Some profiles have stricter privacy settings

---

## 📊 What Data Can Be Fetched

### Available Data:
- ✅ Profile name/username
- ✅ Profile picture status
- ✅ Bio/about information (if public)
- ✅ Friend count (if privacy allows)
- ✅ Post count (limited by API)
- ✅ Verification status (for verified accounts)
- ⚠️ Followers count (only for Pages, not personal profiles)

### Not Available:
- ❌ Account creation date (Facebook doesn't provide this)
- ❌ Full friend list (privacy restrictions)
- ❌ Full post history (API limitations)
- ❌ Private profile information

---

## 🔒 Security Best Practices

1. **Never commit `config.py`** with real credentials
   - ✅ Already in `.gitignore`
   - ⚠️ Double-check before pushing to git

2. **Rotate Access Tokens Regularly**
   - User tokens expire after ~60 days
   - Generate new tokens before they expire

3. **Use Environment Variables in Production**
   ```python
   import os
   FACEBOOK_GRAPH_API = {
       'access_token': os.environ.get('FB_ACCESS_TOKEN', ''),
   }
   ```

4. **Limit Token Permissions**
   - Only request permissions you actually need
   - Don't request sensitive permissions unnecessarily

5. **Keep App Secret Secure**
   - Never expose App Secret in client-side code
   - Store it securely on the server

---

## 🎯 Quick Reference

### Required Permissions:
- `public_profile` - Basic profile information (required)
- `user_friends` - Friend counts (optional)
- `user_posts` - Post data (optional)

### API Endpoints Used:
- `https://graph.facebook.com/v18.0/{user_id}`
- Fields: `id,name,username,about,link,picture,verified,friends,posts`

### Token Types:
- **User Access Token**: For accessing user's own data or public data
- **Page Access Token**: For accessing page data
- **App Access Token**: Limited functionality, mostly for app operations

---

## 📚 Additional Resources

- Facebook Graph API Docs: https://developers.facebook.com/docs/graph-api
- Graph API Explorer: https://developers.facebook.com/tools/explorer/
- Facebook Developers: https://developers.facebook.com/
- Permission Reference: https://developers.facebook.com/docs/permissions/reference

---

## ✅ Setup Checklist

- [ ] Created Facebook Developer account
- [ ] Created Facebook App
- [ ] Got App ID and App Secret
- [ ] Added Facebook Login product
- [ ] Got User Access Token with required permissions
- [ ] Updated `config.py` with access token
- [ ] Set `USE_REAL_API = True`
- [ ] Tested with a real Facebook username/ID
- [ ] Verified data is being fetched correctly

---

## 🆘 Still Having Issues?

1. **Check Console Output**
   - Look for specific error messages
   - Facebook API errors are usually descriptive

2. **Verify Token**
   - Test your token in Graph API Explorer
   - Make sure it has the right permissions

3. **Check Privacy Settings**
   - Target profile must be public
   - Some data requires profile to be public

4. **Review API Limits**
   - Facebook has rate limits
   - Don't make too many requests too quickly

---

**Happy Fake Account Detecting on Facebook! 🚀**

