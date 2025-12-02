# 🚀 Quick Start Guide

## ✅ Your Setup is Complete!

Your Instagram API integration is working and ready to detect fake accounts!

---

## 🎯 Next Steps:

### 1. Start the Flask Application

Open a terminal in this folder and run:

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### 2. Open in Browser

Go to: **http://localhost:5000**

### 3. Test It Out!

1. Enter any Instagram username (e.g., `instagram`, `natgeo`, `cristiano`)
2. Select "Instagram" as platform
3. Click "Analyze Account"
4. See real-time fake account detection results!

---

## 📊 What You'll See:

The system will analyze:
- ✅ **Real follower/following counts** from Instagram
- ✅ **Actual post counts**
- ✅ **Bio content** (checks for spam keywords)
- ✅ **Account verification status**
- ✅ **Profile picture presence**
- ✅ **Fake probability score** (0-100%)

---

## 🔒 Security Reminder:

⚠️ **Important**: Your Instagram credentials are stored in `config.py`

- ✅ `config.py` is already in `.gitignore` (won't be committed)
- ⚠️ Never share or commit your `config.py` file
- 🔄 Consider changing your Instagram password periodically
- 📝 For production, use environment variables instead

---

## 🧪 Test Accounts to Try:

Public accounts (work immediately):
- `instagram` - Official Instagram account
- `natgeo` - National Geographic
- `cristiano` - Cristiano Ronaldo
- Any public Instagram username

---

## ⚠️ Troubleshooting:

**If you get rate-limited:**
- Wait a few minutes between requests
- Instagram may temporarily block frequent requests

**If login fails:**
- Check your credentials in `config.py`
- Make sure 2FA is disabled (or use an app password)

**If account is private:**
- The system will try to access it with your credentials
- Some accounts may still be inaccessible

---

## 📝 What's Happening Behind the Scenes:

1. Your app uses **Instaloader** to fetch real Instagram data
2. It analyzes 7 different factors to detect fake accounts:
   - Follower/Following ratio
   - Post count patterns
   - Spam keywords in bio
   - Profile picture presence
   - Account age
   - Verification status
   - Suspicious activity patterns

3. Returns a **fake probability score** (0-100%)

---

## 🎉 You're Ready!

Run `python app.py` and start detecting fake accounts with real Instagram data!

