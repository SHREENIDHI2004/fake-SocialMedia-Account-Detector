# 🔍 How the Analysis Works

## Overview

The system uses a **rule-based scoring algorithm** that analyzes 7 different factors to determine if an account is fake. Each factor adds "risk points" (0-100 total), which becomes the fake probability percentage.

---

## 📊 The 7 Detection Factors

### 1. **Follower/Following Ratio** (30 points max)
**What it checks:** How many followers vs how many accounts they follow

**Logic:**
- If ratio < 0.1 (following 10x more than followers): +30 points
- If ratio < 0.5 (following 2x more than followers): +15 points

**Example:**
- 25 followers, 4,900 following → Ratio = 0.005 → **+30 points** ⚠️
- 8,500 followers, 450 following → Ratio = 18.9 → **0 points** ✅

**Why:** Real accounts usually have balanced ratios. Fake accounts often follow thousands but have few followers.

---

### 2. **Post Activity Patterns** (25 points max)
**What it checks:** Post count relative to followers

**Logic:**
- No posts but has followers: +25 points
- Extremely high post-to-follower ratio (>10): +20 points

**Example:**
- 0 posts, 150 followers → **+25 points** ⚠️
- 15,000 posts, 10 followers → **+20 points** ⚠️

**Why:** Real accounts have consistent posting. Suspicious patterns include no content or spam posting.

---

### 3. **Spam Keywords in Bio** (25 points max)
**What it checks:** Bio text for common spam phrases

**Spam Keywords Detected:**
```
'giveaway', 'win', 'free', 'click here', 'follow for follow',
'make money', 'easy money', 'get rich', 'spam', 'bot',
'f4f', 'l4l', 'follow me', 'dm me', 'link in bio spam'
```

**Logic:**
- If bio contains any spam keyword: +25 points

**Example:**
- Bio: "Follow me for giveaway win free money" → **+25 points** ⚠️
- Bio: "Photographer | Travel enthusiast" → **0 points** ✅

**Why:** Fake accounts often use spammy language to attract followers.

---

### 4. **Profile Picture** (15 points max)
**What it checks:** Whether account has a profile picture

**Logic:**
- No profile picture: +15 points

**Example:**
- `has_profile_pic: False` → **+15 points** ⚠️
- `has_profile_pic: True` → **0 points** ✅

**Why:** Real accounts almost always have profile pictures. Fake/bot accounts often skip this.

---

### 5. **Account Age** (20 points max)
**What it checks:** How old the account is (in days)

**Logic:**
- Account < 7 days old: +20 points
- Account < 30 days old: +10 points

**Example:**
- Account created 2 days ago → **+20 points** ⚠️
- Account created 1,200 days ago → **0 points** ✅

**Why:** New accounts are more likely to be fake, especially if combined with other suspicious factors.

---

### 6. **Follower Count** (10 points max)
**What it checks:** Very low follower count

**Logic:**
- Less than 10 followers: +10 points

**Example:**
- 5 followers → **+10 points** ⚠️
- 3,200 followers → **0 points** ✅

**Why:** Extremely low follower counts can indicate fake or abandoned accounts.

---

### 7. **Following Count** (15 points max)
**What it checks:** Following too many accounts

**Logic:**
- Following > 3,000 accounts: +15 points

**Example:**
- Following 5,000 accounts → **+15 points** ⚠️
- Following 450 accounts → **0 points** ✅

**Why:** Following thousands of accounts is a common fake account tactic (follow-for-follow spam).

---

## 🧮 How Fake Probability is Calculated

```
Risk Score = Sum of all detected factor points
Fake Probability = min(Risk Score, 100)
```

### Example Calculation:

**Account: fake_account_1**
- Followers: 25, Following: 4,900
  - Ratio = 0.005 (< 0.1) → **+30 points**
- Posts: 2, Followers: 25
  - No suspicious pattern → **0 points**
- Bio: "Follow me for giveaway win free money"
  - Contains spam keywords → **+25 points**
- No profile picture → **+15 points**
- Account age: 5 days → **+20 points**
- Followers: 25 (> 10) → **0 points**
- Following: 4,900 (> 3,000) → **+15 points**

**Total Risk Score: 30 + 25 + 15 + 20 + 15 = 105**
**Fake Probability: min(105, 100) = 100%** 🚨

---

## 📈 Risk Categories

| Probability | Category | Action |
|------------|----------|--------|
| **0-39%** | ✅ Likely Genuine | Safe to interact |
| **40-69%** | ⚠️ Review Carefully | Exercise caution |
| **70-100%** | 🚨 Report or Verify | High risk of being fake |

---

## 🔄 Data Flow

```
1. User enters username → Frontend (JavaScript)
2. JavaScript sends POST request → Flask backend (/analyze)
3. Flask fetches account data:
   - Checks sample dataset first
   - Falls back to CSV dataset if available
   - Generates mock data if not found (for demo)
4. calculate_fake_probability() runs:
   - Checks all 7 factors
   - Calculates risk score
   - Identifies issues
5. Flask returns JSON response:
   {
     "fake_probability": 84.5,
     "issues": [...],
     "action": "Report or Verify",
     "account_data": {...}
   }
6. Frontend displays results:
   - Shows probability with color coding
   - Lists all detected issues
   - Updates analytics dashboard
   - Shows browser alert if high risk
```

---

## 🎯 Current Data Sources

### 1. **Sample Dataset** (Hardcoded in app.py)
- Pre-loaded test accounts for demonstration
- Includes: fake_account_1, genuine_user, spam_bot

### 2. **CSV Dataset** (dataset.csv)
- Can be loaded if file exists
- Contains 10 sample accounts with various characteristics

### 3. **Mock Data Generation** (Fallback)
- Generates random data if account not found
- Used for demonstration purposes

---

## 🔮 Future Enhancements (Real API Integration)

To use **real social media APIs**, you would:

### Twitter/X (tweepy):
```python
import tweepy
api = tweepy.API(auth)
user = api.get_user(screen_name=username)
# Extract: followers, following, posts, bio, etc.
```

### Instagram (instaloader):
```python
import instaloader
L = instaloader.Instaloader()
profile = instaloader.Profile.from_username(L.context, username)
# Extract: followers, following, posts, bio, etc.
```

Then the same `calculate_fake_probability()` function would analyze the real data!

---

## 💡 Why This Approach Works

1. **Multiple Indicators**: Checks 7 different factors, not just one
2. **Weighted Scoring**: More suspicious factors get higher points
3. **Real-world Patterns**: Based on common fake account behaviors
4. **Transparent**: Users can see exactly why an account is flagged
5. **Extensible**: Easy to add more factors or ML models

---

## 🧪 Testing Examples

Try these accounts to see different risk levels:

| Username | Platform | Expected Probability |
|----------|----------|---------------------|
| `fake_account_1` | Instagram | ~100% (High risk) |
| `genuine_user` | Instagram | ~0-10% (Low risk) |
| `spam_bot` | Twitter | ~90%+ (High risk) |
| `suspicious_account` | Instagram | ~60-70% (Medium risk) |

---

**Note:** This is a rule-based system. For production, you could enhance it with machine learning models trained on larger datasets for more accurate predictions!





