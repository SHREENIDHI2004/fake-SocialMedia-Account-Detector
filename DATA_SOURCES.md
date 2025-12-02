# 📊 Current Data Sources - Real vs Mock Data

## ⚠️ **Current Status: NOT Using Real APIs**

Right now, the system is **NOT** fetching real data from social media platforms. It uses three fallback methods:

---

## 🔍 How Data is Currently Fetched

### 1. **Predefined Sample Accounts** (Hardcoded)

**Location:** `app.py` lines 20-46

**Accounts Available:**

- `fake_account_1` (Instagram) - Pre-configured fake account
- `genuine_user` (Instagram) - Pre-configured real account
- `spam_bot` (Twitter) - Pre-configured spam account

**How it works:**

- If you enter one of these exact usernames, it returns the pre-configured data
- These are **NOT real accounts** - just examples for testing

---

### 2. **Instagram Influencers Dataset** (REAL DATA)

**Location:** `social media influencers-INSTAGRAM - -DEC 2022.csv`

**Contains:** 1,000 real Instagram influencer accounts from December 2022

**Data includes:**
- Real usernames (e.g., leomessi, cristiano, neymarjr, kyliejenner)
- Real follower counts (in millions)
- Categories (Sports, Music, Fashion, etc.)
- Countries
- Engagement metrics

**How it works:**

- If you search for an Instagram account and the username matches the dataset, it uses **REAL follower data**
- Missing fields (following, posts, bio) are intelligently estimated based on follower count
- These are **REAL influencer accounts** with actual follower numbers
- The system searches in both the `name` column (username) and `instagram name` column

**Example accounts you can try:**
- `leomessi` - Leo Messi (409.8M followers)
- `cristiano` - Cristiano Ronaldo (523M followers)
- `neymarjr` - Neymar (198.9M followers)
- `kyliejenner` - Kylie Jenner (376.3M followers)
- `taylorswift` - Taylor Swift (236.5M followers)
- And 995 more real influencer accounts!

---

### 3. **Random Mock Data** (Fallback)

**Location:** `app.py` lines 162-176

**What happens:**

- If username doesn't match sample accounts or CSV
- System generates **RANDOM fake data**:
  - Random followers (10-10,000)
  - Random following (50-5,000)
  - Random posts (0-500)
  - Random profile picture (True/False)
  - Random account age (1-2,000 days)

**Example:**

- Enter: `@random_user123` → Gets random data
- Enter: `@test_account` → Gets random data
- **Every time you analyze a new account, it generates different random numbers!**

---

## ❌ What's NOT Working

The system is **NOT**:

- ❌ Fetching real data from Instagram API
- ❌ Fetching real data from Twitter/X API
- ❌ Fetching real data from Facebook API
- ❌ Accessing actual social media profiles
- ❌ Using real usernames to get real account information

---

## ✅ What IS Working

The system **IS**:

- ✅ Analyzing data correctly (the detection algorithm is real)
- ✅ Calculating fake probability based on real patterns
- ✅ Using real detection logic (7 factors, weighted scoring)
- ✅ Displaying results properly
- ✅ Working with sample/mock data for demonstration

---

## 🔧 How to Add REAL Data Integration

To analyze **REAL accounts**, you need to integrate actual APIs. Here's how:

### Option 1: Twitter/X API (tweepy)

**Step 1:** Get Twitter API credentials from https://developer.twitter.com

**Step 2:** Modify `fetch_account_data()` in `app.py`:

```python
def fetch_account_data(username, platform):
    if platform.lower() == 'twitter':
        try:
            import tweepy

            # Your API credentials
            auth = tweepy.OAuthHandler("YOUR_API_KEY", "YOUR_API_SECRET")
            auth.set_access_token("YOUR_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_SECRET")
            api = tweepy.API(auth, wait_on_rate_limit=True)

            # Fetch real user data
            user = api.get_user(screen_name=username)

            return {
                'username': user.screen_name,
                'platform': 'twitter',
                'followers': user.followers_count,
                'following': user.friends_count,
                'posts': user.statuses_count,
                'bio': user.description or '',
                'has_profile_pic': user.profile_image_url is not None,
                'account_age_days': (datetime.now() - user.created_at).days,
                'is_verified': user.verified
            }
        except Exception as e:
            print(f"Twitter API Error: {e}")
            return None

    # ... rest of the code
```

### Option 2: Instagram (instaloader)

**Step 1:** Install: `pip install instaloader`

**Step 2:** Modify `fetch_account_data()`:

```python
def fetch_account_data(username, platform):
    if platform.lower() == 'instagram':
        try:
            import instaloader

            L = instaloader.Instaloader()
            profile = instaloader.Profile.from_username(L.context, username)

            return {
                'username': profile.username,
                'platform': 'instagram',
                'followers': profile.followers,
                'following': profile.followees,
                'posts': profile.mediacount,
                'bio': profile.biography or '',
                'has_profile_pic': profile.profile_pic_url is not None,
                'account_age_days': (datetime.now() - profile.joined_date).days if profile.joined_date else 0,
                'is_verified': profile.is_verified
            }
        except Exception as e:
            print(f"Instagram Error: {e}")
            return None

    # ... rest of the code
```

**Note:** Instagram API access is limited. You may need to log in or use alternative methods.

### Option 3: Use Public Datasets

You can download real datasets from:

- Kaggle: "Fake vs Real Social Media Accounts Dataset"
- Kaggle: "Twitter Bot Detection Dataset"
- UCI Machine Learning Repository

Then load them into your `dataset.csv` file.

---

## 🎯 Summary

| Data Source                 | Real?  | Random?    | When Used                    |
| --------------------------- | ------ | ---------- | ---------------------------- |
| Sample accounts (hardcoded) | ❌ No  | ❌ No      | Exact username match         |
| **Instagram Influencers**   | ✅ **YES** | ❌ No      | **Instagram accounts in dataset** |
| Random mock data            | ❌ No  | ✅ **YES** | Any other username           |
| Twitter API                 | ✅ Yes | ❌ No      | **Not implemented yet**       |
| Instagram API               | ✅ Yes | ❌ No      | **Not implemented yet**       |

---

## 💡 Current Behavior Example

**Scenario 1:** Enter `fake_account_1` (Instagram)

- ✅ Uses predefined sample data (not random)

**Scenario 2:** Enter `genuine_user` (Instagram)

- ✅ Uses predefined sample data (not random)

**Scenario 3:** Enter `@my_test_account` (any platform)

- ⚠️ Generates **RANDOM data** (different each time)
- This is why results vary for unknown accounts!

**Scenario 4:** Enter username from Instagram influencers dataset (e.g., `leomessi`, `cristiano`)

- ✅ Uses **REAL follower data** from the dataset
- ✅ Estimates other fields intelligently based on follower count
- ✅ These are real influencer accounts with actual data!

---

## 🚀 To Get Real Data

You need to:

1. Set up API credentials (Twitter/Instagram)
2. Modify `fetch_account_data()` function
3. Handle API rate limits and errors
4. Test with real usernames

The detection algorithm itself is real and works correctly - it just needs real data to analyze!




