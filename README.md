# 🔍 Fake Social Media Account Detection System

A web-based application that analyzes social media profiles to detect fake or suspicious accounts using HTML, CSS, JavaScript, and Python Flask.

## 🎯 Features

- **Multi-Platform Support**: Analyze accounts from Instagram, Twitter/X, and Facebook
- **Real-time Analysis**: Instant detection of fake account indicators
- **Comprehensive Detection**: Checks multiple factors including:
  - Follower/Following ratio
  - Post frequency and content
  - Bio spam keywords
  - Profile picture presence
  - Account age
  - Verification status
- **Visual Analytics**: Interactive dashboard with Chart.js visualizations
- **Report System**: Submit suspicious accounts to a central logging system
- **Responsive Design**: Modern, mobile-friendly UI

## 📁 Project Structure

```
fake-account-detector/
│
├── static/
│   ├── style.css          # Styling and responsive design
│   └── script.js          # Frontend JavaScript logic
│
├── templates/
│   └── index.html         # Main HTML template
│
├── app.py                 # Flask backend application
├── dataset.csv            # Optional: CSV dataset for training/testing
├── reports.log            # Generated: Central reporting log file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
cd fake-account-detector
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## 🧠 How It Works

### Detection Logic

The system uses a rule-based approach to calculate fake probability (0-100%) based on:

1. **Follower/Following Ratio** (30 points max)

   - Ratio < 0.1: +30 risk points
   - Ratio < 0.5: +15 risk points

2. **Post Activity** (25 points max)

   - No posts but has followers: +25 points
   - Extremely high post-to-follower ratio: +20 points

3. **Spam Keywords** (25 points max)

   - Bio contains spam keywords: +25 points

4. **Profile Picture** (15 points max)

   - No profile picture: +15 points

5. **Account Age** (20 points max)

   - Account < 7 days old: +20 points
   - Account < 30 days old: +10 points

6. **Follower Count** (10 points max)

   - Very low follower count (< 10): +10 points

7. **Following Count** (15 points max)
   - Following > 3000 accounts: +15 points

### Risk Categories

- **0-39%**: Likely Genuine ✅
- **40-69%**: Review Carefully ⚠️
- **70-100%**: Report or Verify 🚨

## 📊 API Endpoints

### GET `/`

Returns the homepage with the analysis interface.

### POST `/analyze`

Analyzes a social media account.

**Request Body:**

```json
{
  "username": "example_user",
  "platform": "instagram"
}
```

**Response:**

```json
{
  "success": true,
  "username": "example_user",
  "platform": "instagram",
  "fake_probability": 84.5,
  "issues": [
    "Very low follower-to-following ratio (0.01)",
    "Bio contains spam keywords",
    "No profile picture"
  ],
  "action": "Report or Verify",
  "account_data": {
    "followers": 25,
    "following": 4900,
    "posts": 2,
    "bio": "Follow me for giveaway win free money",
    "has_profile_pic": false,
    "account_age_days": 5,
    "is_verified": false
  }
}
```

### POST `/report`

Submits a report for a suspicious account.

**Request Body:**

```json
{
  "username": "example_user",
  "platform": "instagram",
  "reason": "Suspected fake account"
}
```

## 🔧 Integration with Real APIs

### Twitter/X API (tweepy)

To use real Twitter data, add your API credentials to `app.py`:

```python
import tweepy

# Configure API
auth = tweepy.OAuthHandler("API_KEY", "API_SECRET")
auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")
api = tweepy.API(auth)

# Fetch user data
user = api.get_user(screen_name=username)
```

### Instagram (instaloader)

```python
import instaloader

L = instaloader.Instaloader()
profile = instaloader.Profile.from_username(L.context, username)

account_data = {
    'followers': profile.followers,
    'following': profile.followees,
    'posts': profile.mediacount,
    'bio': profile.biography,
    'has_profile_pic': profile.profile_pic_url is not None,
    # ...
}
```

### Using CSV Dataset

Place your dataset as `dataset.csv` with columns:

- `username`
- `platform`
- `followers`
- `following`
- `posts`
- `bio`
- `has_profile_pic`
- `account_age_days`
- `is_verified`

## 🎨 Frontend Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Dynamic result display with animations
- **Interactive Charts**: Visual analytics using Chart.js
- **Browser Alerts**: Automatic warnings for high-risk accounts
- **Report System**: One-click reporting to central system

## 📈 Analytics Dashboard

The dashboard shows:

- Total accounts analyzed
- Fake accounts detected
- Genuine accounts
- Visual pie chart of fake vs genuine ratio

## 🛡️ Security Notes

- This is a demonstration system. For production use:
  - Implement proper authentication
  - Add rate limiting
  - Secure API keys and credentials
  - Use HTTPS
  - Validate and sanitize all inputs

## 🧪 Testing

The system includes sample accounts for testing:

- `fake_account_1` (Instagram) - High fake probability
- `genuine_user` (Instagram) - Low fake probability
- `spam_bot` (Twitter) - High fake probability

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Add support for more platforms
- Improve detection algorithms
- Enhance UI/UX
- Add machine learning models

## 📧 Support

For issues or questions, please open an issue in the repository.

---

**Built with ❤️ using Flask, JavaScript, and Chart.js**




