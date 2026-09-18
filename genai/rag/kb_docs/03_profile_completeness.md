# Profile Completeness Signals: Picture, Bio, Verification, External URL

**Source**:
1. Dey et al. (2019). IJCSIT — among the top-5 distinguishing features were:
   profile picture presence (weight ≈ +0.27 towards genuine), description length, and external URL.
2. El-Mawass, N. et al. (2020). "BotDetection in Social Media — A Comparative Survey."
3. Meta: "Signals we use to detect fake accounts", 2023 transparency report.

## Profile picture

- **No profile picture (default egg/avatar):** Strong signal of inauthenticity in most academic datasets.
  In the Bakhshandeh training data, approximately 65% of fakes lack a profile picture versus
  fewer than 8% of genuine accounts.
- **Catfishing pictures:** Profile image does not correlate directly to username/bio content.
  Consider reverse image search as a manual verification step.

## Bio / description

- **Empty bio** is slightly suspicious, but many real users also have empty bios.
- **Bio length** alone is not a strong signal. The content matters far more:
  - Spam keyword hits: "giveaway", "click link", "follow for follow", crypto terms,
    guaranteed income language — *these are much more informative than length*.
- **Bio contains an external URL in combination with young account + low ratio:**
  The URL is often a scam landing page (phishing, fake investment, adult dating).

## Verification badge (blue / verified)

- A genuine platform verification badge is a *very strong* genuine signal.
- Fake accounts almost never obtain legitimate platform verification.
- **Critical rule (per project specification):** The system MUST NOT derive `is_verified` from
  follower count or any other proxy. Verification status is only used when the API explicitly provides it.
- If `is_verified` is unknown/missing, treat it as missing and do NOT fill it in.

## Private account status

- "Private" (locked) accounts are commonly genuine, as humans often limit viewership.
- Fake accounts typically want maximum reach so they are *unlocked* more often.
  However, many fake accounts have started using private status to evade automated detection scrapers,
  so this signal is weaker in 2024-2025 datasets.

## Manual follow-up verification steps for borderline cases

1. **Reverse image search** the profile picture using Google Images or Yandex.
2. **Check recent 30 posts** if visible: Are they all identical captions? Do they all post within minutes of each other?
3. **Check comments**: Do they post identical comments ("Great post!", "Love it!") on many accounts rapidly?
4. **Cross-platform username search**: Has the same username been reported elsewhere for spam/scams?
