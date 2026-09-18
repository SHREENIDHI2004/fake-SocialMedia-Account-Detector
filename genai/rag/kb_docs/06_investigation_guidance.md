# Investigation Guidance: Manual Verification Steps

This document provides step-by-step instructions for a human reviewer who has received
a risk report from the Suspicious Account Risk Assessment System and wants to verify manually.

**Source**: Synthesized from:
1. Instagram / Meta — "How to spot fake accounts" public help pages.
2. FTC (U.S. Federal Trade Commission): "How to Spot, Stop, and Report Romance Scams", 2023.
3. FBI IC3 — Social Media Fraud PSA, 2024.

## Step 1: Reverse image search the profile picture

If the risk score is high AND profile picture exists, do a reverse image search
(Google Images, Yandex, TinEye). Stolen celebrity photos and stock photos are the #1 clue for
romance scams and impersonation accounts.

**If result:** Photo matches a known public figure / stock website — extremely strong positive signal for fake.

## Step 2: Review recent posts / captions

Look at the 15 most recent posts if the account is public:
- Are all posts identical? (e.g., reposting the same graphic over and over)
- Are all captions identical templates with minor variations?
- Do all posts tag random people / use the same 30 hashtags?
- Are the posts time-stamped suspiciously close together (<2 minutes apart)?
- Does every post link to an external crypto site, storefront, or giveaway?

## Step 3: Review follower and following lists

If public:
- Do followers have profile pictures? Or are 90% of them also new, pictureless accounts with random usernames?
  → Bought followers signal
- Was 80% of the following list followed in a single 48-hour window? → follow-churn bot

## Step 4: Bio text sanity check

- Are there common spam keywords from our known list? (giveaway, free, follow-back, investment, guaranteed, crypto, earn money, etc.)
- Does the bio claim to be a public figure but the account is unverified and has <10K followers?
- Does the bio link to a suspicious URL?
  - Check WHOIS of the domain: registered in the last 6 months? Privacy-protected? Non-standard TLD?
  - Scan link with VirusTotal / URLVoid before clicking.

## Step 5: Cross-platform search

- Copy the username and search across other platforms (Twitter/X, TikTok, Facebook, LinkedIn, Telegram).
- If the same username appears on multiple platforms with identical bios recently created,
  it is often a coordinated fake persona network.

## Step 6: DM / Comment interaction pattern

If visible:
- Do they post generic comments on high-follower accounts?
- Do they unsolicited-DM users with a "too good to be true" offer?
- Do they request moving the conversation off-platform to WhatsApp / Telegram / email quickly?
  → Very high risk of scam.

## Step 7: Make a decision

- **Low risk + no manual red flags**: Treat as genuine
- **Medium risk + no manual red flags**: Monitor / pass to second reviewer
- **Medium risk + any manual red flag**: Report to platform as suspicious
- **High risk (≥70%) + any manual red flag**: Report immediately
- **High risk + no manual red flags found**: Treat as borderline, re-scan in 7 days to look for behavior changes
