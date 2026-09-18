# Suspicious Account Indicator: Follower/Following Ratio Patterns

**Source**: This document synthesizes findings from:
1. Cresci, S. et al. (2017). "The Paradigm-Shift of Social Spambot Detection: A Survey." WWW 2017 Companion, pp. 963–972.
   DOI: 10.1145/3041021.3055135
2. Bakhshandeh, B. (2019). "Instagram fake spammer genuine accounts dataset." Kaggle, CC BY 3.0.
3. Dey, A., Reddy, H., Manjistha, D., Sinha, N. (2019). "Detection of Fake Accounts in Instagram using Machine Learning." IJCSIT 11(5).
4. Meta / Instagram internal enforcement indicators publicly documented at:
   https://transparency.meta.com/en-gb/policies-and-impact/how-we-remove-accounts/

## What the ratio measures

`follower_following_ratio = followers / following`
`following_follower_ratio = following / followers`

## Expected patterns for genuine human accounts

Typically, for organic real accounts:
- Most personal accounts: 0.2 to 4.0
- Established creators: 5 to 50+
- Microcelebrities: 50 to 500
- Large influencers: 500+ (but very low following)

Accounts rarely follow many more users than follow them back for long, because:
- Humans find it cognitively expensive to maintain >1,000 active followings
- Genuine accounts usually hit an equilibrium after 6-12 months of activity

## Suspicious patterns

1. **Extreme following-follower inversion (ratio < 0.05):**
   Typical follow-churn behavior: account follows 5,000+ people, has ≤100 followers,
   unfollows them after a few days ("follow/unfollow" growth hacking).
   High probability of either a fake growth account or a low-quality scraper.

2. **Ratio near 1.0 with counts exactly at platform limits:**
   E.g., following = 7,499 and followers = 7,499 on Instagram (close to 7,500 following cap).
   Very strong indicator of follow-for-follow churning, not genuine audience growth.

3. **Massive following (≥3,000) with ≤50 posts and low follower counts:**
   Indicates automated follow-bot behavior, particularly when the account is young.

4. **High followers + 0 posts + 0 following** can be legitimate for inactive accounts,
   but if combined with no profile picture, it suggests a purchased / shell account.

## Important caveats

A low ratio BY ITSELF is NOT proof of fakery. Many legitimate accounts are new and are following
many people in their first week. Combine this signal with:
- Account age (very young + low ratio = much more suspicious)
- Bio spam keywords
- Post count / posting frequency
- Profile picture presence
