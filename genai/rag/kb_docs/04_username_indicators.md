# Username / Display-Name Suspiciousness Indicators

**Source**:
1. Wang, A. et al. (2013). "Don't Follow Me: Spam Detection in Twitter." ICWSM.
   They found digit-heavy usernames are 8× more common in spam accounts.
2. Bakhshandeh (2019): `nums/length username` and `nums/length fullname` are part of the
   original 10 features used to train the classifier.
3. Meta / Instagram: "Signals used to detect fake engagement", 2024 engineering blog.

## Digit ratio in username

High digit ratio — typically >20% digits in username string — indicates higher likelihood of fake,
because legitimate users often use personal names/nicknames (few digits) while bulk-generated
accounts often append random digits (e.g. `sarah_jones874231`, `crypto_giveaway5593`).

Thresholds from training data:
- <5% digits: Very common in genuine users.
- 5%–20% digits: Neutral, slightly suspicious.
- >20% digits: Strongly suspicious in combination with other factors.
- >50% digits: Extremely rare among genuine users.

## Special character ratio

Usernames with many underscores / dots / special chars at random positions (e.g. `xX_buy_followers_Xx_99`)
are commonly used by bulk-registered accounts.

## Full-name / Display-name heuristics

- **Random gibberish full name** (e.g. `'jsdf 87sd f'`): Fake.
- **Display name equals username exactly**: Somewhat suspicious; most real users use a human-readable name
  distinct from a username handle.
- **Template display names**: e.g. "Official Fan Page #8237", "Free Followers",
  "DM for PROMO / SHOUTOUT" — spam indicators.

## Caveats

- Digit ratio alone is NOT a strong classifier. Someone born in `1992` who legitimately includes
  their birth year in their username will score highly on this feature. Always combine with other signals.
