# Suspicious Account Indicator: Account Age

**Source**:
1. Stringhini, G., Kruegel, C., Vigna, G. (2010). "Detecting Spammers on Twitter." ACSAC 2010.
2. Yang, M.-C., et al. (2019). "Uncovering Large Groups of Active Malicious Accounts in Online Social Networks."
3. Bakhshandeh (2019) — training set bias: fake accounts median age <30 days, genuine median >200 days.

## What this indicator measures

`account_age_days`: the number of days since the account was created.

## Suspicious patterns

1. **Very young accounts (< 7 days):**
   Accounts younger than one week are heavily overrepresented in the fake/fraud class.
   Many fake accounts are created in bulk, used for a few days for:
   - spam comments on giveaway posts
   - review manipulation
   - phishing DM sending
   - fake engagement (like/retweet boosting farms)
   Risk increases heavily when combined with other signals.

2. **Young accounts (< 30 days) + high following count:**
   New accounts with >500 following in the first month typically engage in follow-back farming.

3. **Aged accounts that suddenly spike in activity:**
   Older accounts are not automatically trustworthy. Compromised accounts (phished / bought on marketplaces)
   often show: months of inactivity, then burst of spam posting, or follower/following count
   increasing by 1,000% in 48 hours.

## Positive signals

- Account ≥ 1 year old with consistent posting history
- Account ≥ 6 months old with slow, steady follower growth (not step-function spikes)

## Caveats

- New real people create accounts every day. Never use age alone as a classifier.
- `account_age_days` may be IMPUTED or MISSING when APIs do not expose creation date.
  In this system, missing age has an explicit missing-indicator feature so the model
  knows it is uncertain.
