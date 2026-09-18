# Spam / Bot Behavior Taxonomy

**Source (traceable references for each bot type):**

1. **Traditional spam bots** (Type 1):
   - Wang et al. (2013). ICWSM "Don't Follow Me: Spam Detection in Twitter"
   - Behavior: High-volume tweeting/posting of links to scam/malware/pharma/phishing URLs.
   - Account fingerprint: Very old or very new accounts, low following, massive following, URLs in every bio/post.

2. **Social spambots 2.0** (Type 2):
   - Cresci et al. (2017). "Social Fingerprinting: Detection of Spambot Groups through DNA-inspired Behavioral Modeling"
     (IEEE TDSC) and the related Cresci-2017 dataset.
   - Behavior: Content-aware, mimics real-user posts, follows/unfollows in waves (follow-churn),
     targeted @mentions of influencers.
   - Harder to detect using profile features only; requires post/engagement temporal data.

3. **Fake follower bots** (Type 3):
   - Jiang et al. (2014). "The Anonymity of Social Spammers: A Measurement Study."
   - Behavior: Shell accounts, 0 posts, no profile picture, created in synchronized batches,
     used solely to inflate another account's follower count.
   - Fingerprint: 0 posts, 0 following, <100 followers, no picture, random username.
   Our profile-based model is well suited for this class.

4. **Impersonation / catfishing accounts** (Type 4):
   - Thomas et al. (2017). "Platforms vs. Fake Accounts: The Never-Ending Arms Race"
   - Behavior: Impersonates a real public figure or creates a fake persona for romance scams / pig-butchering.
   - Fingerprint: Often uses stolen profile pictures (reverse image search is your best manual tool),
     bio points to a fraudulent investment site or crypto wallet.

5. **Compromised genuine accounts (Type 5):**
   - Very hard to detect with profile features alone. An account that was genuine for 5 years then
     suddenly spams was not in our training data distribution. Look for sudden step-changes in behavior:
     posting frequency, language, follower/following deltas.

## Fraud typologies commonly observed in 2024-2025

1. **Crypto giveaway scams**: "Send 0.1 ETH → receive 1.0 ETH back" — bio contains wallet
   link, often uses impersonation of Elon Musk / CZ / Vitalik / famous crypto accounts.
2. **Romance scams / pig butchering**: Long-form messaging via platform, moves to WhatsApp,
   eventually requests money for fake investment / medical emergency.
3. **Fake investment platforms**: "High-frequency AI trading bot guaranteed 5% daily return" — account
   often has a fake "trading guru" persona and lots of fake profit screenshots.
4. **Fake verification / account-rescue phishing**: "Your account will be deleted — click to verify" DMs.
5. **Follow-for-sale / engagement farms**: Sell likes, followers, comments via DMs or bio links.
