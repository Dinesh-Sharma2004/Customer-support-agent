import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.settings import settings

def assign_pseudo_intent(text: str) -> str:
    text = str(text).lower()
    if any(k in text for k in ['delivery', 'track', 'shipped', 'arrive', 'package', 'missing']):
        return "Order & Delivery Issues"
    elif any(k in text for k in ['prime', 'membership', 'subscribe', 'video', 'music']):
        return "Prime & Membership"
    elif any(k in text for k in ['refund', 'charge', 'money', 'bank', 'payment', 'bill']):
        return "Refunds & Financials"
    elif any(k in text for k in ['return', 'exchange', 'defective', 'broken', 'wrong item']):
        return "Returns & Exchanges"
    elif any(k in text for k in ['account', 'password', 'hack', 'login', 'security', 'email']):
        return "Account Security & Private Support"
    else:
        return "Other / Human Review"

def prepare_v3():
    twcs_path = os.path.join(settings.BASE_DIR, "twcs", "twcs.csv")
    golden_path = os.path.join(settings.BASE_DIR, "amazonhelp_golden_set_final.csv")
    
    twcs = pd.read_csv(twcs_path)
    golden = pd.read_csv(golden_path)
    
    # 1. Cleaner Customer/Brand Merge
    brand_tweets = twcs[twcs["author_id"] == "AmazonHelp"].copy()
    customer_tweets = twcs[
        twcs["tweet_id"].isin(brand_tweets["in_response_to_tweet_id"])
        & (twcs["author_id"] != "AmazonHelp")
    ].copy()
    
    corpus_raw = (
        customer_tweets[["tweet_id", "text"]]
        .rename(columns={"text": "customer_text"})
        .dropna(subset=["customer_text"])
        .drop_duplicates(subset=["tweet_id"])
    )
    
    # 2. Strict Golden Leakage Exclusion
    golden_ids = set(golden["tweet_id_customer"].astype(str))
    corpus_raw = corpus_raw[~corpus_raw["tweet_id"].astype(str).isin(golden_ids)]
    
    golden_texts = set(golden["text_customer"].fillna("").str.lower().str.strip())
    corpus_raw["normalized_text"] = corpus_raw["customer_text"].fillna("").str.lower().str.strip()
    corpus_raw = corpus_raw[~corpus_raw["normalized_text"].isin(golden_texts)].drop(columns=["normalized_text"])
    
    # 3. Deterministic 2000 Sample
    SAMPLE_SIZE = 2000
    if len(corpus_raw) < SAMPLE_SIZE:
        raise ValueError(f"Only {len(corpus_raw)} eligible tweets available.")
        
    sampled_data = corpus_raw.sample(n=SAMPLE_SIZE, random_state=settings.RANDOM_SEED).reset_index(drop=True)
    
    # Save the exact 2000 sample
    sample_path = os.path.join(settings.ARTIFACTS_DIR, "v3_sample.csv")
    sampled_data.to_csv(sample_path, index=False)
    print(f"Saved {SAMPLE_SIZE} fixed sample to {sample_path}")
    
    # 4. Weak Supervision (Pseudo-labeling)
    sampled_data['intent'] = sampled_data['customer_text'].apply(assign_pseudo_intent)
    
    # 5. Split 1600 / 400
    train_df, cal_df = train_test_split(
        sampled_data,
        test_size=400,
        random_state=settings.RANDOM_SEED,
        stratify=sampled_data['intent']
    )
    
    train_path = os.path.join(settings.ARTIFACTS_DIR, "train_v3.csv")
    cal_path = os.path.join(settings.ARTIFACTS_DIR, "calibration_v3.csv")
    
    train_df.to_csv(train_path, index=False)
    cal_df.to_csv(cal_path, index=False)
    
    print(f"Train split saved to {train_path} ({len(train_df)} rows)")
    print(f"Calibration split saved to {cal_path} ({len(cal_df)} rows)")

if __name__ == "__main__":
    prepare_v3()
