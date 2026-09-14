import os
import pandas as pd
import numpy as np
import hashlib
import json
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.settings import settings

def assign_pseudo_intent(text: str) -> str:
    """Simple keyword-based pseudo-labeler to bootstrap training data."""
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

def create_splits():
    print("Loading data...")
    # Load TWCS
    twcs_path = os.path.join(settings.BASE_DIR, "twcs", "twcs.csv")
    golden_path = os.path.join(settings.BASE_DIR, "amazonhelp_golden_set_final.csv")
    
    if not os.path.exists(twcs_path):
        raise FileNotFoundError(f"Source data not found at {twcs_path}")
        
    twcs = pd.read_csv(twcs_path)
    golden = pd.read_csv(golden_path)
    
    print("Processing AmazonHelp conversations...")
    # Get all amazonhelp tweets
    brand_tweets = twcs[twcs['author_id'] == 'AmazonHelp']
    
    # Merge to find the customer tweets they responded to
    customer_tweets = twcs[twcs['tweet_id'].isin(brand_tweets['in_response_to_tweet_id'])]
    
    merged = pd.merge(
        customer_tweets[['tweet_id', 'author_id', 'text', 'in_response_to_tweet_id']],
        brand_tweets[['tweet_id', 'author_id', 'text', 'in_response_to_tweet_id']],
        left_on='tweet_id',
        right_on='in_response_to_tweet_id',
        suffixes=('_customer', '_brand')
    )
    
    # Rename for consistency
    corpus_raw = merged[['tweet_id_customer', 'text_customer', 'text_brand']].copy()
    corpus_raw = corpus_raw.rename(columns={
        'tweet_id_customer': 'tweet_id',
        'text_customer': 'customer_text',
        'text_brand': 'brand_response'
    })
    
    # Drop NAs
    corpus_raw = corpus_raw.dropna(subset=['customer_text', 'brand_response'])
    
    print(f"Total AmazonHelp interactions found: {len(corpus_raw)}")
    
    # CRITICAL: Remove ANY overlap with the locked golden set
    golden_customer_texts = set(golden['text_customer'].str.lower().str.strip())
    corpus_raw['clean_lower'] = corpus_raw['customer_text'].str.lower().str.strip()
    
    # Filter out exact matches
    corpus_raw = corpus_raw[~corpus_raw['clean_lower'].isin(golden_customer_texts)]
    
    # If golden set has tweet_ids, remove them too (assuming it might be named differently)
    if 'tweet_id' in golden.columns:
        golden_ids = set(golden['tweet_id'])
        corpus_raw = corpus_raw[~corpus_raw['tweet_id'].isin(golden_ids)]
        
    corpus_raw = corpus_raw.drop(columns=['clean_lower'])
    print(f"Interactions after removing golden set overlap: {len(corpus_raw)}")
    
    # We need 6,000 for training classifier, and the REST (or all of it) for retrieval.
    # Let's shuffle the entire corpus first.
    corpus_raw = corpus_raw.sample(frac=1, random_state=settings.RANDOM_SEED)
    
    # Retrieval Corpus (entire corpus minus golden set)
    retrieval_corpus = corpus_raw.copy()
    retrieval_corpus['intent'] = retrieval_corpus['customer_text'].apply(assign_pseudo_intent)
    
    # The first 6,000 for training classifier
    classifier_data = corpus_raw.iloc[:6000].copy()
    classifier_data['intent'] = classifier_data['customer_text'].apply(assign_pseudo_intent)
    
    # Split classifier data: Train (60%), Cal (20%), Dev (20%)
    train_data, temp_data = train_test_split(classifier_data, test_size=0.4, random_state=settings.RANDOM_SEED, stratify=classifier_data['intent'])
    cal_data, dev_data = train_test_split(temp_data, test_size=0.5, random_state=settings.RANDOM_SEED, stratify=temp_data['intent'])
    
    # Save datasets
    splits_dir = os.path.join(settings.DATA_DIR, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    retrieval_path = os.path.join(splits_dir, "retrieval_corpus.csv")
    train_path = os.path.join(splits_dir, "train.csv")
    cal_path = os.path.join(splits_dir, "calibration.csv")
    dev_path = os.path.join(splits_dir, "dev.csv")
    golden_out_path = os.path.join(splits_dir, "golden_test.csv")
    
    retrieval_corpus.to_csv(retrieval_path, index=False)
    train_data.to_csv(train_path, index=False)
    cal_data.to_csv(cal_path, index=False)
    dev_data.to_csv(dev_path, index=False)
    
    # Copy golden set verbatim to splits dir to represent the locked test set
    golden.to_csv(golden_out_path, index=False)
    
    # Generate Manifest
    def hash_file(filepath):
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
        
    manifest = {
        "version": "1.0",
        "seed": settings.RANDOM_SEED,
        "splits": {
            "retrieval_corpus": {"rows": len(retrieval_corpus), "hash": hash_file(retrieval_path)},
            "train": {"rows": len(train_data), "hash": hash_file(train_path)},
            "calibration": {"rows": len(cal_data), "hash": hash_file(cal_path)},
            "dev": {"rows": len(dev_data), "hash": hash_file(dev_path)},
            "golden_test": {"rows": len(golden), "hash": hash_file(golden_out_path), "locked": True},
        }
    }
    
    with open(os.path.join(splits_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Data splitting complete. Manifest saved at {os.path.join(splits_dir, 'manifest.json')}")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    create_splits()
