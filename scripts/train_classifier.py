import os
import pandas as pd
import json
import hashlib
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, brier_score_loss, log_loss

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings
from src.classification.model import build_base_classifier, build_calibrated_classifier, save_model

def train():
    splits_dir = os.path.join(settings.DATA_DIR, "splits")
    train_path = os.path.join(splits_dir, "train.csv")
    cal_path = os.path.join(splits_dir, "calibration.csv")
    dev_path = os.path.join(splits_dir, "dev.csv")
    
    if not all(os.path.exists(p) for p in [train_path, cal_path, dev_path]):
        print("Data splits not found. Please run prepare_data_splits.py first.")
        sys.exit(1)
        
    print("Loading data splits...")
    train_df = pd.read_csv(train_path)
    cal_df = pd.read_csv(cal_path)
    dev_df = pd.read_csv(dev_path)
    
    # 1. Train Base Classifier
    print("Training base classifier on train.csv...")
    base_clf = build_base_classifier()
    
    X_train = train_df['customer_text'].fillna('').tolist()
    y_train = train_df['intent'].tolist()
    
    base_clf.fit(X_train, y_train)
    save_model(base_clf, "base_classifier.joblib")
    print("Base classifier trained and saved.")
    
    # 2. Train Calibrated Classifier
    print("Calibrating classifier on calibration.csv...")
    X_cal = cal_df['customer_text'].fillna('').tolist()
    y_cal = cal_df['intent'].tolist()
    
    calibrated_clf = build_calibrated_classifier(base_clf)
    calibrated_clf.fit(X_cal, y_cal)
    save_model(calibrated_clf, "calibrated_classifier.joblib")
    print("Calibrated classifier trained and saved.")
    
    # 3. Evaluate on Dev
    print("Evaluating on dev.csv...")
    X_dev = dev_df['customer_text'].fillna('').tolist()
    y_dev = dev_df['intent'].tolist()
    
    y_pred = calibrated_clf.predict(X_dev)
    y_proba = calibrated_clf.predict_proba(X_dev)
    
    acc = accuracy_score(y_dev, y_pred)
    mac_f1 = f1_score(y_dev, y_pred, average='macro', zero_division=0)
    wt_f1 = f1_score(y_dev, y_pred, average='weighted', zero_division=0)
    bacc = balanced_accuracy_score(y_dev, y_pred)
    
    # Log loss requires predicting proba for all classes
    ll = log_loss(y_dev, y_proba)
    
    metrics = {
        "accuracy": acc,
        "macro_f1": mac_f1,
        "weighted_f1": wt_f1,
        "balanced_accuracy": bacc,
        "log_loss": ll
    }
    
    print("\nDev Set Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    metrics_path = os.path.join(settings.ARTIFACTS_DIR, "classifier_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    train()
