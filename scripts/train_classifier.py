import os
import pandas as pd
import json
import hashlib
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, brier_score_loss, log_loss

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings
from src.classification.model import build_base_classifier, build_calibrated_classifier, save_model

import argparse

def train(version="v1"):
    splits_dir = os.path.join(settings.DATA_DIR, "splits")
    
    if version == "v1":
        train_path = os.path.join(splits_dir, "train.csv")
        cal_path = os.path.join(splits_dir, "calibration.csv")
        dev_path = os.path.join(splits_dir, "dev.csv")
        model_suffix = ""
    elif version == "v3":
        train_path = os.path.join(settings.ARTIFACTS_DIR, "train_v3.csv")
        cal_path = os.path.join(settings.ARTIFACTS_DIR, "calibration_v3.csv")
        dev_path = None # Dev set skipped for v3 experiment
        model_suffix = "_v3"
    else:
        train_path = os.path.join(splits_dir, f"train_{version}.csv")
        cal_path = os.path.join(splits_dir, f"calibration_{version}.csv")
        dev_path = os.path.join(splits_dir, f"dev_{version}.csv")
        model_suffix = f"_{version}"
    
    if not all(os.path.exists(p) for p in [train_path, cal_path]):
        print(f"Data splits for {version} not found.")
        sys.exit(1)
        
    print(f"Loading data splits for {version}...")
    train_df = pd.read_csv(train_path)
    cal_df = pd.read_csv(cal_path)
    
    dev_df = pd.read_csv(dev_path) if dev_path and os.path.exists(dev_path) else None
    
    # 1. Train Base Classifier
    print(f"Training base classifier on {train_path}...")
    base_clf = build_base_classifier()
    
    X_train = train_df['customer_text'].fillna('').tolist()
    y_train = train_df['intent'].tolist()
    
    base_clf.fit(X_train, y_train)
    save_model(base_clf, f"base_classifier{model_suffix}.joblib")
    print("Base classifier trained and saved.")
    
    # 2. Train Calibrated Classifier
    print(f"Calibrating classifier on {cal_path}...")
    X_cal = cal_df['customer_text'].fillna('').tolist()
    y_cal = cal_df['intent'].tolist()
    
    calibrated_clf = build_calibrated_classifier(base_clf)
    calibrated_clf.fit(X_cal, y_cal)
    save_model(calibrated_clf, f"calibrated_classifier{model_suffix}.joblib")
    print(f"Calibrated classifier trained and saved as calibrated_classifier{model_suffix}.joblib")
    
    # 3. Evaluate on Dev
    if dev_df is not None:
        print(f"Evaluating on {dev_path}...")
        X_dev = dev_df['customer_text'].fillna('').tolist()
        y_dev = dev_df['intent'].tolist()
        
        y_pred = calibrated_clf.predict(X_dev)
        y_proba = calibrated_clf.predict_proba(X_dev)
        
        acc = accuracy_score(y_dev, y_pred)
        mac_f1 = f1_score(y_dev, y_pred, average='macro', zero_division=0)
        wt_f1 = f1_score(y_dev, y_pred, average='weighted', zero_division=0)
        bacc = balanced_accuracy_score(y_dev, y_pred)
        
        ll = log_loss(y_dev, y_proba)
        
        metrics = {
            "accuracy": acc,
            "macro_f1": mac_f1,
            "weighted_f1": wt_f1,
            "balanced_accuracy": bacc,
            "log_loss": ll
        }
        
        print(f"\nDev Set Metrics ({version}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
        metrics_path = os.path.join(settings.ARTIFACTS_DIR, f"classifier_metrics{model_suffix}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1", help="Version identifier for data and models")
    args = parser.parse_args()
    train(version=args.version)
