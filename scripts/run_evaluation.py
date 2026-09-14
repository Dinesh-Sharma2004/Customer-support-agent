import os
import pandas as pd
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings
from src.pipeline import RAGPipeline
from sklearn.metrics import accuracy_score, f1_score

def run_evaluation():
    splits_dir = os.path.join(settings.DATA_DIR, "splits")
    golden_path = os.path.join(splits_dir, "golden_test.csv")
    
    if not os.path.exists(golden_path):
        print(f"Golden test set not found at {golden_path}")
        sys.exit(1)
        
    print("Loading Golden Test Set (LOCKED - EVALUATION ONLY)...")
    golden_df = pd.read_csv(golden_path)
    
    print("Initializing Pipeline...")
    pipeline = RAGPipeline()
    
    results = []
    
    # We evaluate only the subset of valid intent for classification metrics
    # But behavior metrics on the whole set
    valid_intents = [
        "Order & Delivery Issues",
        "Prime & Membership",
        "Refunds & Financials",
        "Returns & Exchanges",
        "Account Security & Private Support",
        "Other / Human Review"
    ]
    
    print(f"Running pipeline on {len(golden_df)} golden queries...")
    for idx, row in golden_df.iterrows():
        query = row.get('text_customer', '')
        true_intent = row.get('primary_intent', 'Other / Human Review')
        
        # Ensure true_intent is valid, else map to Other
        if true_intent not in valid_intents:
            true_intent = "Other / Human Review"
            
        res = pipeline.process(session_id=f"eval_{idx}", query=query, is_new_session=True)
        
        predicted_intent = res.get('intent')
        if predicted_intent is None:
            predicted_intent = "Other / Human Review"
            
        results.append({
            "query": query,
            "true_intent": true_intent,
            "predicted_intent": predicted_intent,
            "confidence": res['confidence'],
            "decision": res['decision'],
            "reason": res['reason'],
            "grounded": res['grounded']
        })
        
    eval_df = pd.DataFrame(results)
    
    y_true = eval_df['true_intent']
    y_pred = eval_df['predicted_intent']
    
    acc = accuracy_score(y_true, y_pred)
    mac_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    auto_count = (eval_df['decision'] == 'AUTO_HANDLE').sum()
    esc_count = (eval_df['decision'] == 'ESCALATE').sum()
    
    print("\n=== GOLDEN SET EVALUATION REPORT ===")
    print(f"Classification Accuracy:  {acc:.4f}")
    print(f"Classification Macro-F1:  {mac_f1:.4f}")
    print(f"Auto-handled:             {auto_count} ({auto_count/len(eval_df)*100:.1f}%)")
    print(f"Escalated:                {esc_count} ({esc_count/len(eval_df)*100:.1f}%)")
    
    report_path = os.path.join(settings.ARTIFACTS_DIR, "golden_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "accuracy": acc,
            "macro_f1": mac_f1,
            "auto_handled": int(auto_count),
            "escalated": int(esc_count),
            "total": len(eval_df)
        }, f, indent=4)
        
    eval_df.to_csv(os.path.join(settings.ARTIFACTS_DIR, "golden_predictions.csv"), index=False)
    print(f"\nReport and predictions saved to {settings.ARTIFACTS_DIR}")

if __name__ == "__main__":
    run_evaluation()
