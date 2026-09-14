import os
import sys
import json
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings
from src.pipeline import RAGPipeline
from src.classification.model import load_model

def run_evaluation(version="v1"):
    golden_path = os.path.join(settings.BASE_DIR, "amazonhelp_golden_set_final.csv")
    
    if not os.path.exists(golden_path):
        print(f"Golden test set not found at {golden_path}")
        sys.exit(1)
        
    print(f"Loading Golden Test Set for {version} evaluation (LOCKED - EVALUATION ONLY)...")
    golden_df = pd.read_csv(golden_path)
    
    print("Initializing Pipeline...")
    pipeline = RAGPipeline()
    
    if version != "v1":
        model_name = f"calibrated_classifier_{version}.joblib"
        print(f"Injecting experimental classifier: {model_name}")
        pipeline.classifier = load_model(model_name)
    
    results = []
    
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
        
        if true_intent not in valid_intents:
            true_intent = "Other / Human Review"
            
        res = pipeline.process(session_id=f"eval_{version}_{idx}", query=query, is_new_session=True)
        
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
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    
    auto_count = (eval_df['decision'] == 'AUTO_HANDLE').sum()
    esc_count = (eval_df['decision'] == 'ESCALATE').sum()
    
    # Correct vs incorrect Auto-Handle
    auto_df = eval_df[eval_df['decision'] == 'AUTO_HANDLE']
    correct_auto = (auto_df['true_intent'] == auto_df['predicted_intent']).sum()
    incorrect_auto = len(auto_df) - correct_auto
    
    avg_confidence = eval_df['confidence'].mean()
    
    print("\n=== GOLDEN SET EVALUATION REPORT ===")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Macro-F1:           {mac_f1:.4f}")
    print(f"Balanced Accuracy:  {bacc:.4f}")
    print(f"Avg Confidence:     {avg_confidence:.4f}")
    print(f"\nAuto-handled:       {auto_count} ({auto_count/len(eval_df)*100:.1f}%)")
    print(f"  - Correct:        {correct_auto}")
    print(f"  - Incorrect:      {incorrect_auto}")
    print(f"Escalated:          {esc_count} ({esc_count/len(eval_df)*100:.1f}%)")
    
    print("\nClassification Report (Per-Class):")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    suffix = "" if version == "v1" else f"_{version}"
    report_path = os.path.join(settings.ARTIFACTS_DIR, f"golden_evaluation_report{suffix}.json")
    
    out_data = {
        "accuracy": acc,
        "macro_f1": mac_f1,
        "balanced_accuracy": bacc,
        "avg_confidence": avg_confidence,
        "auto_handled": int(auto_count),
        "correct_auto_handle": int(correct_auto),
        "incorrect_auto_handle": int(incorrect_auto),
        "escalated": int(esc_count),
        "total": len(eval_df),
        "classification_report": report,
        "confusion_matrix": conf_matrix
    }
    
    with open(report_path, "w") as f:
        json.dump(out_data, f, indent=4)
        
    eval_df.to_csv(os.path.join(settings.ARTIFACTS_DIR, f"golden_predictions{suffix}.csv"), index=False)
    print(f"\nReport and predictions saved to {settings.ARTIFACTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1", help="Version identifier (e.g., v3)")
    args = parser.parse_args()
    run_evaluation(version=args.version)
