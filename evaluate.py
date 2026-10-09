import argparse
import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error

from utils import CEFR_TO_NUM, CEFR_LEVELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV')
    parser.add_argument('--test-file', type=str, required=True, help='Path to test dataset Excel file')
    parser.add_argument('--output', type=str, default='evaluation_results.json')
    args = parser.parse_args()

    preds_df = pd.read_csv(args.predictions)
    test_df = pd.read_excel(args.test_file)
    
    # Predictions and test sets are aligned 1-to-1 by row index
    if len(preds_df) == len(test_df):
        merged_df = test_df.copy()
        merged_df['grammar_pred'] = preds_df['grammar_pred'].values
        merged_df['content_pred'] = preds_df['content_pred'].values
        merged_df['cefr_pred'] = preds_df['cefr_pred'].values
    else:
        merged_df = pd.merge(test_df, preds_df, on='questionID')
    
    if 'grammar' not in merged_df.columns or 'content' not in merged_df.columns or 'cefr' not in merged_df.columns:
        print("Missing ground truth columns (grammar, content, cefr) in test dataset.")
        return

    # Extract arrays
    grammar_true = merged_df['grammar'].values
    grammar_pred = merged_df['grammar_pred'].values
    
    content_true = merged_df['content'].values
    content_pred = merged_df['content_pred'].values
    
    cefr_true = merged_df['cefr'].values
    cefr_pred = merged_df['cefr_pred'].values
    
    cefr_true_num = np.array([CEFR_TO_NUM.get(x, 1) for x in cefr_true])
    cefr_pred_num = np.array([CEFR_TO_NUM.get(x, 1) for x in cefr_pred])

    # Compute metrics
    grammar_pearson, _ = pearsonr(grammar_true, grammar_pred)
    content_pearson, _ = pearsonr(content_true, content_pred)
    cefr_pearson, _ = pearsonr(cefr_true_num, cefr_pred_num)
    
    cefr_acc = accuracy_score(cefr_true, cefr_pred)
    
    grammar_mse = mean_squared_error(grammar_true, grammar_pred)
    grammar_mae = mean_absolute_error(grammar_true, grammar_pred)
    content_mse = mean_squared_error(content_true, content_pred)
    content_mae = mean_absolute_error(content_true, content_pred)
    
    cefr_cm = confusion_matrix(cefr_true, cefr_pred, labels=CEFR_LEVELS)
    cefr_report = classification_report(cefr_true, cefr_pred, labels=CEFR_LEVELS, output_dict=True, zero_division=0)

    print("=== Evaluation Results ===")
    print(f"Grammar Pearson: {grammar_pearson:.4f}")
    print(f"Content Pearson: {content_pearson:.4f}")
    print(f"CEFR Pearson: {cefr_pearson:.4f}")
    print(f"CEFR Exact Match Accuracy: {cefr_acc*100:.2f}%")
    print("\nGrammar MSE / MAE: {:.4f} / {:.4f}".format(grammar_mse, grammar_mae))
    print("Content MSE / MAE: {:.4f} / {:.4f}".format(content_mse, content_mae))
    
    print("\nCEFR Confusion Matrix:")
    print(cefr_cm)
    
    print("\nAcceptance Criteria Check:")
    print(f"- Grammar Pearson >= 0.60: {'PASS' if grammar_pearson >= 0.60 else 'FAIL'}")
    print(f"- Content Pearson >= 0.60: {'PASS' if content_pearson >= 0.60 else 'FAIL'}")
    print(f"- CEFR Pearson >= 0.60: {'PASS' if cefr_pearson >= 0.60 else 'FAIL'}")
    print(f"- CEFR Exact Match >= 30%: {'PASS' if cefr_acc >= 0.30 else 'FAIL'}")

    results = {
        'grammar': {
            'pearson': grammar_pearson,
            'mse': grammar_mse,
            'mae': grammar_mae
        },
        'content': {
            'pearson': content_pearson,
            'mse': content_mse,
            'mae': content_mae
        },
        'cefr': {
            'pearson': cefr_pearson,
            'accuracy': cefr_acc,
            'confusion_matrix': cefr_cm.tolist(),
            'classification_report': cefr_report
        },
        'acceptance_criteria': {
            'grammar_pearson_pass': bool(grammar_pearson >= 0.60),
            'content_pearson_pass': bool(content_pearson >= 0.60),
            'cefr_pearson_pass': bool(cefr_pearson >= 0.60),
            'cefr_accuracy_pass': bool(cefr_acc >= 0.30)
        }
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()
