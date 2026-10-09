"""
Inference Pipeline for Multilingual Email Assessment
====================================================
Authors: Research Engineering Team, SHL Assessment

Supports dual evaluation backends:
1. Deep Transformer Neural Pipeline: Multi-task XLM-RoBERTa architecture
2. High-Dimensional Statistical Linguistic Pipeline: Multi-Task Orthographic & Lexical Ridge Ensemble
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import pickle

from utils import (
    MODEL_NAME, MAX_LENGTH, GRAMMAR_RANGE, CONTENT_RANGE,
    CEFR_LEVELS, CEFR_TO_NUM, NUM_TO_CEFR, ordinal_probs_to_cefr
)

def prompt_overlap(text, p):
    w1 = set(str(text).lower().split())
    w2 = set(str(p).lower().split())
    jaccard = len(w1 & w2) / (len(w1 | w2) + 1e-5)
    recall = len(w1 & w2) / (len(w2) + 1e-5)
    prec = len(w1 & w2) / (len(w1) + 1e-5)
    return [jaccard, recall, prec]

def run_statistical_inference(test_df, model_dir):
    """Run inference using the statistical linguistic feature representation model."""
    print("Executing statistical multilingual feature pipeline...")
    with open(os.path.join(model_dir, 'statistical_model.pkl'), 'rb') as f:
        artifacts = pickle.load(f)
        
    vec_c = artifacts['vec_char']
    vec_w = artifacts['vec_word']
    mg = artifacts['model_grammar']
    mc = artifacts['model_content']
    mce = artifacts['model_cefr']
    
    X_c = vec_c.transform(test_df['text']).toarray()
    X_w = vec_w.transform(test_df['text']).toarray()
    
    n_words = np.log1p(test_df['text'].apply(lambda x: len(str(x).split())).values).reshape(-1, 1)
    n_chars = np.log1p(test_df['text'].apply(lambda x: len(str(x))).values).reshape(-1, 1)
    p_feats = np.array([prompt_overlap(t, p) for t, p in zip(test_df['text'], test_df['questionStatement'])])
    
    X_all = np.hstack([X_c, X_w, n_words, n_chars, p_feats])
    
    pred_g = mg.predict(X_all)
    pred_c = mc.predict(X_all)
    pred_ce = mce.predict(X_all)
    
    disc_g = np.clip(np.round(pred_g), GRAMMAR_RANGE[0], GRAMMAR_RANGE[1]).astype(int)
    disc_c = np.clip(np.round(pred_c), CONTENT_RANGE[0], CONTENT_RANGE[1]).astype(int)
    disc_ce = np.clip(np.round(pred_ce), 1, 6).astype(int)
    cefr_str = [NUM_TO_CEFR[v] for v in disc_ce]
    
    return disc_g, disc_c, cefr_str

def run_transformer_inference(test_df, model_dir, batch_size=16):
    """Run inference using the neural XLM-RoBERTa model."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from model import EmailAssessmentModel, EmailDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing transformer pipeline on device: {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EmailAssessmentModel(MODEL_NAME)
    
    model_path = os.path.join(model_dir, "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {model_path}. Running initialized architecture.")
        
    model.to(device)
    model.eval()
    
    dataset = EmailDataset(test_df, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    grammar_preds, content_preds, cefr_preds = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            grammar_out, content_out, cefr_out = model(input_ids, attention_mask)
            
            g_preds = grammar_out.squeeze().cpu().numpy()
            c_preds = content_out.squeeze().cpu().numpy()
            cefr_probs = torch.sigmoid(cefr_out).cpu().numpy()
            
            if g_preds.ndim == 0:
                g_preds = [g_preds.item()]
                c_preds = [c_preds.item()]
                cefr_probs = [cefr_probs]
                
            for g, c, cefr_p in zip(g_preds, c_preds, cefr_probs):
                g_val = np.clip(np.round(g), GRAMMAR_RANGE[0], GRAMMAR_RANGE[1])
                c_val = np.clip(np.round(c), CONTENT_RANGE[0], CONTENT_RANGE[1])
                cefr_level = ordinal_probs_to_cefr(cefr_p)
                
                grammar_preds.append(int(g_val))
                content_preds.append(int(c_val))
                cefr_preds.append(cefr_level)
                
    return grammar_preds, content_preds, cefr_preds

def main():
    parser = argparse.ArgumentParser(description="Multilingual Email Assessment Inference Pipeline")
    parser.add_argument('--model-dir', type=str, default='./model_checkpoint/', help='Path to model checkpoints')
    parser.add_argument('--test-file', type=str, default='test_dataset.xlsx', help='Path to test dataset (Excel or CSV)')
    parser.add_argument('--output-file', type=str, default='predictions.csv', help='Path to output predictions CSV')
    parser.add_argument('--backend', type=str, default='statistical', choices=['statistical', 'transformer'],
                        help='Inference engine backend')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for neural evaluation')
    args = parser.parse_args()

    print(f"Loading test dataset from: {args.test_file}")
    if args.test_file.endswith('.csv'):
        test_df = pd.read_csv(args.test_file, encoding='utf-8')
    else:
        test_df = pd.read_excel(args.test_file)
        
    if 'questionID' not in test_df.columns:
        test_df['questionID'] = test_df.index

    # Check for statistical model artifacts or fallback
    stat_path = os.path.join(args.model_dir, 'statistical_model.pkl')
    if args.backend == 'statistical' and not os.path.exists(stat_path):
        print("Statistical artifacts not found, running neural transformer backend...")
        args.backend = 'transformer'
        
    if args.backend == 'statistical':
        pred_g, pred_c, pred_ce = run_statistical_inference(test_df, args.model_dir)
    else:
        pred_g, pred_c, pred_ce = run_transformer_inference(test_df, args.model_dir, args.batch_size)
        
    results_df = pd.DataFrame({
        'questionID': test_df['questionID'],
        'grammar_pred': pred_g,
        'content_pred': pred_c,
        'cefr_pred': pred_ce
    })
    
    results_df.to_csv(args.output_file, index=False, encoding='utf-8')
    print(f"Saved {len(results_df)} predictions to: {args.output_file}")

if __name__ == '__main__':
    main()
