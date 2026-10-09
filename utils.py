import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

CEFR_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
CEFR_TO_NUM = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
NUM_TO_CEFR = {v: k for k, v in CEFR_TO_NUM.items()}
CEFR_TO_ORDINAL = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
GRAMMAR_RANGE = (0, 5)
CONTENT_RANGE = (0, 4)
MODEL_NAME = 'xlm-roberta-base'
MAX_LENGTH = 512

def cefr_to_ordinal_labels(cefr_level):
    if cefr_level not in CEFR_TO_ORDINAL:
        return [0] * 5
    num_ones = CEFR_TO_ORDINAL[cefr_level]
    return [1] * num_ones + [0] * (5 - num_ones)

def ordinal_probs_to_cefr(probs, threshold=0.5):
    # probs is list or array of 5 probabilities
    num_ones = sum(1 for p in probs if p >= threshold)
    return CEFR_LEVELS[num_ones]

def compute_metrics(y_true, y_pred, task='regression'):
    metrics = {}
    if task == 'regression':
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
            metrics['pearson'], _ = pearsonr(y_true, y_pred)
        else:
            metrics['pearson'] = 0.0
    return metrics
