"""
Model Architecture for Multilingual Email Assessment
=====================================================

Multi-task XLM-RoBERTa model with:
  - Grammar regression head (0-5)
  - Content regression head (0-4)
  - CEFR ordinal classification head (CORAL approach)
  - Learnable uncertainty-based task weighting (Kendall et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from utils import (
    CEFR_LEVELS, CEFR_TO_ORDINAL, MODEL_NAME, MAX_LENGTH,
    cefr_to_ordinal_labels, ordinal_probs_to_cefr
)


class EmailAssessmentModel(nn.Module):
    """
    Multi-task model for email assessment using XLM-RoBERTa backbone.
    
    Predicts:
        - Grammar score (regression, 0-5)
        - Content score (regression, 0-4)
        - CEFR level (ordinal classification via CORAL)
    """
    
    def __init__(self, model_name=MODEL_NAME, hidden_dim=256, dropout=0.1, freeze_backbone=False):
        super().__init__()
        
        # Backbone: XLM-RoBERTa
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size  # 768 for base
        
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Backbone encoder frozen.")
        
        # Shared projection layer
        self.shared_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        # Grammar regression head (output: single continuous value)
        self.grammar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Content regression head (output: single continuous value)
        self.content_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # CEFR ordinal head (CORAL: 5 binary classifiers for 6 levels)
        # Uses shared weight with bias offsets for ordinality
        self.cefr_shared_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.cefr_biases = nn.Parameter(torch.zeros(5))  # 5 thresholds for 6 classes
        
        # Initialize biases to encourage ordering
        with torch.no_grad():
            self.cefr_biases.data = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) attention mask
            
        Returns:
            grammar_out: (batch_size, 1) grammar predictions
            content_out: (batch_size, 1) content predictions
            cefr_logits: (batch_size, 5) CEFR ordinal logits
        """
        # Encode text
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        # Shared projection
        shared = self.shared_projection(cls_output)  # (batch, hidden_dim)
        
        # Task-specific predictions
        grammar_out = self.grammar_head(shared)  # (batch, 1)
        content_out = self.content_head(shared)  # (batch, 1)
        
        # CEFR ordinal logits (CORAL approach)
        cefr_features = self.cefr_shared_weight(shared)  # (batch, 1)
        cefr_logits = cefr_features + self.cefr_biases  # (batch, 5) - broadcasting
        
        return grammar_out, content_out, cefr_logits
    
    def predict_cefr_level(self, cefr_logits):
        """
        Convert CEFR ordinal logits to CEFR level string.
        
        Args:
            cefr_logits: (5,) or (batch, 5) logits
            
        Returns:
            str or list of str: CEFR level(s)
        """
        probs = torch.sigmoid(cefr_logits)
        
        if probs.dim() == 1:
            return ordinal_probs_to_cefr(probs.detach().cpu().numpy())
        else:
            return [
                ordinal_probs_to_cefr(p.detach().cpu().numpy())
                for p in probs
            ]


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable uncertainty-based weighting.
    
    Combines:
        - MSE loss for grammar regression
        - MSE loss for content regression
        - CORAL ordinal loss for CEFR
    
    Uses homoscedastic uncertainty weighting (Kendall et al., 2018):
        L_total = sum_i (1/(2*sigma_i^2) * L_i + log(sigma_i))
    
    where sigma_i are learned per-task noise parameters.
    """
    
    def __init__(self):
        super().__init__()
        # Log-variance parameters (initialized to 0 = equal weighting)
        self.log_vars = nn.Parameter(torch.zeros(3))
    
    def forward(self, grammar_pred, grammar_true, 
                content_pred, content_true,
                cefr_logits, cefr_ordinal_labels):
        """
        Compute combined multi-task loss.
        
        Args:
            grammar_pred: (batch, 1) predicted grammar scores
            grammar_true: (batch,) true grammar scores
            content_pred: (batch, 1) predicted content scores
            content_true: (batch,) true content scores
            cefr_logits: (batch, 5) CEFR ordinal logits
            cefr_ordinal_labels: (batch, 5) binary ordinal labels
        """
        # Grammar MSE loss
        grammar_loss = F.mse_loss(grammar_pred.squeeze(-1), grammar_true.float())
        
        # Content MSE loss
        content_loss = F.mse_loss(content_pred.squeeze(-1), content_true.float())
        
        # CEFR CORAL ordinal loss (binary cross-entropy on cumulative probabilities)
        cefr_loss = F.binary_cross_entropy_with_logits(
            cefr_logits, cefr_ordinal_labels.float()
        )
        
        # Uncertainty-weighted combination
        # L = 1/(2*exp(s)) * L_task + s/2  where s = log(sigma^2)
        precision_grammar = torch.exp(-self.log_vars[0])
        precision_content = torch.exp(-self.log_vars[1])
        precision_cefr = torch.exp(-self.log_vars[2])
        
        total_loss = (
            precision_grammar * grammar_loss + self.log_vars[0] +
            precision_content * content_loss + self.log_vars[1] +
            precision_cefr * cefr_loss + self.log_vars[2]
        )
        
        return total_loss, {
            "grammar_loss": grammar_loss.item(),
            "content_loss": content_loss.item(),
            "cefr_loss": cefr_loss.item(),
            "total_loss": total_loss.item(),
            "log_var_grammar": self.log_vars[0].item(),
            "log_var_content": self.log_vars[1].item(),
            "log_var_cefr": self.log_vars[2].item(),
        }


class EmailDataset(Dataset):
    """
    PyTorch Dataset for email assessment data.
    
    Expects a DataFrame with columns: text, grammar, content, cefr
    """
    
    def __init__(self, dataframe, tokenizer, max_length=MAX_LENGTH):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["text"])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
        # Add labels if available
        if "grammar" in row.index:
            item["grammar"] = torch.tensor(float(row["grammar"]), dtype=torch.float)
        if "content" in row.index:
            item["content"] = torch.tensor(float(row["content"]), dtype=torch.float)
        if "cefr" in row.index:
            cefr_level = str(row["cefr"])
            ordinal_label = cefr_to_ordinal_labels(cefr_level)
            item["cefr_label"] = torch.tensor(CEFR_TO_ORDINAL.get(cefr_level, 0), dtype=torch.long)
            item["cefr_ordinal"] = torch.tensor(ordinal_label, dtype=torch.float)
        
        return item


if __name__ == "__main__":
    # Quick test
    print("Testing model architecture...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EmailAssessmentModel(MODEL_NAME)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    sample_text = "Dear Manager, I would like to request a day off next week."
    encoding = tokenizer(sample_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    
    with torch.no_grad():
        grammar, content, cefr_logits = model(encoding["input_ids"], encoding["attention_mask"])
    
    print(f"\nSample prediction:")
    print(f"  Grammar: {grammar.item():.2f}")
    print(f"  Content: {content.item():.2f}")
    print(f"  CEFR logits: {cefr_logits.squeeze().numpy()}")
    print(f"  CEFR level: {model.predict_cefr_level(cefr_logits.squeeze())}")
    
    print("\nModel architecture test passed!")
