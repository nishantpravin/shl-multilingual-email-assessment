"""
Training Script for Multilingual Email Assessment Model
========================================================

Trains the multi-task XLM-RoBERTa model on synthetic training data.

Features:
  - Stratified train/val split by CEFR level
  - AdamW optimizer with linear warmup and decay
  - Gradient clipping
  - Mixed precision training (CUDA)
  - Early stopping
  - Comprehensive validation metrics
  - Training log saved as JSON
"""

import argparse
import json
import logging
import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from model import EmailAssessmentModel, MultiTaskLoss, EmailDataset
from utils import (
    MODEL_NAME, MAX_LENGTH, CEFR_LEVELS, CEFR_TO_NUM,
    ordinal_probs_to_cefr, compute_metrics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_validation_metrics(model, dataloader, device):
    """Compute all validation metrics."""
    model.eval()
    
    all_grammar_true = []
    all_grammar_pred = []
    all_content_true = []
    all_content_pred = []
    all_cefr_true = []
    all_cefr_pred = []
    total_loss = 0
    n_batches = 0
    
    loss_fn = MultiTaskLoss().to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            grammar_true = batch["grammar"].to(device)
            content_true = batch["content"].to(device)
            cefr_ordinal = batch["cefr_ordinal"].to(device)
            cefr_label = batch["cefr_label"].to(device)
            
            grammar_pred, content_pred, cefr_logits = model(input_ids, attention_mask)
            
            loss, _ = loss_fn(
                grammar_pred, grammar_true,
                content_pred, content_true,
                cefr_logits, cefr_ordinal
            )
            total_loss += loss.item()
            n_batches += 1
            
            # Collect predictions
            all_grammar_true.extend(grammar_true.cpu().numpy())
            all_grammar_pred.extend(grammar_pred.squeeze(-1).cpu().numpy())
            all_content_true.extend(content_true.cpu().numpy())
            all_content_pred.extend(content_pred.squeeze(-1).cpu().numpy())
            
            # CEFR predictions
            cefr_probs = torch.sigmoid(cefr_logits).cpu().numpy()
            for i in range(len(cefr_probs)):
                pred_level = ordinal_probs_to_cefr(cefr_probs[i])
                all_cefr_pred.append(pred_level)
            
            cefr_labels = cefr_label.cpu().numpy()
            for label in cefr_labels:
                all_cefr_true.append(CEFR_LEVELS[label])
    
    # Compute metrics
    metrics = {"val_loss": total_loss / max(n_batches, 1)}
    
    # Grammar Pearson
    if len(set(all_grammar_pred)) > 1:
        metrics["grammar_pearson"], _ = pearsonr(all_grammar_true, all_grammar_pred)
    else:
        metrics["grammar_pearson"] = 0.0
    metrics["grammar_mse"] = np.mean((np.array(all_grammar_true) - np.array(all_grammar_pred))**2)
    
    # Content Pearson
    if len(set(all_content_pred)) > 1:
        metrics["content_pearson"], _ = pearsonr(all_content_true, all_content_pred)
    else:
        metrics["content_pearson"] = 0.0
    metrics["content_mse"] = np.mean((np.array(all_content_true) - np.array(all_content_pred))**2)
    
    # CEFR metrics
    cefr_true_num = [CEFR_TO_NUM[c] for c in all_cefr_true]
    cefr_pred_num = [CEFR_TO_NUM[c] for c in all_cefr_pred]
    
    if len(set(cefr_pred_num)) > 1:
        metrics["cefr_pearson"], _ = pearsonr(cefr_true_num, cefr_pred_num)
    else:
        metrics["cefr_pearson"] = 0.0
    
    cefr_exact_match = sum(1 for t, p in zip(all_cefr_true, all_cefr_pred) if t == p)
    metrics["cefr_accuracy"] = cefr_exact_match / len(all_cefr_true)
    
    return metrics


def train(args):
    """Main training function."""
    set_seed(args.seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load training data
    logger.info(f"Loading training data from {args.train_data}...")
    df = pd.read_csv(args.train_data, encoding="utf-8")
    logger.info(f"Loaded {len(df)} training samples")
    logger.info(f"CEFR distribution:\n{df['cefr'].value_counts().sort_index()}")
    
    # Stratified train/val split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=args.seed, stratify=df["cefr"]
    )
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EmailAssessmentModel(
        MODEL_NAME, hidden_dim=256, dropout=0.1,
        freeze_backbone=args.freeze_backbone
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Create datasets and dataloaders
    train_dataset = EmailDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = EmailDataset(val_df, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    
    # Loss function
    loss_fn = MultiTaskLoss().to(device)
    
    # Optimizer
    # Separate learning rates: lower for backbone, higher for task heads
    backbone_params = list(model.encoder.parameters())
    head_params = (
        list(model.shared_projection.parameters()) +
        list(model.grammar_head.parameters()) +
        list(model.content_head.parameters()) +
        list(model.cefr_shared_weight.parameters()) +
        [model.cefr_biases]
    )
    loss_params = list(loss_fn.parameters())
    
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 5},  # Higher LR for heads
        {"params": loss_params, "lr": args.lr * 10},  # Higher LR for loss weights
    ], weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Batch size: {args.batch_size}, LR: {args.lr}, Weight decay: {args.weight_decay}")
    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    best_val_loss = float("inf")
    best_metrics = {}
    patience_counter = 0
    training_log = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_losses = {"grammar_loss": 0, "content_loss": 0, "cefr_loss": 0}
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            grammar_true = batch["grammar"].to(device)
            content_true = batch["content"].to(device)
            cefr_ordinal = batch["cefr_ordinal"].to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast("cuda"):
                    grammar_pred, content_pred, cefr_logits = model(input_ids, attention_mask)
                    loss, loss_dict = loss_fn(
                        grammar_pred, grammar_true,
                        content_pred, content_true,
                        cefr_logits, cefr_ordinal
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                grammar_pred, content_pred, cefr_logits = model(input_ids, attention_mask)
                loss, loss_dict = loss_fn(
                    grammar_pred, grammar_true,
                    content_pred, content_true,
                    cefr_logits, cefr_ordinal
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            
            epoch_loss += loss_dict["total_loss"]
            for k in epoch_losses:
                epoch_losses[k] += loss_dict.get(k, 0)
            n_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.4f}",
                "g_loss": f"{loss_dict['grammar_loss']:.4f}",
                "c_loss": f"{loss_dict['content_loss']:.4f}",
                "cefr_loss": f"{loss_dict['cefr_loss']:.4f}",
            })
        
        avg_train_loss = epoch_loss / n_batches
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        # Validation
        val_metrics = compute_validation_metrics(model, val_loader, device)
        
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_losses": avg_losses,
            "val_loss": val_metrics["val_loss"],
            "grammar_pearson": val_metrics["grammar_pearson"],
            "content_pearson": val_metrics["content_pearson"],
            "cefr_pearson": val_metrics["cefr_pearson"],
            "cefr_accuracy": val_metrics["cefr_accuracy"],
            "learning_rate": scheduler.get_last_lr()[0],
            "log_var_grammar": loss_fn.log_vars[0].item(),
            "log_var_content": loss_fn.log_vars[1].item(),
            "log_var_cefr": loss_fn.log_vars[2].item(),
        }
        training_log.append(epoch_log)
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Grammar r: {val_metrics['grammar_pearson']:.4f} | "
            f"Content r: {val_metrics['content_pearson']:.4f} | "
            f"CEFR r: {val_metrics['cefr_pearson']:.4f} | "
            f"CEFR acc: {val_metrics['cefr_accuracy']:.4f}"
        )
        
        # Early stopping check
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_metrics = val_metrics.copy()
            patience_counter = 0
            
            # Save best model
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
            
            # Save tokenizer
            tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
            
            # Save config
            config = {
                "model_name": MODEL_NAME,
                "hidden_dim": 256,
                "dropout": 0.1,
                "max_length": MAX_LENGTH,
                "best_epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_metrics": {k: float(v) for k, v in best_metrics.items()},
            }
            with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"  -> Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience: {args.patience})")
                break
    
    # Save training log
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model_name": MODEL_NAME,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "seed": args.seed,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
            },
            "best_metrics": {k: float(v) for k, v in best_metrics.items()},
            "training_log": training_log,
        }, f, indent=2)
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best validation metrics:")
    for k, v in best_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"\nModel saved to: {args.output_dir}")
    logger.info(f"Training log saved to: {log_path}")
    
    # Check acceptance criteria
    logger.info(f"\nAcceptance Criteria (on validation set):")
    logger.info(f"  Grammar Pearson >= 0.60: {'PASS' if best_metrics.get('grammar_pearson', 0) >= 0.60 else 'FAIL'} ({best_metrics.get('grammar_pearson', 0):.4f})")
    logger.info(f"  Content Pearson >= 0.60: {'PASS' if best_metrics.get('content_pearson', 0) >= 0.60 else 'FAIL'} ({best_metrics.get('content_pearson', 0):.4f})")
    logger.info(f"  CEFR Pearson >= 0.60:    {'PASS' if best_metrics.get('cefr_pearson', 0) >= 0.60 else 'FAIL'} ({best_metrics.get('cefr_pearson', 0):.4f})")
    logger.info(f"  CEFR Accuracy >= 30%:    {'PASS' if best_metrics.get('cefr_accuracy', 0) >= 0.30 else 'FAIL'} ({best_metrics.get('cefr_accuracy', 0)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Train multilingual email assessment model"
    )
    parser.add_argument("--train-data", type=str, default="training_data.csv",
                        help="Path to training CSV")
    parser.add_argument("--output-dir", type=str, default="./model_checkpoint/",
                        help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone encoder and only train task heads (much faster on CPU)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
