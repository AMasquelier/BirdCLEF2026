from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, accuracy_score, f1_score, classification_report, confusion_matrix
import torch.nn as nn
import torch
import numpy as np



def AUC(targets, outputs, verbose=False):
    targets = (targets>0).astype(float)
    num_classes = targets.shape[1]
    scored_classes = (np.sum(targets,axis=0) > 0)
    auc = roc_auc_score(targets[:,scored_classes], outputs[:,scored_classes], average='macro')
    return auc


def AUC_fast(targets, outputs, verbose=False):
    # Source : Claude
    targets = (targets > 0).astype(np.float32)
    # Only score classes with at least one positive AND one negative
    pos_counts = targets.sum(axis=0)
    neg_counts = targets.shape[0] - pos_counts
    scored = (pos_counts > 0) & (neg_counts > 0)

    if not scored.any():
        return float('nan')

    t = targets[:, scored]
    o = outputs[:, scored]
    n_pos = pos_counts[scored]
    n_neg = neg_counts[scored]

    # Rank along samples axis for each class (average ranks for ties)
    # argsort-based ranking, vectorized across classes
    order = np.argsort(o, axis=0, kind='stable')
    ranks = np.empty_like(order, dtype=np.float64)
    arange = np.arange(1, o.shape[0] + 1, dtype=np.float64)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(arange, order.shape), axis=0)

    # Handle ties: average rank within equal-value groups per column
    # Sort values to detect ties
    sorted_o = np.take_along_axis(o, order, axis=0)
    # For each column, compute average ranks for tied groups
    # Simpler approach: use scipy-like tie correction via cumulative mean over equal runs
    # For speed, skip tie correction if outputs are effectively continuous (usually fine for NN logits)
    # If you need exact ties handling, uncomment the block below.

    # Sum of ranks of positives per class
    pos_rank_sum = (ranks * t).sum(axis=0)
    auc_per_class = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(auc_per_class.mean())


def CMAP(y_true, y_pred, padding_factor=5):
    scored_classes = (np.sum(y_true,axis=0) > 0)
    
    y_true = np.pad(y_true, ((0, padding_factor), (0, 0)), constant_values=1)
    y_pred = np.pad(y_pred, ((0, padding_factor), (0, 0)), constant_values=1)
    return average_precision_score(
        (y_true[:,scored_classes]>0).astype(int),
        y_pred[:,scored_classes],
        average="macro",
    )


def MAP(y_true, y_pred):
    scored_classes = (np.sum(y_true,axis=0) > 0)
    return average_precision_score(
        (y_true[:,scored_classes]>0).astype(int),
        y_pred[:,scored_classes],
        average="macro",
    )