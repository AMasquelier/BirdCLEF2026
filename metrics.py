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