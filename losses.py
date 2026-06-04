from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, accuracy_score, f1_score, classification_report, confusion_matrix
import torch.nn as nn
import torch

import numpy as np
import pandas as pd




class ASL(nn.Module):
    def __init__(self, lam=(2,0.), margin=0.05, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.lam = lam
        self.margin = margin
    
    def forward(self, logits, y, weights=[], reduction=None):
        if reduction==None: reduction = self.reduction

        n_classes = logits.size(1)
        p = logits.sigmoid()
        p = torch.clamp(p, 1e-7, 1-1e-7)

        p_m = torch.maximum(torch.tensor(0.), p-self.margin)

        l_1 = -((1-p)**self.lam[1])*torch.log(p)
        l_0 = -(p_m**self.lam[0])*torch.log(1-p_m)
        asl = y * l_1 + (1-y) * l_0

        if len(weights)>0:
            asl = asl * weights

        if reduction=='mean': asl = asl.mean()
            
        return asl

class CASL(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        pos_gamma = {"Aves":.5, "Insecta":0, "Amphibia":0, "Mammalia":.5, "Reptilia":0}
        neg_gamma = {"Aves":.5, "Insecta":2, "Amphibia":.5, "Mammalia":.1, "Reptilia":.5}
        self.reduction = reduction

        tax = pd.read_csv('../../Datasets/BirdCLEF/2026/taxonomy.csv')
        pos_gamma = torch.tensor(tax.class_name.map(pos_gamma).values, dtype=torch.float32)
        neg_gamma = torch.tensor(tax.class_name.map(neg_gamma).values, dtype=torch.float32)

        self.register_buffer('pos_gamma', pos_gamma)
        self.register_buffer('neg_gamma', neg_gamma)
    
    def forward(self, logits, y, weights=[], reduction=None):
        if reduction==None: reduction = self.reduction
        pos_gamma = self.pos_gamma.unsqueeze(0).to(y.device)
        neg_gamma = self.neg_gamma.unsqueeze(0).to(y.device)

        n_classes = logits.size(1)
        p = logits.sigmoid()
        p = torch.clamp(p, 1e-7, 1-1e-7)

        l_1 = -((1-p)**pos_gamma)*torch.log(p)
        l_0 = -(p**neg_gamma)*torch.log(1-p)
        asl = y * l_1 + (1-y) * l_0

        if len(weights)>0:
            asl = asl * weights

        if reduction=='mean': asl = asl.mean()
            
        return asl



class BCE(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, y, weights=[], reduction=None):
        if reduction==None: reduction = self.reduction

        n_classes = logits.size(1)

        bce = self.bce(logits, y)
        if len(weights)>0:
            bce = bce * weights

        if reduction=='mean': bce = bce.mean()

        return bce

class CE(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.0)
    
    def forward(self, logits, y, weights=[], reduction=None):
        if reduction==None: reduction = self.reduction

        n_classes = logits.size(1)
        y = y/y.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        ce = -(y * log_probs).sum(dim=-1)

        if reduction=='mean': ce = ce.mean()
        else: ce = ce.unsqueeze(-1).repeat(1,234)

        return ce