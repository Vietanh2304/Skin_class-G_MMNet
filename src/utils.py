import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score
)
from src.config import cfg 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# --- 1. LOSS FUNCTIONS (Clean & Focal with Label Smoothing support) ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Tích hợp Label Smoothing vào CrossEntropyLoss (PyTorch hỗ trợ sẵn)
        # Khi tính Focal, ta dùng CE loss đã smooth này làm cơ sở
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        # inputs: (Batch, NumClasses) - Logits
        # targets: (Batch) - Long/Int Labels (khi bỏ Mixup)
        
        # Tính Cross Entropy (có sẵn Label Smoothing nếu config có)
        ce_loss = self.ce_loss(inputs, targets)

        # Tính pt (xác suất dự đoán đúng lớp đó)
        pt = torch.exp(-ce_loss)
        
        # Công thức Focal Loss: (1 - pt)^gamma * CE
        focal_loss_val = (1.0 - pt).pow(self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss_val.mean()
        elif self.reduction == 'sum':
            return focal_loss_val.sum()
        else:
            return focal_loss_val

# --- 2. METRICS (Giữ nguyên, sửa Recall tính tổng quát) ---
def compute_metrics(labels, probs):
    # labels: (N,)
    # probs: (N, NumClasses)
    preds = np.argmax(probs, axis=1)
    
    metrics = {
        'Accuracy': accuracy_score(labels, preds),
        'BAcc': balanced_accuracy_score(labels, preds),
        'Kappa': cohen_kappa_score(labels, preds),
        'F1-Score': f1_score(labels, preds, average='weighted', zero_division=0),
        'Precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'Recall': recall_score(labels, preds, average='weighted', zero_division=0),
    }
    
    # Tính Recall riêng từng lớp để in log cho chi tiết
    # Trả về dạng list để bên train.py dễ tính trung bình
    try:
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_per_class
    except:
        metrics['recall_per_class'] = []
        
    return metrics