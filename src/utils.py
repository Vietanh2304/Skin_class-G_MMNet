import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, cohen_kappa_score
)
from src.config import cfg # Import cfg

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# --- 0. HELPER FUNCTIONS FOR MIXUP/CUTMIX (Giữ nguyên V26.1) ---
def mixup_data(x_img, x_meta, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = torch.tensor(np.random.beta(alpha, alpha), dtype=torch.float32, device=device)
    else:
        lam = torch.tensor(1.0, dtype=torch.float32, device=device)
    batch_size = x_img.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x_img = lam * x_img + (1.0 - lam) * x_img[index, :]
    mixed_x_meta = lam * x_meta + (1.0 - lam) * x_meta[index, :]
    y_a, y_b = y, y[index]
    return mixed_x_img, mixed_x_meta, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]; H = size[3]
    cut_rat = np.sqrt(1. - lam.item())
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W); bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W); bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x_img, x_meta, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam_beta = torch.tensor(np.random.beta(alpha, alpha), dtype=torch.float32, device=device)
    else:
        lam_beta = torch.tensor(1.0, dtype=torch.float32, device=device)
    batch_size = x_img.size()[0]
    index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x_img.size(), lam_beta)
    mixed_x_img = x_img.clone()
    mixed_x_img[:, :, bbx1:bbx2, bby1:bby2] = x_img[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_img.size()[-1] * x_img.size()[-2]))
    lam = torch.tensor(lam, dtype=torch.float32, device=device)
    mixed_x_meta = lam * x_meta + (1.0 - lam) * x_meta[index, :]
    return mixed_x_img, mixed_x_meta, y_a, y_b, lam

# --- 1. LOSS FUNCTIONS ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none') 

    def forward(self, inputs, targets):
        if targets.ndim == 2 and targets.dtype == torch.float32:
            log_pt = F.log_softmax(inputs, dim=1)
            ce_loss = -torch.sum(targets * log_pt, dim=1) # (B)
        else:
            ce_loss = self.ce_loss(inputs, targets) # (B)

        pt = torch.exp(-ce_loss)
        focal_loss_val = (1.0 - pt).pow(self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss_val.mean()
        else:
            return focal_loss_val.sum()

def mixup_criterion(criterion_base, pred, y_a, y_b, lam):
    y_a_onehot = F.one_hot(y_a, num_classes=cfg.NUM_CLASSES).float().to(pred.device)
    y_b_onehot = F.one_hot(y_b, num_classes=cfg.NUM_CLASSES).float().to(pred.device)
    mixed_targets = lam.unsqueeze(-1) * y_a_onehot + (1.0 - lam.unsqueeze(-1)) * y_b_onehot
    return criterion_base(pred, mixed_targets)


# --- 2. METRICS (Giữ nguyên) ---
def compute_metrics(labels, probs):
    preds = np.argmax(probs, axis=1)
    return {
        'Accuracy': accuracy_score(labels, preds), 'BAcc': balanced_accuracy_score(labels, preds),
        'Kappa': cohen_kappa_score(labels, preds), 'F1-Score': f1_score(labels, preds, average='weighted', zero_division=0),
        'Precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'Recall': recall_score(labels, preds, average='weighted', zero_division=0),
    }