# ===============================================================
# CELL 8: METRICS & TRAINING FUNCTIONS (V32 - Tối ưu Memory/AMP)
# ===============================================================
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from src.config import cfg
from src.utils import compute_metrics
from torch.amp import autocast, GradScaler
from torch.cuda.amp.grad_scaler import GradScaler # Ensure GradScaler is imported

# --- 0. HELPER: RANDOM MASKING (Giữ nguyên) ---
def random_mask_image(imgs, mask_ratio=0.25):
    # ... (Logic giữ nguyên)
    B, C, H, W = imgs.shape
    masked_imgs = imgs.clone()
    grid_size = 16
    h_step = H // grid_size
    w_step = W // grid_size
    num_patches = grid_size * grid_size
    num_masked = int(num_patches * mask_ratio)
    
    for i in range(B):
        mask_indices = torch.randperm(num_patches)[:num_masked]
        for idx in mask_indices:
            h_idx = idx // grid_size
            w_idx = idx % grid_size
            h_start = h_idx * h_step
            w_start = w_idx * w_step
            masked_imgs[i, :, h_start:h_start+h_step, w_start:w_start+w_step] = 0.0
            
    return masked_imgs

# --- 1. TRAIN FUNCTION (V32 - Fix Memory Leak và OOM) ---
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scheduler):
    model.train()
    total_loss = 0.0
    
    use_amp = getattr(cfg, 'USE_AMP', False)
    scaler = GradScaler(enabled=use_amp)
    
    use_masking = getattr(cfg, 'USE_MASKING_LOSS', False)
    mask_ratio = getattr(cfg, 'MASKING_RATIO', 0.15)
    mask_weight = getattr(cfg, 'MASKING_LOSS_WEIGHT', 1.0)
    consistency_criterion = nn.MSELoss()
    
    pbar = tqdm(loader, desc=f"Ep {epoch}", leave=False, dynamic_ncols=True, file=sys.stdout)
    
    for imgs, metas, labels in pbar:
        imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 🔥 Vòng lặp chính tính toán (autocast)
        with autocast(device_type='cuda', enabled=use_amp):
            # 1. Clean Pass (Phân loại)
            logits_clean = model(imgs, metas)
            loss_cls = criterion(logits_clean, labels)
            
            loss_final = loss_cls
            loss_consist_val = 0.0

            # 2. Masked Pass (Consistency)
            if use_masking:
                # Tạo ảnh bị che
                masked_imgs = random_mask_image(imgs, mask_ratio=mask_ratio)
                
                # Chạy ảnh bị che qua model
                logits_masked = model(masked_imgs, metas)
                
                # Consistency Loss: detach() ở đây là rất quan trọng để tránh graph phức tạp
                loss_consist = consistency_criterion(logits_masked, logits_clean.detach())
                
                # Tổng hợp Loss
                loss_final = loss_cls + (mask_weight * loss_consist)
                loss_consist_val = loss_consist.item()

        # 🔥 Backward & Step (Dùng Scaler)
        scaler.scale(loss_final).backward()
        scaler.unscale_(optimizer) 
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler: scheduler.step()
        
        total_loss += loss_final.item() * imgs.size(0)
        
        pbar.set_postfix(loss=f"{loss_final.item():.4f} Mask:{loss_consist_val:.3f}")
    
    return total_loss / len(loader.dataset)


# --- 2. VALID FUNCTION (Tối ưu AMP) ---
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = [] 
    
    val_criterion = criterion 

    use_tta_local = getattr(cfg, 'USE_TTA', True) 
    use_amp = getattr(cfg, 'USE_AMP', False)

    if use_tta_local:
        if not hasattr(valid_one_epoch, 'logged_tta'): 
            print("   (Validating with x2 TTA: Original + Horizontal Flip...)")
            valid_one_epoch.logged_tta = True 
    else:
        if not hasattr(valid_one_epoch, 'logged_no_tta'): 
            print("   (Validating NO TTA...)")
            valid_one_epoch.logged_no_tta = True
            
    with torch.no_grad():
        for imgs, metas, labels in tqdm(loader, desc="Valid", leave=False, dynamic_ncols=True, file=sys.stdout):
            imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
            
            # Sửa thành:
            with autocast(device_type='cuda', enabled=use_amp):
                logits_orig = model(imgs, metas)
                probs_orig = F.softmax(logits_orig, dim=1)
                
                if use_tta_local:
                    imgs_flipped = torch.flip(imgs, dims=[3]) 
                    logits_flipped = model(imgs_flipped, metas)
                    probs_flipped = F.softmax(logits_flipped, dim=1)
                    probs_avg = (probs_orig + probs_flipped) / 2.0
                else:
                    probs_avg = probs_orig
                
                loss = val_criterion(logits_orig, labels)
            
            total_loss += loss.item() * imgs.size(0)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs_avg.cpu().numpy())
    
    print() 
    metrics = compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs))
    return total_loss / len(loader.dataset), metrics

print("✅ Training Functions V32 (Masking Consistency Loss & Memory Fix) READY")