# ===============================================================
# CELL 8: METRICS & TRAINING FUNCTIONS (V-JEPA Enhanced)
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

# --- 0. HELPER: SEMANTIC BLOCK MASKING (V-JEPA Style) ---
def generate_block_mask(img_size, mask_ratio=0.4):
        """
        Tạo mask dạng khối lớn (Block Masking) - V-JEPA Style.
        """
        H, W = img_size
        mask = torch.ones((H, W), dtype=torch.float32)
        
        # Số lượng khối ít nhưng to hơn
        num_masking_patches = int(mask_ratio * 5) 
        
        for _ in range(num_masking_patches):
            # Kích thước khối che ngẫu nhiên (lớn)
            block_h = np.random.randint(H // 5, H // 2.5)
            block_w = np.random.randint(W // 5, W // 2.5)
            
            top = np.random.randint(0, H - block_h)
            left = np.random.randint(0, W - block_w)
            
            mask[top:top+block_h, left:left+block_w] = 0.0
            
        return mask
def apply_masking(imgs, mask_ratio=0.4):
        B, C, H, W = imgs.shape
        masked_imgs = imgs.clone()
        
        for i in range(B):
            mask = generate_block_mask((H, W), mask_ratio=mask_ratio)
            mask = mask.to(imgs.device)
            mask = mask.unsqueeze(0).expand_as(imgs[i])
            masked_imgs[i] = masked_imgs[i] * mask
            
        return masked_imgs    

# --- 1. TRAIN FUNCTION (V-JEPA ENABLED) ---
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scheduler):
    model.train()
    total_loss = 0.0
    
    scaler = GradScaler(enabled=cfg.USE_AMP)
    
    # V-JEPA Config
    use_vjepa = True
    mask_ratio = 0.4 
    consist_weight = 2.0 # Tăng trọng số học ngữ cảnh
    
    # Dùng Cosine Similarity cho Feature (Vector) thay vì MSE
    feature_criterion = nn.CosineEmbeddingLoss()
    
    pbar = tqdm(loader, desc=f"Ep {epoch}", leave=False, dynamic_ncols=True, file=sys.stdout)
    
    for imgs, metas, labels in pbar:
        # Check NaN lần cuối để an toàn
        if torch.isnan(metas).any(): continue

        imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Cờ target cho Cosine Loss (1 nghĩa là muốn 2 vector giống nhau)
        target_flag = torch.ones(imgs.size(0)).to(device)
        
        with autocast(device_type='cuda', enabled=cfg.USE_AMP):
            # 1. Clean Pass (Giáo viên)
            # Lấy cả Logits và Features sạch
            logits_clean, feats_clean = model(imgs, metas, return_feats=True)
            loss_cls = criterion(logits_clean, labels)
            
            loss_final = loss_cls
            loss_consist_val = 0.0

            # 2. Masked Pass (Học sinh V-JEPA)
            if use_vjepa:
                # Che ảnh
                masked_imgs = apply_masking(imgs, mask_ratio=mask_ratio)
                
                # Chạy qua model
                _, feats_masked = model(masked_imgs, metas, return_feats=True)
                
                # Consistency Loss: Feature ảnh che phải giống Feature ảnh sạch
                # .detach() ở feats_clean là cực kỳ quan trọng (Stop Gradient)
                loss_consist = feature_criterion(feats_masked, feats_clean.detach(), target_flag)
                
                # Tổng hợp Loss
                loss_final = loss_cls + (consist_weight * loss_consist)
                loss_consist_val = loss_consist.item()

        # Backward
        scaler.scale(loss_final).backward()
        scaler.unscale_(optimizer) 
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler: scheduler.step()
        
        total_loss += loss_final.item() * imgs.size(0)
        
        pbar.set_postfix(cls=f"{loss_cls.item():.3f}", jepa=f"{loss_consist_val:.3f}")
    
    return total_loss / len(loader.dataset)

# --- 2. VALID FUNCTION ---
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
            
            with autocast(device_type='cuda', enabled=use_amp):
                # Valid chỉ cần Logits, không cần features
                logits_orig = model(imgs, metas, return_feats=False)
                probs_orig = F.softmax(logits_orig, dim=1)
                
                if use_tta_local:
                    imgs_flipped = torch.flip(imgs, dims=[3]) 
                    logits_flipped = model(imgs_flipped, metas, return_feats=False)
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

print("✅ Training Functions V36 (V-JEPA Feature Consistency) READY")