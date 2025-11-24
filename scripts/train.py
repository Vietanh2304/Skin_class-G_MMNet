import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import StratifiedKFold

# Import từ các module .py trong 'src'
from src.config import cfg
from src.utils import seed_everything, FocalLoss
from src.dataset import preprocess_metadata_for_transformer, HAM10000Dataset
from src.augmentations import train_tf, valid_tf
from src.model import G_MMNet
from src.engine import train_one_epoch, valid_one_epoch

# --- 0. SEEDING ---
# Lấy seed từ config và chạy
seed_everything(cfg.SEED)

def main():
    print("="*70)
    print(f"MAIN TRAINING - V27 (Focal Loss + Mixup, {cfg.SCHEDULER_TYPE} LR)")
    print(f"🔥 Dataset: HAM10000 | Folds: {cfg.FOLDS_TO_RUN}")
    print(f"🔥 CHẠY TRÊN THIẾT BỊ: {cfg.DEVICE}")
    print("="*70)

    # 1. DATA SETUP (HAM10000)
    df_full = pd.read_csv(cfg.CSV_FILE)
    nunique_labels = sorted(df_full['dx'].unique()) 
    LABEL_MAP = {name: idx for idx, name in enumerate(nunique_labels)}
    
    # Cập nhật lại NUM_CLASSES trong cfg nếu cần
    if cfg.NUM_CLASSES != len(LABEL_MAP):
        print(f"Cập nhật NUM_CLASSES: {cfg.NUM_CLASSES} -> {len(LABEL_MAP)}")
        cfg.NUM_CLASSES = len(LABEL_MAP)

    print(f"✅ Đã tải {len(df_full)} mẫu. Các lớp: {LABEL_MAP}\n")

    labels = df_full['dx'] 
    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)
    splits = list(skf.split(df_full, labels))

    print(f"✅ Đã tạo {len(splits)} folds (StratifiedKFold)")

    # --- 2. Bắt đầu lặp qua các Folds --
    all_fold_metrics = []

    for fold_id in cfg.FOLDS_TO_RUN:
        train_idx, val_idx = splits[fold_id]
        
        print(f"\n{'='*80}")
        print(f"🔥 BẮT ĐẦU FOLD {fold_id + 1}/{cfg.N_SPLITS}")
        print(f"{'='*80}")
        
        train_df = df_full.iloc[train_idx].copy()
        val_df = df_full.iloc[val_idx].copy()
        
        print(f"  Train: {len(train_df)}")
        print(f"  Val:   {len(val_df)}")

        # --- 3. Preprocessing --
        (train_meta, val_meta, _), cat_dims, num_continuous = \
            preprocess_metadata_for_transformer(train_df, val_df, val_df.copy()) 
        print(f"  ✅ Meta-features: Num={num_continuous}, Cat={len(cat_dims)} {cat_dims}")

        # --- 4. Datasets & Dataloaders --
        train_ds = HAM10000Dataset(train_df.reset_index(drop=True), train_meta.reset_index(drop=True), cfg.IMG_ROOTS, LABEL_MAP, train_tf)
        val_ds = HAM10000Dataset(val_df.reset_index(drop=True), val_meta.reset_index(drop=True), cfg.IMG_ROOTS, LABEL_MAP, valid_tf)

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
        print("  ✅ Dataloaders created.")

        # --- 5. Loss (🔥 THAY ĐỔI: DÙNG FOCAL LOSS) --
        if cfg.USE_FOCAL_LOSS:
            criterion = FocalLoss(gamma=cfg.FOCAL_LOSS_GAMMA).to(cfg.DEVICE)
            print(f"  ✅ Đã dùng FocalLoss (gamma={cfg.FOCAL_LOSS_GAMMA}).")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING).to(cfg.DEVICE)
            print(f"  ✅ Đã dùng CrossEntropyLoss (LS={cfg.LABEL_SMOOTHING}).")
            
        # --- 6. Model, Optimizer, Scheduler --
        model = G_MMNet(num_classes=cfg.NUM_CLASSES, cat_dims=cat_dims, num_continuous=num_continuous, use_cross_scale=cfg.USE_CROSS_SCALE).to(cfg.DEVICE)
        
        # Optimizer 4-phần (Giữ nguyên)
        backbone_decay = []; backbone_no_decay = []; head_decay = []; head_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if name.startswith("stem.backbone."):
                if param.ndim <= 1 or name.endswith(".bias"): backbone_no_decay.append(param)
                else: backbone_decay.append(param)
            else:
                if param.ndim <= 1 or name.endswith(".bias"): head_no_decay.append(param)
                else: head_decay.append(param)
        optimizer_grouped_parameters = [
            {'params': backbone_decay, 'lr': cfg.BACKBONE_LR, 'weight_decay': cfg.WEIGHT_DECAY},      
            {'params': backbone_no_decay, 'lr': cfg.BACKBONE_LR, 'weight_decay': 0.0},   
            {'params': head_decay, 'lr': cfg.HEAD_LR, 'weight_decay': cfg.WEIGHT_DECAY},              
            {'params': head_no_decay, 'lr': cfg.HEAD_LR, 'weight_decay': 0.0}             
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.HEAD_LR, eps=cfg.EPS, betas=cfg.BETAS)
        print(f"✅ Optimizer: AdamW (4-Part Smart Weight Decay)")
        
        # Scheduler (Giữ nguyên V26)
        steps_per_epoch = len(train_loader)
        
        if cfg.SCHEDULER_TYPE == 'cosine':
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.01, 
                end_factor=1.0, 
                total_iters=cfg.WARMUP_EPOCHS * steps_per_epoch
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=(cfg.EPOCHS - cfg.WARMUP_EPOCHS) * steps_per_epoch, 
                eta_min=1e-7 # LR nhỏ nhất
            )
            scheduler = SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, cosine_scheduler], 
                milestones=[cfg.WARMUP_EPOCHS * steps_per_epoch]
            )
            print(f"✅ Scheduler: CosineAnnealingLR (Max LR: {cfg.HEAD_LR})\n")
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[cfg.BACKBONE_LR, cfg.BACKBONE_LR, cfg.HEAD_LR, cfg.HEAD_LR], 
                epochs=cfg.EPOCHS,
                steps_per_epoch=steps_per_epoch,
                pct_start=cfg.WARMUP_EPOCHS / cfg.EPOCHS
            )
            print(f"✅ Scheduler: OneCycleLR (Max LR: {cfg.HEAD_LR})\n")

        # Reset TTA log (vì valid_one_epoch nằm trong module khác)
        if hasattr(valid_one_epoch, 'logged_tta'): del valid_one_epoch.logged_tta
        if hasattr(valid_one_epoch, 'logged_no_tta'): del valid_one_epoch.logged_no_tta
            
        # --- 7. Training Loop (cho fold này) ---
        best_kappa = 0 
        best_acc = 0
        best_f1 = 0
        best_epoch = 0
        patience_counter = 0
        total_start_time = time.time()

        for epoch in range(1, cfg.EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE, epoch, scheduler)
            val_loss, metrics = valid_one_epoch(model, val_loader, criterion, cfg.DEVICE)
            
            lr_now = optimizer.param_groups[2]['lr'] 
            
            if np.isnan(train_loss) or np.isnan(val_loss):
                print(f"❌ ERROR: Loss is NaN at Epoch {epoch}. Dừng fold.")
                break

            print(f"Ep {epoch:3d} | T_Loss: {train_loss:.4f} | V_Loss: {val_loss:.4f} | LR: {lr_now:.1e} | "
                  f"Kappa: {metrics['Kappa']*100:.2f}% | F1-Score: {metrics['F1-Score']*100:.2f}% | "
                  f"Precision: {metrics['Precision']*100:.2f}% | Recall: {metrics['Recall']*100:.2f}% | "
                  f"Accuracy: {metrics['Accuracy']*100:.2f}%")
            
            # Save Best
            if metrics['Kappa'] > best_kappa:
                best_kappa = metrics['Kappa']
                best_acc = metrics['Accuracy']
                best_f1 = metrics['F1-Score']
                best_epoch = epoch
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_kappa': best_kappa
                }, os.path.join(cfg.OUTPUT_DIR, f"best_fold{fold_id+1}.pth"))
                
                print(f"  🏆 NEW BEST (Fold {fold_id+1})! K: {best_kappa*100:.2f}% | F1: {metrics['F1-Score']*100:.2f}% | A: {metrics['Accuracy']*100:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.PATIENCE:
                    print(f"\n  🛑 Early stop at epoch {epoch}")
                    break

        # --- 8. Kết thúc Fold ---
        total_time = (time.time() - total_start_time) / 60
        
        # 🔥 ĐÃ LÀM RÕ PHẦN TÓM TẮT CUỐI FOLD
        print(f"\n{'='*80}")
        print(f"🎉 KẾT THÚC FOLD {fold_id+1} / {cfg.N_SPLITS} (Sau {total_time:.2f} phút)")
        print(f"{'-'*80}")
        print(f"  Kết quả tốt nhất (Best Model Metrics) của Fold {fold_id+1}:")
        print(f"  > Epoch đạt Tốt Nhất: {best_epoch}")
        print(f"  > Best Kappa:    {best_kappa*100:.2f}%")
        print(f"  > Best F1-Score: {best_f1*100:.2f}%")
        print(f"  > Best Accuracy: {best_acc*100:.2f}%")
        print(f"  > (Model đã được lưu tại: {os.path.join(cfg.OUTPUT_DIR, f'best_fold{fold_id+1}.pth')})")
        print(f"{'='*80}\n") # Thêm một dòng trống để tách biệt

        all_fold_metrics.append({
            'fold': fold_id + 1,
            'kappa': best_kappa,
            'f1': best_f1,
            'acc': best_acc
        })
        
        del model, train_loader, val_loader, optimizer, scheduler, criterion
        gc.collect()
        torch.cuda.empty_cache()

    # --- 9. In kết quả cuối cùng ---
    print(f"\n{'='*80}")
    print(f"🏁 HOÀN TẤT {len(cfg.FOLDS_TO_RUN)} FOLD VỪA CHẠY (Fold ID: {cfg.FOLDS_TO_RUN})")
    print(f"Sử dụng Focal Loss: {cfg.USE_FOCAL_LOSS}")
    print(f"{'='*80}")

    metrics_df = pd.DataFrame(all_fold_metrics)
    
    # In kết quả chi tiết của từng fold (bỏ cột recall_per_class cho gọn)
    print("--- Kết quả chi tiết (Các Fold vừa chạy) ---")
    columns_to_print = ['fold', 'kappa', 'f1', 'acc']
    print(metrics_df[columns_to_print].to_string())
    print(f"{'-'*80}")

    # --- TÓM TẮT KẾT QUẢ TRUNG BÌNH (Mean ± Std) ---
    if len(metrics_df) > 1:
        mean_kappa = metrics_df['kappa'].mean() * 100
        std_kappa = metrics_df['kappa'].std() * 100
        
        mean_f1 = metrics_df['f1'].mean() * 100
        std_f1 = metrics_df['f1'].std() * 100
        
        mean_acc = metrics_df['acc'].mean() * 100
        std_acc = metrics_df['acc'].std() * 100
        
        print("--- Tóm tắt Trung bình Tổng thể (Mean ± Std) ---")
        print(f"  > Average Kappa:    {mean_kappa:.2f}% ± {std_kappa:.2f}%")
        print(f"  > Average F1-Score: {mean_f1:.2f}% ± {std_f1:.2f}%")
        print(f"  > Average Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")

        # --- 🔥 BỔ SUNG: TÍNH TRUNG BÌNH RECALL TỪNG LỚP ---
        print("\n  --- Trung bình Recall các lớp (Mean ± Std) ---")
        try:
            # Chuyển list các array [[...], [...]] thành 1 matrix
            all_recalls = np.stack(metrics_df['recall_per_class']) 
            mean_recalls = np.mean(all_recalls, axis=0) * 100
            std_recalls = np.std(all_recalls, axis=0) * 100
            
            # Đảo LABEL_MAP (đã được định nghĩa ở đầu file train.py)
            inv_label_map = {v: k for k, v in LABEL_MAP.items()} 
            
            for i in range(len(mean_recalls)):
                class_name = inv_label_map.get(i, f"Class {i}").upper()
                # Dùng ljust để căn chỉnh cho đẹp
                print(f"    > {class_name.ljust(6)}: {mean_recalls[i]:.2f}% ± {std_recalls[i]:.2f}%")
        except Exception as e:
            print(f"    (Lỗi khi tính trung bình recall: {e})")
        # ----------------------------------------------------
        
        print(f"{'='*80}")
    
    # --- TÓM TẮT KẾT QUẢ (Nếu chỉ chạy 1 fold) ---
    elif len(metrics_df) == 1:
        print("--- Tóm tắt Kết quả (Chỉ chạy 1 fold) ---")
        print(f"  > Kappa:    {metrics_df['kappa'].iloc[0]*100:.2f}%")
        print(f"  > F1-Score: {metrics_df['f1'].iloc[0]*100:.2f}%")
        print(f"  > Accuracy: {metrics_df['acc'].iloc[0]*100:.2f}%")
        
        # In recall chi tiết cho 1 fold đó
        print("\n  --- Recall các lớp (Fold 1) ---")
        try:
            all_recalls = metrics_df['recall_per_class'].iloc[0] * 100
            inv_label_map = {v: k for k, v in LABEL_MAP.items()}
            for i in range(len(all_recalls)):
                class_name = inv_label_map.get(i, f"Class {i}").upper()
                print(f"    > {class_name.ljust(6)}: {all_recalls[i]:.2f}%")
        except Exception as e:
            print(f"    (Lỗi khi in recall: {e})")
            
        print(f"{'='*80}")

    print("\n(Đã hoàn tất.)")

# Bảo vệ code để đảm bảo nó chỉ chạy khi được gọi trực tiếp
if __name__ == "__main__":
    main()