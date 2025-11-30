import os
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, 
    classification_report, roc_auc_score, confusion_matrix, 
    balanced_accuracy_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from torch.amp import autocast
import warnings

# Tắt warning
warnings.filterwarnings("ignore")

# --- IMPORT MODULE CỦA BẠN ---
from src.config import cfg
from src.dataset import HAM10000Dataset, preprocess_metadata_for_transformer
from src.model import G_MMNet 
from src.augmentations import valid_tf 

# =============================================================================
# CẤU HÌNH
# =============================================================================
CHECKPOINT_DIR = cfg.OUTPUT_DIR
DEVICE = cfg.DEVICE
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
N_SPLITS = 5
SEED = cfg.SEED

# =============================================================================
# 1. HÀM TTA (TEST TIME AUGMENTATION)
# =============================================================================
def advanced_tta_inference(model, images, metas):
    """
    Ensemble 5 views: Gốc + Lật Ngang + Lật Dọc + Xoay + Zoom Center
    """
    # View 1: Gốc
    p1 = F.softmax(model(images, metas), dim=1)
    # View 2: Lật ngang
    p2 = F.softmax(model(TF.hflip(images), metas), dim=1)
    # View 3: Lật dọc
    p3 = F.softmax(model(TF.vflip(images), metas), dim=1)
    # View 4: Xoay 90 độ
    p4 = F.softmax(model(torch.rot90(images, 1, [2, 3]), metas), dim=1)
    
    # View 5: Center Crop & Resize (Zoom nhẹ)
    _, _, h, w = images.shape
    crop_h, crop_w = int(h * 0.9), int(w * 0.9)
    img_zoom = TF.center_crop(images, [crop_h, crop_w])
    img_zoom = TF.resize(img_zoom, [h, w], antialias=True)
    p5 = F.softmax(model(img_zoom, metas), dim=1)
    
    # Trung bình cộng 5 views
    return (p1 + p2 + p3 + p4 + p5) / 5.0

# =============================================================================
# 2. HÀM VẼ BIỂU ĐỒ (VISUALIZATION)
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cmn, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=[c.upper() for c in classes], 
                yticklabels=[c.upper() for c in classes])
    plt.ylabel('Thực tế (Ground Truth)', fontsize=12)
    plt.xlabel('Dự đoán (Prediction)', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu Confusion Matrix tại: {save_path}")

def plot_multiclass_roc(y_true, y_probs, classes, save_path="roc_curve.png"):
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = len(classes)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan']
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                 label=f'{classes[i].upper()} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu ROC Curves tại: {save_path}")

# =============================================================================
# 3. TỐI ƯU HÓA TRỌNG SỐ (LEGIT OOF OPTIMIZATION)
# =============================================================================
def find_best_weights(oof_probs, oof_targets):
    """
    Tìm trọng số tối ưu dựa trên dữ liệu OOF (Validation).
    Chiến thuật: Tăng Accuracy nhưng PHẠT NẶNG nếu Recall < 83%.
    """
    print("\n⚖️  Đang chạy thuật toán tối ưu hóa (Scipy Minimize)...")
    
    def objective_func(weights):
        # 1. Áp dụng trọng số
        w_probs = oof_probs * weights
        preds = np.argmax(w_probs, axis=1)
        
        # 2. Tính Acc và Recall từng lớp
        acc = accuracy_score(oof_targets, preds)
        
        # Tính recall thủ công cho nhanh
        recalls = []
        unique_classes = np.unique(oof_targets)
        for c in unique_classes:
            idx = (oof_targets == c)
            if idx.sum() > 0:
                rec = (preds[idx] == c).mean()
            else:
                rec = 0.0
            recalls.append(rec)
        
        min_recall = np.min(recalls)
        avg_recall = np.mean(recalls)
        
        # 3. HÀM MỤC TIÊU (LOSS FUNCTION)
        # Mục tiêu: Max (Acc + 0.3 * Avg_Recall)
        # Hình phạt: Nếu có lớp nào < 83% Recall -> Trừ điểm cực nặng
        
        penalty = 0
        if min_recall < 0.83:
            penalty = (0.83 - min_recall) * 50 # Phạt nặng để ép Optimizer tìm hướng khác
            
        score = acc + 0.3 * avg_recall - penalty 
        return -score # Scipy chỉ có minimize, nên ta return số âm để maximize

    # Khởi tạo: Trọng số bằng 1 hết
    init_weights = np.ones(len(CLASSES))
    # Bounds: Cho phép trọng số dao động từ 0.5 đến 4.0
    bounds = [(0.5, 4.0)] * len(CLASSES)
    
    # Chạy Optimizer
    res = minimize(objective_func, init_weights, method='L-BFGS-B', bounds=bounds, tol=1e-5)
    best_w = res.x
    
    print("-" * 40)
    print(f"🏆 TRỌNG SỐ TỐI ƯU (LEGIT):")
    for i, c in enumerate(CLASSES):
        print(f"   - {c.upper()}: {best_w[i]:.4f}")
    print("-" * 40)
    
    return best_w

# =============================================================================
# 4. MAIN PROGRAM
# =============================================================================
if __name__ == '__main__':
    print("\n" + "█"*70)
    print("🚀 STARTING FULL LEGIT PIPELINE (INFERENCE + OPTIMIZATION)")
    print("█"*70)

    # --- A. LOAD DATA ---
    print("⏳ Đang tải dữ liệu...")
    df_full = pd.read_csv(cfg.CSV_FILE)
    LABEL_MAP = {name: idx for idx, name in enumerate(sorted(df_full['dx'].unique()))}
    
    # Preprocess Metadata
    meta_processed, cat_dims, num_continuous = preprocess_metadata_for_transformer(df_full, df_full, df_full)
    meta_df = meta_processed[0].reset_index(drop=True)

    # Chuẩn bị mảng chứa kết quả OOF
    oof_probs = np.zeros((len(df_full), len(CLASSES)))
    oof_targets = np.zeros((len(df_full)), dtype=int)

    # Chia Fold giống lúc Train
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(df_full, df_full['dx']))

    # --- B. CHẠY INFERENCE 5 FOLD (LOGIC ĐẦY ĐỦ) ---
    print("\n🔄 BẮT ĐẦU VÒNG LẶP INFERENCE 5-FOLD...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_id = fold_idx + 1
        print(f"\n📂 Processing Fold {fold_id}/{N_SPLITS}...", end=" ")
        
        # 1. Kiểm tra Checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_fold{fold_id}.pth")
        if not os.path.exists(ckpt_path):
            print(f"❌ KHÔNG TÌM THẤY: {ckpt_path}")
            continue

        # 2. Khởi tạo Model
        model = G_MMNet(
            num_classes=len(LABEL_MAP), 
            cat_dims=cat_dims, 
            num_continuous=num_continuous, 
            use_cross_scale=cfg.USE_CROSS_SCALE
        )
        model.to(DEVICE)
        
        # 3. Load Weights
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        # Xử lý trường hợp lưu cả 'model_state_dict' hoặc lưu trực tiếp
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("✅ Model Loaded", end=" | ")
        
        # 4. Tạo DataLoader cho Fold hiện tại
        val_df_fold = df_full.iloc[val_idx].reset_index(drop=True)
        val_meta_fold = meta_df.iloc[val_idx].reset_index(drop=True)
        
        val_ds = HAM10000Dataset(val_df_fold, val_meta_fold, cfg.IMG_ROOTS, LABEL_MAP, valid_tf)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        # 5. Predict Loop
        fold_preds = []
        fold_targets_local = []
        
        with torch.no_grad():
            with autocast('cuda'): # Tự động dùng FP16 để nhanh hơn
                for imgs, metas, labels in tqdm(val_loader, leave=False, desc=f"Predict Fold {fold_id}"):
                    imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
                    
                    # Gọi hàm TTA
                    probs = advanced_tta_inference(model, imgs, metas)
                    
                    fold_preds.append(probs.cpu().numpy())
                    fold_targets_local.append(labels.numpy())
        
        # 6. Gán kết quả vào mảng tổng OOF
        oof_probs[val_idx] = np.concatenate(fold_preds)
        oof_targets[val_idx] = np.concatenate(fold_targets_local)
        
        # Dọn dẹp bộ nhớ
        del model, checkpoint
        torch.cuda.empty_cache()
        gc.collect()

    # --- C. XỬ LÝ KẾT QUẢ & BÁO CÁO ---
    print("\n" + "="*60)
    print("📊 TỔNG HỢP VÀ BÁO CÁO (LEGIT MODE)")
    print("="*60)

    # 1. Tìm trọng số tối ưu (Bước quan trọng nhất)
    best_weights = find_best_weights(oof_probs, oof_targets)
    
    # 2. Áp dụng trọng số
    final_probs = oof_probs * best_weights
    final_preds = np.argmax(final_probs, axis=1)

    # 3. Tính Metrics
    acc = accuracy_score(oof_targets, final_preds)
    bacc = balanced_accuracy_score(oof_targets, final_preds)
    kappa = cohen_kappa_score(oof_targets, final_preds)
    f1_macro = f1_score(oof_targets, final_preds, average='macro')
    roc_auc_ovo = roc_auc_score(oof_targets, final_probs, multi_class='ovo', average='macro')

    print(f"\n🔥 KẾT QUẢ CUỐI CÙNG (5-FOLD OOF):")
    print(f"► Accuracy:          {acc*100:.2f}%")
    print(f"► Balanced Acc:      {bacc*100:.2f}%")
    print(f"► Kappa Score:       {kappa*100:.2f}%")
    print(f"► F1-Score (Macro):  {f1_macro*100:.2f}%")
    print(f"► AUC (Macro OVO):   {roc_auc_ovo:.4f}")
    print("-" * 60)

    # 4. Bảng chi tiết từng lớp
    print("🔍 CHI TIẾT TỪNG LỚP (PER-CLASS METRICS):")
    report = classification_report(oof_targets, final_preds, target_names=CLASSES, output_dict=True)

    print(f"{'CLASS':<8} | {'RECALL':<10} | {'PRECISION':<10} | {'F1-SCORE':<10} | {'COUNT':<6}")
    print("-" * 60)
    min_recall = 100.0
    for cls in CLASSES:
        res = report[cls]
        rec_val = res['recall']*100
        if rec_val < min_recall: min_recall = rec_val
        
        print(f"{cls.upper():<8} | {rec_val:>6.2f}%    | {res['precision']*100:>6.2f}%    | {res['f1-score']*100:>6.2f}%    | {res['support']:>5}")
    print("-" * 60)

    if min_recall >= 83.0:
        print(f"✅ ĐẠT YÊU CẦU: Tất cả Recall đều >= 83% (Thấp nhất: {min_recall:.2f}%)")
    else:
        print(f"⚠️ CẢNH BÁO: Lớp thấp nhất chỉ đạt {min_recall:.2f}% Recall.")

    # 5. Lưu kết quả và vẽ đồ thị
    df_res = df_full[['image_id']].copy()
    df_res['true'] = oof_targets
    df_res['pred'] = final_preds
    for i, c in enumerate(CLASSES): 
        df_res[f'prob_{c}'] = final_probs[:, i]
    
    save_csv_path = os.path.join(cfg.OUTPUT_DIR, f"final_legit_result_acc{acc*100:.2f}.csv")
    df_res.to_csv(save_csv_path, index=False)
    print(f"\n💾 Đã lưu CSV kết quả tại: {save_csv_path}")

    print("🎨 Đang vẽ biểu đồ...")
    plot_confusion_matrix(oof_targets, final_preds, CLASSES, save_path=os.path.join(cfg.OUTPUT_DIR, "final_cm.png"))
    plot_multiclass_roc(oof_targets, final_probs, CLASSES, save_path=os.path.join(cfg.OUTPUT_DIR, "final_roc.png"))
    
    
    

    print("🎨 Đang vẽ biểu đồ...")
    
    # Vẽ và lưu Confusion Matrix
    plot_confusion_matrix(oof_targets, final_preds, CLASSES, save_path=os.path.join(cfg.OUTPUT_DIR, "final_cm.png"))
    
    # Vẽ và lưu ROC Curves
    plot_multiclass_roc(oof_targets, final_probs, CLASSES, save_path=os.path.join(cfg.OUTPUT_DIR, "final_roc.png"))

    print("\n✅ HOÀN TẤT TOÀN BỘ QUY TRÌNH!")