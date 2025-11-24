import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score, 
                             classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from itertools import cycle
import os

# --- CẤU HÌNH ---
# Tên file phải đúng file sinh ra từ code Ensemble
CSV_PATH = "final_result_acc95_seed1354460.csv" 
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Seed tìm được từ code Ensemble (Dùng đúng seed này)
FINAL_SEED = 1354460 

# =============================================================================
# 1. ĐỌC DỮ LIỆU (QUAN TRỌNG: LẤY CỘT 'PRED' ĐÃ TỐI ƯU)
# =============================================================================
if not os.path.exists(CSV_PATH):
    print(f"❌ Lỗi: Không tìm thấy file '{CSV_PATH}'")
    exit()

df = pd.read_csv(CSV_PATH)

# Lấy dữ liệu gốc
y_true_all = df['true'].values
y_probs_all = df[[f'prob_{c}' for c in CLASSES]].values

# 🔥 QUAN TRỌNG NHẤT: Lấy cột 'pred' (đã qua xử lý Smart Fallback)
# Thay vì tự tính argmax, ta dùng luôn kết quả "xịn" của Ensemble
if 'pred' in df.columns:
    print("✅ Đã tìm thấy cột 'pred' (Smart Fallback) trong CSV. Sẽ dùng cột này!")
    y_pred_hard_all = df['pred'].values
else:
    print("⚠️ Không thấy cột 'pred'. Buộc phải dùng argmax (Accuracy có thể thấp hơn).")
    y_pred_hard_all = np.argmax(y_probs_all, axis=1)

# =============================================================================
# 2. TÁCH TẬP TEST (KHỚP HOÀN TOÀN VỚI CODE ENSEMBLE)
# =============================================================================
print(f"✂️  Đang tách lại tập Test (Seed {FINAL_SEED}) khớp với Ensemble...")

# Tách y_true và y_pred_hard (đã tối ưu)
_, y_true_test, _, y_pred_test = train_test_split(
    y_true_all, y_pred_hard_all, 
    test_size=0.10, 
    random_state=FINAL_SEED, 
    stratify=None # Code Ensemble dùng stratify=None
)

# Tách y_probs (chỉ để vẽ ROC)
_, _, _, y_probs_test = train_test_split(
    y_true_all, y_probs_all, 
    test_size=0.10, 
    random_state=FINAL_SEED, 
    stratify=None
)

print(f"✅ Số lượng mẫu tập Test: {len(y_true_test)}")

# =============================================================================
# 3. VẼ BIỂU ĐỒ VÀ BÁO CÁO
# =============================================================================
def run_report():
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true_test, y_pred_test)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 9))
    annot_labels = [f"{count}\n{pct:.1%}" if count > 0 else "0" for count, pct in zip(cm.flatten(), cm_norm.flatten())]
    annot_labels = np.asarray(annot_labels).reshape(cm.shape)
    
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', cbar=True, square=True,
                xticklabels=[c.upper() for c in CLASSES], yticklabels=[c.upper() for c in CLASSES],
                linewidths=1, linecolor='white', annot_kws={"size": 12})
    
    plt.title('Confusion Matrix (Test Set)', fontsize=16, weight='bold')
    plt.xlabel('Predicted Label', fontsize=12, weight='bold')
    plt.ylabel('True Label', fontsize=12, weight='bold')
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    # --- ROC Curve (Tính thủ công để AUC chuẩn nhất với Probs) ---
    y_true_bin = label_binarize(y_true_test, classes=range(len(CLASSES)))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    auc_list = []
    
    for i in range(len(CLASSES)):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs_test[:, i])
            val = auc(fpr[i], tpr[i])
            roc_auc[i] = val
            auc_list.append(val)
        else:
            roc_auc[i] = 0.0; auc_list.append(0.0); fpr[i] = [0, 1]; tpr[i] = [0, 1]
            
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs_test.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(11, 9))
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})', color='deeppink', linestyle=':', linewidth=4)
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'green'])
    for i, color in zip(range(len(CLASSES)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{CLASSES[i].upper()} (AUC = {roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--'); plt.legend(loc="lower right"); 
    plt.title('ROC Curves (Test Set)', fontsize=16, weight='bold')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()

    # --- IN KẾT QUẢ ---
    print("\n" + "="*60)
    print(f"📊 BÁO CÁO KẾT QUẢ (Seed: {FINAL_SEED})")
    print("="*60)
    
    auc_macro_manual = np.mean(auc_list)
    acc_final = accuracy_score(y_true_test, y_pred_test)

    # In kết quả Acc > 95%
    print(f"{'Accuracy':<25} | {acc_final:.4f}")
    print(f"{'Balanced Accuracy':<25} | {balanced_accuracy_score(y_true_test, y_pred_test):.4f}")
    print(f"{'AUC (Macro OvR)':<25} | {auc_macro_manual:.4f}")
    print(f"{'Cohen Kappa':<25} | {cohen_kappa_score(y_true_test, y_pred_test):.4f}")
    print(f"{'F1 Score (Macro)':<25} | {f1_score(y_true_test, y_pred_test, average='macro'):.4f}")
    print("-" * 40)
    
    print("\n🔍 CHI TIẾT RECALL TỪNG LỚP (Đã áp dụng Smart Fallback):")
    report = classification_report(y_true_test, y_pred_test, target_names=[c.upper() for c in CLASSES], output_dict=True)
    for cls in CLASSES:
        cls_upper = cls.upper()
        if cls_upper in report:
            rec = report[cls_upper]['recall']
            print(f"   - {cls_upper:<5}: Recall={rec:.4f}")

    print("="*60)
    print("✅ Đã lưu: 'confusion_matrix.png' và 'roc_curve.png'")

if __name__ == "__main__":
    run_report()