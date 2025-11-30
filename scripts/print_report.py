import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. NHẬP SỐ LIỆU ĐÃ CHỐT (TỪ ẢNH BẠN GỬI)
# =============================================================================
# Class: [Recall, Precision, Support]
stats = {
    'akiec': {'rec': 0.8349, 'prec': 0.8505, 'supp': 327},
    'bcc':   {'rec': 0.9339, 'prec': 0.9108, 'supp': 514},
    'bkl':   {'rec': 0.9445, 'prec': 0.9746, 'supp': 1099},
    'df':    {'rec': 0.8348, 'prec': 0.6713, 'supp': 115},
    'mel':   {'rec': 0.9371, 'prec': 0.9604, 'supp': 1113},
    'nv':    {'rec': 0.9857, 'prec': 0.9847, 'supp': 6705},
    'vasc':  {'rec': 0.8310, 'prec': 0.7329, 'supp': 142}
}

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
total_samples = sum([stats[c]['supp'] for c in CLASSES])

# =============================================================================
# 2. TÍNH TOÁN CÁC CHỈ SỐ
# =============================================================================
# Tính Weighted Averages
weighted_recall = sum([stats[c]['rec'] * stats[c]['supp'] for c in CLASSES]) / total_samples
weighted_precision = sum([stats[c]['prec'] * stats[c]['supp'] for c in CLASSES]) / total_samples

# Tính F1-Score từng class
f1_scores = {}
weighted_f1_sum = 0
for c in CLASSES:
    r = stats[c]['rec']
    p = stats[c]['prec']
    f1 = 2 * (p * r) / (p + r)
    f1_scores[c] = f1
    weighted_f1_sum += f1 * stats[c]['supp']

# Metrics tổng hợp
acc = 0.9643  # Lấy từ ảnh
bacc = sum([stats[c]['rec'] for c in CLASSES]) / len(CLASSES)
weighted_f1 = weighted_f1_sum / total_samples
macro_f1 = sum(f1_scores.values()) / len(CLASSES)
macro_recall = sum([stats[c]['rec'] for c in CLASSES]) / len(CLASSES)
macro_auc = 0.9714 # Lấy từ ảnh

# Tính Kappa (Ước lượng)
count_preds = {}
for c in CLASSES:
    count_preds[c] = (stats[c]['rec'] * stats[c]['supp']) / stats[c]['prec']
pe = 0
for c in CLASSES:
    prob_true = stats[c]['supp'] / total_samples
    prob_pred = count_preds[c] / total_samples
    pe += prob_true * prob_pred
kappa = (acc - pe) / (1 - pe)

# =============================================================================
# 3. TẠO NỘI DUNG BÁO CÁO (STRING)
# =============================================================================
lines = []
lines.append("="*60)
lines.append("🏆  FINAL CLINICAL EVALUATION REPORT (Q1 STANDARD)")
lines.append("="*60)
lines.append(f"1. Overall Accuracy      : {acc*100:.2f}%")
lines.append(f"2. Balanced Accuracy     : {bacc*100:.2f}%")
lines.append(f"3. Kappa Score           : {kappa*100:.2f}%")
lines.append(f"4. Macro F1-Score        : {macro_f1*100:.2f}%")
lines.append(f"5. Weighted F1-Score     : {weighted_f1*100:.2f}%")
lines.append(f"6. Macro Recall          : {macro_recall*100:.2f}%")
lines.append(f"7. Macro AUC (One-vs-Rest): {macro_auc:.4f}")
lines.append("-" * 60)
lines.append(f"{'CLASS':<8} {'RECALL':<10} {'PRECISION':<10} {'F1-SCORE':<10} {'SUPPORT':<8}")
lines.append("-" * 60)

for cls in CLASSES:
    rec = stats[cls]['rec']
    prec = stats[cls]['prec']
    f1 = f1_scores[cls]
    supp = stats[cls]['supp']
    lines.append(f"{cls.upper():<8} {rec*100:>8.2f}% {prec*100:>8.2f}% {f1*100:>8.2f}% {supp:>8}")
lines.append("="*60)

report_text = "\n".join(lines)

# In ra màn hình để kiểm tra
print(report_text)

# =============================================================================
# 4. VẼ VÀ LƯU ẢNH (FINAL_RESULT.PNG)
# =============================================================================
def text_to_image(text, filename):
    # Tạo một figure trống, kích thước vừa đủ
    plt.figure(figsize=(10, 8))
    
    # Xóa các trục (axes)
    plt.axis('off')
    
    # Vẽ chữ lên hình
    # family='monospace': Để các cột thẳng hàng nhau (như trong terminal)
    plt.text(0.05, 0.95, text, 
             fontsize=12, 
             family='monospace', 
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=1", facecolor="white", alpha=1)) # Nền trắng
    
    # Lưu ảnh
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n[IMAGE] ✅ Đã lưu ảnh báo cáo sắc nét: {filename}")

# Thực hiện lưu
text_to_image(report_text, "final_result.png")