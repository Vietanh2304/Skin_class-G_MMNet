# import os
# import torch
# import random
# import numpy as np

# class Config:
#     # ============ DATA & PATHS (HAM10000) ===========
#     CSV_FILE = "/home/ibmelab/Documents/skin/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"
#     IMG_ROOTS = [
#         "/home/ibmelab/Documents/skin/skin-cancer-mnist-ham10000/HAM10000_images_part_1", 
#         "/home/ibmelab/Documents/skin/skin-cancer-mnist-ham10000/HAM10000_images_part_2",
#     ]
#     OUTPUT_DIR = "/home/ibmelab/Documents/G_MMNet/checkpoints"
    
#     # 🔥 KHÔNG DÙNG RESUME - Train từ đầu
#     RESUME_CHECKPOINT = None 
    
#     # 🔥 THAY ĐỔI: CHỈ ĐỊNH GPU 1 CỦA MÁY TRẠM
#     DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
#     SEED = 42
#     N_SPLITS = 5
    
#     # ============ MODEL ARCHITECTURE (V24 Asymmetric) ===========
#     NUM_CLASSES = 7
#     IMG_SIZE = 224
#     META_DIM = 256
#     IMG_EMBED_DIM_FINE = 192  
#     IMG_EMBED_DIM_COARSE = 768 
#     FUSION_MLP_DIM_FINE = 768    
#     FUSION_MLP_DIM_COARSE = 3072
#     FUSION_NUM_LAYERS = 4
#     FUSION_NUM_HEADS = 8    
#     USE_CROSS_SCALE = True 
    
#     # ============ REGULARIZATION (Giữ nguyên) ===========
#     META_DROPOUT = 0.2
#     FUSION_DROPOUT = 0.2
#     META_FEATURE_DROPOUT_RATE = 0.1
#     MODALITY_DROPOUT_RATE = 0.15
    
#     # ============ TRAINING (Giữ nguyên) ===========
#     BATCH_SIZE = 12
#     EPOCHS = 200 # Chạy tối đa 200 (hoặc hết 30 tiếng)
#     PATIENCE = 200 # Tắt Early Stop
#     FOLDS_TO_RUN = [0,1,2,3,4]
    
#     # ============ LOSS & STRATEGY (🔥 V27 - DÙNG FOCAL LOSS) ===========
#     USE_HYBRID_LOSS = False    
#     LABEL_SMOOTHING = 0.0 # ❌ TẮT Label Smoothing (Focal Loss không cần)
#     USE_FOCAL_LOSS = True # ✅ BẬT FOCAL LOSS
#     FOCAL_LOSS_GAMMA = 2.0 # Giá trị tiêu chuẩn
    
#     # ============ OPTIMIZER (Giữ nguyên) ===========
#     WEIGHT_DECAY = 0.05       
#     BETAS = (0.9, 0.999)    
#     EPS = 1e-6
    
#     # ============ LEARNING RATE (V26 - Ổn định) ===========
#     HEAD_LR = 1e-4            # (0.0002) - Mức ổn định
#     BACKBONE_LR = 1e-5        
    
#     # ============ SCHEDULER (V26 - Ổn định) ===========
#     SCHEDULER_TYPE = 'cosine' 
#     WARMUP_EPOCHS = 15        
    
#     # ============ AUGMENTATION (V26 - Bật Mixup) ===========
#     USE_TTA = True
#     USE_MIXUP = False
#     MIXUP_PROB = 0.5 
#     MIXUP_ALPHA = 0.4
    
#     # ============ OTHERS (Giữ nguyên) ===========
#     USE_AMP = False         
#     GRAD_CLIP = 0.5  
#     STOCHASTIC_DEPTH_RATE = 0.1       
#     USE_MASKING_LOSS = True   # ✅ Bật tính năng Masking Consistency
#     MASKING_RATIO = 0.15      # Giảm xuống 15% (thay vì 25% đã thử)
#     MASKING_LOSS_WEIGHT = 1.0
# # Khởi tạo một instance của Config để các file khác import
# cfg = Config()

# # Tạo thư mục output
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# # In thông báo xác nhận
# print(f"\n✅ Config V27 (Focal Loss + Mixup - Train từ đầu) Ready:")
# print(f"   🔥 ĐÃ CHỈ ĐỊNH CHẠY TRÊN: {cfg.DEVICE}")
# print(f"   Model: EffNetB1-Stem + Asymmetric Mamba (192D/768D)")
# print(f"   🔥 Chiến lược Kappa: BẬT Focal Loss (gamma={cfg.FOCAL_LOSS_GAMMA})")
# print(f"   🔥 Chiến lược Kappa: BẬT Mixup/Cutmix (p={cfg.MIXUP_PROB})")
# print(f"   🔥 LR: Head={cfg.HEAD_LR}, Backbone={cfg.BACKBONE_LR} (Ổn định)")
# ===============================================================
# CELL 4: CONFIG - V36 (Single-Stream Bidirectional Mamba)
# ===============================================================
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Bật nếu cần
import torch
import random
import numpy as np

class Config:
    # ============ DATA & PATHS (HAM10000) ===========
    CSV_FILE = "/home/ibmelab/Documents/skin/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"
    IMG_ROOTS = [
        "/home/ibmelab/Documents/skin/skin-cancer-mnist-ham10000/HAM10000_images_part_1", 
        "/home/ibmelab/Documents/skin/skin-cancer-mnist-ham10000/HAM10000_images_part_2",
    ]
    OUTPUT_DIR = "/home/ibmelab/Documents/G_MMNet/checkpoints"
    
    # 🔥 KHÔNG DÙNG RESUME - Train từ đầu
    RESUME_CHECKPOINT = None 
    
    # 🔥 CHỈ ĐỊNH GPU 1
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    SEED = 42
    N_SPLITS = 5
    
    # ============ MODEL ARCHITECTURE (V36 - Single Stream Lite) ===========
    NUM_CLASSES = 7
    IMG_SIZE = 224            # Giữ 224x224 cho an toàn và nhanh
    META_DIM = 256
    
    # Các tham số này không còn dùng cho V36 (vì V36 tự định nghĩa D_MODEL=512)
    # Nhưng cứ để đây để tránh lỗi nếu có code cũ gọi tới
    IMG_EMBED_DIM_FINE = 192  
    IMG_EMBED_DIM_COARSE = 768 
    FUSION_MLP_DIM_FINE = 768    
    FUSION_MLP_DIM_COARSE = 3072
    FUSION_NUM_LAYERS = 4
    FUSION_NUM_HEADS = 8    
    USE_CROSS_SCALE = False   # V36 là Single Stream, không cần Cross Scale
    
    # ============ REGULARIZATION (V36 - Ổn định) ===========
    META_DROPOUT = 0.2
    FUSION_DROPOUT = 0.2
    META_FEATURE_DROPOUT_RATE = 0.1
    MODALITY_DROPOUT_RATE = 0.15
    STOCHASTIC_DEPTH_RATE = 0.05 # Mức rất nhẹ
    
    # ============ TRAINING (Tối ưu tốc độ) ===========
    BATCH_SIZE = 16       # ⬆️ Tăng lên 16 (Vì V36 nhẹ hơn V35)
    EPOCHS = 200 
    PATIENCE = 200
    FOLDS_TO_RUN = [0,1,2,3,4]
    
    # ============ LOSS & STRATEGY ===========
    USE_HYBRID_LOSS = False    
    LABEL_SMOOTHING = 0.0 
    USE_FOCAL_LOSS = True 
    FOCAL_LOSS_GAMMA = 2.0 
    
    # ============ OPTIMIZER ===========
    WEIGHT_DECAY = 0.05       
    BETAS = (0.9, 0.999)    
    EPS = 1e-6
    
    # ============ LEARNING RATE (An toàn) ===========
    HEAD_LR = 1e-4            # Giữ mức 1e-4 để tránh NaN
    BACKBONE_LR = 1e-5        
    
    # ============ SCHEDULER ===========
    SCHEDULER_TYPE = 'cosine' 
    WARMUP_EPOCHS = 15        
    
    # ============ AUGMENTATION (Tắt Mixup để dùng Masking) ===========
    USE_TTA = True
    USE_MIXUP = False 
    MIXUP_PROB = 0.5 
    MIXUP_ALPHA = 0.4
    
    # ============ CONSISTENCY LOSS (V36 - Masking) ===========
    USE_AMP = False           # Tắt AMP (FP32) để tránh NaN tuyệt đối
    GRAD_CLIP = 0.5           
    USE_MASKING_LOSS = True   # ✅ Bật Masking Consistency
    MASKING_RATIO = 0.15      # Mức nhẹ nhàng 15%
    MASKING_LOSS_WEIGHT = 1.0 
    
cfg = Config()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print(f"\n✅ Config V36 (Single-Stream Bidirectional + Masking) Ready:")
print(f"   🔥 GPU: {cfg.DEVICE}")
print(f"   Model: DenseNet121 -> Single Stream Mamba (512D)")
print(f"   Strategy: Masking Loss (15%) | FP32 | Batch={cfg.BATCH_SIZE}")