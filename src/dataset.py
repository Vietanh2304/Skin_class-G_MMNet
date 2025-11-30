import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import cfg

print("="*70)
print("DATA PREPROCESSING (HAM10000 - SAFE MODE)")
print("="*70)

def preprocess_metadata_for_transformer(df_train, df_val, df_test=None):
    """
    Phiên bản AN TOÀN TUYỆT ĐỐI: 
    1. Gom tất cả dữ liệu lại để fit LabelEncoder (tránh sót nhãn lạ ở tập Val/Test).
    2. Xử lý NaN triệt để cho cả cột số và cột phân loại.
    """
    print("⚡ Bắt đầu xử lý Metadata...")
    
    # Copy để không ảnh hưởng dữ liệu gốc
    df_train = df_train.copy()
    df_val = df_val.copy()
    if df_test is not None:
        df_test = df_test.copy()
    
    # Định nghĩa cột
    cat_cols = ["dx_type", "sex", "localization"]
    num_cols = ["age"]
    
    # 1. GOM DỮ LIỆU (QUAN TRỌNG ĐỂ KHÔNG BỊ LỖI INDEX)
    # Gom tất cả lại để LabelEncoder học được mọi giá trị có thể xuất hiện
    all_dfs = [df_train, df_val]
    if df_test is not None:
        all_dfs.append(df_test)
    
    full_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    
    # 2. XỬ LÝ SỐ (NUMERICAL)
    for c in num_cols:
        # Ép kiểu số, lỗi thành NaN
        full_df[c] = pd.to_numeric(full_df[c], errors='coerce')
        
        # Điền tuổi thiếu bằng trung bình (An toàn hơn median nếu dữ liệu ít)
        mean_val = full_df[c].mean()
        if pd.isna(mean_val): mean_val = 50.0 # Fallback
        
        full_df[c] = full_df[c].fillna(mean_val)
        
        # Chuẩn hóa (StandardScaler tốt hơn MinMaxScaler cho Transformer)
        scaler = StandardScaler()
        full_df[[c]] = scaler.fit_transform(full_df[[c]])

    # 3. XỬ LÝ PHÂN LOẠI (CATEGORICAL)
    cat_dims = []
    encoders = {}
    
    for c in cat_cols:
        # Chuyển về string và xử lý NaN
        full_df[c] = full_df[c].fillna("unknown").astype(str)
        full_df[c] = full_df[c].replace(['nan', 'NaN', 'UNK'], "unknown")
        
        # Fit LabelEncoder trên TOÀN BỘ dữ liệu
        le = LabelEncoder()
        le.fit(full_df[c])
        
        # Transform
        full_df[c] = le.transform(full_df[c])
        
        # Lưu lại số lượng class (Cộng thêm 1 để dự phòng cho Embedding)
        n_classes = len(le.classes_)
        cat_dims.append(n_classes)
        encoders[c] = le
        print(f"   > Cột '{c}': {n_classes} classes -> {le.classes_}")

    # 4. TRẢ DỮ LIỆU VỀ TỪNG PHẦN
    # Cắt full_df ra lại thành train, val, test
    len_train = len(df_train)
    len_val = len(df_val)
    
    train_meta_df = full_df.iloc[:len_train].copy()
    val_meta_df = full_df.iloc[len_train : len_train + len_val].copy()
    
    test_meta_df = None
    if df_test is not None:
        test_meta_df = full_df.iloc[len_train + len_val :].copy()
    
    # Chọn cột để trả về
    meta_cols = num_cols + cat_cols
    
    # Helper: Chuyển DataFrame thành Tensor
    def to_tensor(df):
        return torch.tensor(df[meta_cols].values, dtype=torch.float32)

    train_tensor = to_tensor(train_meta_df)
    val_tensor = to_tensor(val_meta_df)
    test_tensor = to_tensor(test_meta_df) if test_meta_df is not None else None
    
    print(f"✅ Metadata đã xử lý xong! (Num: {len(num_cols)}, Cat: {len(cat_cols)})")
    return (train_tensor, val_tensor, test_tensor), cat_dims, len(num_cols)


class HAM10000Dataset(Dataset):
    def __init__(self, df, meta_data, img_root, label_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.meta_data = meta_data # Đây là Tensor
        self.img_root = img_root   # List các đường dẫn ảnh
        self.label_map = label_map
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 1. Lấy thông tin cơ bản
        row = self.df.iloc[idx]
        img_id = str(row['image_id']).strip()
        
        # 2. Tìm ảnh (Hỗ trợ nhiều thư mục & đuôi file)
        img_path = None
        extensions = [".jpg", ".jpeg", ".png"]
        
        if isinstance(self.img_root, str):
            self.img_root = [self.img_root]
            
        for root in self.img_root:
            for ext in extensions:
                temp_path = os.path.join(root, img_id + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            if img_path: break
            
        # 3. Load ảnh (Có Fallback nếu lỗi)
        try:
            if img_path:
                img = np.array(Image.open(img_path).convert("RGB"))
            else:
                # Tạo ảnh đen nếu không tìm thấy (tránh crash)
                img = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        except Exception:
            img = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        
        # 4. Augmentation
        if self.transform:
            img = self.transform(image=img)['image']
            
        # 5. Lấy Metadata & Label
        meta = self.meta_data[idx] # Tensor đã xử lý ở trên
        label = torch.tensor(self.label_map[row['dx']], dtype=torch.long)
        
        return img, meta, label