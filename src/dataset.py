import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from src.config import cfg # Import cfg

print("="*70)
print("DATA PREPROCESSING (DÙNG CHO HAM10000)")
print("="*70)

def preprocess_metadata_for_transformer(df_train, df_val, df_test):
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    
    # Các cột của HAM10000
    raw_cat_cols = ["dx_type", "sex", "localization"]
    raw_num_cols = ["age"]
    
    final_num_cols = []
    final_cat_cols = []
    
    # Process numerical
    for c in raw_num_cols:
        for df in [df_train, df_val, df_test]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        flag_col = f"{c}_MISSING_FLAG"
        for df in [df_train, df_val, df_test]:
            df[flag_col] = df[c].isna().astype(int).astype(str)
            df[flag_col] = df[flag_col].replace('1', 'MISSING').replace('0', 'PRESENT')
        final_cat_cols.append(flag_col)
        
        median_val = df_train[c].median()
        if pd.isna(median_val):
            median_val = 50.0
        for df in [df_train, df_val, df_test]:
            df[c] = df[c].fillna(median_val)
        
        final_num_cols.append(c)
    
    scaler = MinMaxScaler()
    df_train[final_num_cols] = scaler.fit_transform(df_train[final_num_cols])
    df_val[final_num_cols] = scaler.transform(df_val[final_num_cols])
    df_test[final_num_cols] = scaler.transform(df_test[final_num_cols])
    
    all_cat_cols = raw_cat_cols + final_cat_cols
    cat_dims = []
    
    for c in all_cat_cols:
        for df in [df_train, df_val, df_test]:
            df[c] = df[c].astype(str).replace('nan', 'MISSING')
            df[c] = df[c].replace('UNK', 'MISSING').replace('unknown', 'MISSING')
        
        le = LabelEncoder()
        train_cats = df_train[c].unique()
        le.fit(np.append(train_cats, 'UNKNOWN'))
        
        df_train[c] = le.transform(df_train[c])
        df_val[c] = df_val[c].map(lambda s: s if s in le.classes_ else 'UNKNOWN')
        df_val[c] = le.transform(df_val[c].astype(str))
        df_test[c] = df_test[c].map(lambda s: s if s in le.classes_ else 'UNKNOWN')
        df_test[c] = le.transform(df_test[c].astype(str))
        cat_dims.append(len(le.classes_))
    
    meta_cols = final_num_cols + all_cat_cols
    
    print(f"  ✅ Metadata processed (HAM10000):\\n     Total features: {len(meta_cols)}")
    
    return (df_train[meta_cols], df_val[meta_cols], df_test[meta_cols]), cat_dims, len(final_num_cols)


class HAM10000Dataset(Dataset):
    """Dataset gốc trả về (img, meta, label)"""
    def __init__(self, df, meta_df, img_root, label_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.meta_df = meta_df.reset_index(drop=True)
        self.img_root = img_root # Đây là một list [root1, root2]
        self.transform = transform
        self.label_map = label_map
        print(f"  [Dataset] Created: {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Logic của HAM10000
        img_id = str(row['image_id']).strip() + ".jpg"
        meta = self.meta_df.iloc[idx].values.astype(np.float32)
        label = self.label_map[row['dx']]
        
        img_path = None
        for root in self.img_root: # Lặp qua list các thư mục gốc
            potential_path = os.path.join(root, img_id)
            if os.path.exists(potential_path):
                img_path = potential_path; break
        try:
            if img_path: img = np.array(Image.open(img_path).convert("RGB"))
            else: img = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        except Exception:
            img = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, torch.tensor(meta, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

print("✅ Data preprocessing (HAM10000) loaded")
print("="*70 + "\n")