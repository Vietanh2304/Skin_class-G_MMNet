import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 
from mamba_ssm import Mamba
from src.config import cfg 

print("="*70)
print(f"MODEL V36 - 🔥 Single-Stream Bidirectional Mamba + V-JEPA Ready")
print("="*70)

# ------------------------------------------------------------------
# 1. META ENCODER & FILM (SAFE MODE ADDED)
# ------------------------------------------------------------------
class OptimizedMetadataEncoder(nn.Module):
    def __init__(self, cat_dims, num_continuous, embed_dim=32, output_dim=cfg.META_DIM):
        super().__init__()
        self.num_continuous = num_continuous
        # Lưu giới hạn để kẹp giá trị (Tránh lỗi Index Out of Bounds)
        self.cat_limits = nn.Parameter(torch.tensor(cat_dims, dtype=torch.long), requires_grad=False)
        
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_classes, embed_dim) for num_classes in cat_dims])
        self.num_processor = nn.Sequential(nn.LayerNorm(num_continuous), nn.Linear(num_continuous, embed_dim * 2), nn.GELU(), nn.LayerNorm(embed_dim * 2))
        
        total_embed_dim = (embed_dim * len(cat_dims)) + (embed_dim * 2)
        self.final_mlp = nn.Sequential(
            nn.LayerNorm(total_embed_dim), nn.Dropout(cfg.META_DROPOUT),
            nn.Linear(total_embed_dim, 128), nn.GELU(), 
            nn.LayerNorm(128), nn.Dropout(cfg.META_DROPOUT),
            nn.Linear(128, output_dim), nn.GELU(), 
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, meta_tensor):
        x_num = meta_tensor[:, :self.num_continuous]
        x_cat = meta_tensor[:, self.num_continuous:].long()
        
        # --- 🔥 SAFE GUARD: Tự động sửa lỗi index ---
        for i in range(len(self.cat_embeddings)):
            limit = self.cat_limits[i]
            # Kẹp giá trị trong khoảng [0, limit-1]
            x_cat[:, i] = torch.clamp(x_cat[:, i], min=0, max=limit - 1)
        # ------------------------------------------

        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x_cat_proc = torch.cat(cat_embeds, dim=1)
        
        x_num_proc = self.num_processor(x_num)
        x_combined = torch.cat([x_num_proc, x_cat_proc], dim=1)
        
        return self.final_mlp(x_combined)

class AdaptiveFiLMLayer(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super().__init__(); self.meta_proj = nn.Sequential(nn.Linear(condition_dim, 128), nn.GELU(), nn.LayerNorm(128))
        self.img_proj = nn.Sequential(nn.Linear(feature_dim, 128), nn.GELU(), nn.LayerNorm(128))
        self.film_gen = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, feature_dim * 2))
    def forward(self, features, context):
        img_summary = features.mean(dim=1); meta_emb = self.meta_proj(context); img_emb = self.img_proj(img_summary)
        combined = torch.cat([meta_emb, img_emb], dim=-1); gamma_beta = self.film_gen(combined)
        gamma, beta = gamma_beta.chunk(2, dim=-1); return features * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

# ------------------------------------------------------------------
# 2. BIDIRECTIONAL MAMBA BLOCK
# ------------------------------------------------------------------
class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model, condition_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.film = AdaptiveFiLMLayer(d_model, condition_dim)
        self.mamba_forward = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_backward = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, meta_vec):
        x_norm = self.norm(x)
        x_mod = self.film(x_norm, meta_vec)
        out_fwd = self.mamba_forward(x_mod)
        x_rev = torch.flip(x_mod, dims=[1])
        out_bwd = self.mamba_backward(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])
        out_mixed = self.out_proj(out_fwd + out_bwd)
        return x + self.dropout(out_mixed)

# ------------------------------------------------------------------
# 3. SINGLE STREAM MODEL (V36 - UPDATED FOR V-JEPA)
# ------------------------------------------------------------------
class G_MMNet(nn.Module):
    def __init__(self, num_classes, cat_dims, num_continuous, use_cross_scale=True):
        super().__init__()
        self.num_continuous = num_continuous
        
        # 1. Backbone
        self.backbone = timm.create_model('densenet121', pretrained=True, features_only=True, out_indices=(2, 3))
        dims = self.backbone.feature_info.channels()
        dim_fine, dim_coarse = dims[0], dims[1]
        
        # 2. Projection
        self.D_MODEL = 512 
        self.proj_fine = nn.Conv2d(dim_fine, 256, kernel_size=1)
        self.proj_coarse = nn.Conv2d(dim_coarse, 256, kernel_size=1)
        self.fusion_conv = nn.Conv2d(512, self.D_MODEL, kernel_size=1)
        
        # 3. Meta Encoder
        self.meta_encoder = OptimizedMetadataEncoder(cat_dims, num_continuous, output_dim=cfg.META_DIM)
        
        # 4. Mamba Tower
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(self.D_MODEL, cfg.META_DIM, cfg.FUSION_DROPOUT)
            for _ in range(6) 
        ])
        
        # 5. Classifier Head (Tách Norm ra để lấy Feature)
        self.norm_head = nn.LayerNorm(self.D_MODEL)
        self.dropout_head = nn.Dropout(cfg.FUSION_DROPOUT)
        self.fc_head = nn.Linear(self.D_MODEL, num_classes)

    def forward(self, img, meta, return_feats=False): # <--- Thêm cờ return_feats
        B = img.shape[0]
        
        # A. Vision Stem
        feats = self.backbone(img)
        f_fine, f_coarse = feats[0], feats[1] 
        f_fine = self.proj_fine(f_fine)     
        f_coarse = self.proj_coarse(f_coarse) 
        f_coarse_up = F.interpolate(f_coarse, size=f_fine.shape[-2:], mode='bilinear', align_corners=False)
        f_fused = torch.cat([f_fine, f_coarse_up], dim=1) 
        f_fused = self.fusion_conv(f_fused) 
        x_seq = f_fused.flatten(2).transpose(1, 2) 
        
        # B. Metadata
        if self.training and cfg.META_FEATURE_DROPOUT_RATE > 0:
            meta_num = meta[:, :self.num_continuous]; meta_cat = meta[:, self.num_continuous:]
            keep_prob = 1.0 - cfg.META_FEATURE_DROPOUT_RATE
            if self.num_continuous > 0:
                mask = torch.bernoulli(torch.full((1, meta_num.shape[1]), keep_prob, device=meta.device))
                if keep_prob > 0: mask = mask / keep_prob
                meta_num = meta_num * mask
            if meta_cat.shape[1] > 0:
                mask = torch.bernoulli(torch.full((1, meta_cat.shape[1]), keep_prob, device=meta.device))
                if keep_prob > 0: mask = mask / keep_prob
                meta_cat = meta_cat * mask
            meta = torch.cat([meta_num, meta_cat], dim=1)
        
        if self.training and torch.rand(1) < cfg.MODALITY_DROPOUT_RATE:
             meta = torch.zeros_like(meta)
            
        meta_vec = self.meta_encoder(meta)
        
        # C. Mamba Tower
        for layer in self.layers:
            x_seq = layer(x_seq, meta_vec)
            
        # D. Head
        x_pool = x_seq.mean(dim=1) 
        
        # Lấy Feature Vector sạch (Trước khi vào lớp Classification cuối)
        features = self.norm_head(x_pool) 
        
        logits = self.fc_head(self.dropout_head(features))
        
        if return_feats:
            return logits, features # Trả về cả hai cho V-JEPA
            
        return logits