# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm 
# from mamba_ssm import Mamba
# from src.config import cfg # Import cfg



# # ------------------------------------------------------------------
# # --- 1. BASIC BLOCKS ---\
# # ------------------------------------------------------------------

# class OptimizedMetadataEncoder(nn.Module):
#     # (Giữ nguyên code V22)
#     def __init__(self, cat_dims, num_continuous, embed_dim=32, output_dim=cfg.META_DIM):
#         super().__init__()
#         self.num_continuous = num_continuous
#         self.cat_embeddings = nn.ModuleList([nn.Embedding(num_classes, embed_dim) for num_classes in cat_dims])
#         self.num_processor = nn.Sequential(nn.LayerNorm(num_continuous), nn.Linear(num_continuous, embed_dim * 2), nn.GELU(), nn.LayerNorm(embed_dim * 2))
#         total_embed_dim = (embed_dim * len(cat_dims)) + (embed_dim * 2)
#         self.final_mlp = nn.Sequential(
#             nn.LayerNorm(total_embed_dim), nn.Dropout(cfg.META_DROPOUT),
#             nn.Linear(total_embed_dim, 128), nn.GELU(), 
#             nn.LayerNorm(128), nn.Dropout(cfg.META_DROPOUT),
#             nn.Linear(128, output_dim), nn.GELU(), 
#             nn.LayerNorm(output_dim)
#         )
#     def forward(self, meta_tensor):
#         x_num = meta_tensor[:, :self.num_continuous]; x_cat = meta_tensor[:, self.num_continuous:].long()
#         cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]; x_cat_proc = torch.cat(cat_embeds, dim=1)
#         x_num_proc = self.num_processor(x_num); x_combined = torch.cat([x_num_proc, x_cat_proc], dim=1)
#         return self.final_mlp(x_combined)

# class PretrainedHybridStem(nn.Module):
#     # (🔥 V35: DENSE-NET 121 LIGHTWEIGHT)
#     def __init__(self, fine_dim, coarse_dim, pretrained=True):
#         super().__init__()
#         self.fine_dim = fine_dim
#         self.coarse_dim = coarse_dim
        
#         # 🔥 Change 1: Use DenseNet-121
#         self.backbone = timm.create_model(
#             'densenet121', pretrained=pretrained,
#             features_only=True, out_indices=(2, 3) # Output Fine (Stride 8) & Coarse (Stride 16)
#         )
        
#         feature_channels = self.backbone.feature_info.channels()
#         fine_ch, coarse_ch = feature_channels[0], feature_channels[1] 
        
#         print(f"  [Stem] DenseNet-121 (Pretrained): Fine={fine_ch}ch, Coarse={coarse_ch}ch")
        
#         # 🔥 Dynamic calculation MUST use cfg.IMG_SIZE (224)
#         img_size = cfg.IMG_SIZE 
        
#         self.proj_fine = nn.Conv2d(fine_ch, self.fine_dim, kernel_size=1)
#         self.norm_fine = nn.LayerNorm(self.fine_dim)
#         # Tính Patch Count dựa trên Stride 8 (224/8 = 28)
#         self.num_patches_fine = (img_size // 8) * (img_size // 8) 
#         print(f"    [Fine] Proj -> {self.fine_dim}D (Patches: {self.num_patches_fine})")
        
#         self.proj_coarse = nn.Conv2d(coarse_ch, self.coarse_dim, kernel_size=1)
#         self.norm_coarse = nn.LayerNorm(self.coarse_dim)
#         # Tính Patch Count dựa trên Stride 16 (224/16 = 14)
#         self.num_patches_coarse = (img_size // 16) * (img_size // 16)
#         print(f"    [Coarse] Proj -> {self.coarse_dim}D (Patches: {self.num_patches_coarse})")

#     def forward(self, x):
#         features = self.backbone(x)
#         x_fine = self.proj_fine(features[0]).flatten(2).transpose(1, 2)
#         x_coarse = self.proj_coarse(features[1]).flatten(2).transpose(1, 2)
#         return self.norm_fine(x_fine), self.norm_coarse(x_coarse)

# # ------------------------------------------------------------------
# # --- 2. ADAPTIVE FiLM (Giữ nguyên) ---\
# # ------------------------------------------------------------------
# class AdaptiveFiLMLayer(nn.Module):
#     # (Giữ nguyên code V22)
#     def __init__(self, feature_dim, condition_dim):
#         super().__init__(); self.meta_proj = nn.Sequential(nn.Linear(condition_dim, 128), nn.GELU(), nn.LayerNorm(128))
#         self.img_proj = nn.Sequential(nn.Linear(feature_dim, 128), nn.GELU(), nn.LayerNorm(128))
#         self.film_gen = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, feature_dim * 2))
#         print(f"    [A-FiLM] Adaptive FiLM created (meta + img → gamma/beta)")
#     def forward(self, features, context):
#         img_summary = features.mean(dim=1); meta_emb = self.meta_proj(context); img_emb = self.img_proj(img_summary)
#         combined = torch.cat([meta_emb, img_emb], dim=-1); gamma_beta = self.film_gen(combined)
#         gamma, beta = gamma_beta.chunk(2, dim=-1); return features * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

# # ------------------------------------------------------------------
# # --- 3. ADAPTIVE FiLMed MAMBA BLOCK (Giữ nguyên) ---\
# # ------------------------------------------------------------------
# class AdaptiveFiLMedMambaBlock(nn.Module):
#     # (Giữ nguyên code V22)
#     def __init__(self, d_model, mlp_dim, condition_dim, dropout):
#         super().__init__(); self.norm_mamba = nn.LayerNorm(d_model)
#         self.film_mamba = AdaptiveFiLMLayer(d_model, condition_dim)
#         self.mamba_core = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
#         self.norm_mlp = nn.LayerNorm(d_model); self.film_mlp = AdaptiveFiLMLayer(d_model, condition_dim)
#         self.mlp = nn.Sequential(nn.Linear(d_model, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, d_model), nn.Dropout(dropout))
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, x, meta_vec):
#         x_norm = self.norm_mamba(x); x_modulated = self.film_mamba(x_norm, meta_vec)
#         x = x + self.dropout(self.mamba_core(x_modulated)); x_norm = self.norm_mlp(x)
#         x_modulated = self.film_mlp(x_norm, meta_vec); x = x + self.dropout(self.mlp(x_modulated))
#         return x

# # ------------------------------------------------------------------
# # --- 4. 🔥 NÂNG CẤP: CROSS-SCALE (Lấy từ V17) ---\
# # ------------------------------------------------------------------
# class CrossScaleAttention(nn.Module):
#     # (Lấy code từ V17 - Bất đối xứng)
#     def __init__(self, query_dim, kv_dim, num_heads=8, dropout=0.1):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=query_dim, kdim=kv_dim, vdim=kv_dim,
#             num_heads=num_heads, dropout=dropout, batch_first=True
#         )
#         self.norm = nn.LayerNorm(query_dim) 
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, query_tokens, key_value_tokens):
#         attn_out, _ = self.cross_attn(query_tokens, key_value_tokens, key_value_tokens)
#         return query_tokens + self.dropout(self.norm(attn_out))

# # ------------------------------------------------------------------
# # --- 5. 🔥 NÂNG CẤP: FUSION (Lấy từ V17) ---\
# # ------------------------------------------------------------------
# class AdvancedMultiScaleFusion(nn.Module):
#     # (Lấy code từ V17 - Bất đối xứng)
#     def __init__(self, fine_dim, coarse_dim, num_heads=8, dropout=0.1):
#         super().__init__()
#         self.fusion_dim = coarse_dim 
#         self.fine_proj = nn.Linear(fine_dim, self.fusion_dim)
#         print(f"  [Fusion V24] Advanced Multi-Scale (Chiếu {fine_dim}D -> {self.fusion_dim}D)")
#         self.cross_attn = nn.MultiheadAttention(
#             self.fusion_dim, num_heads, dropout=dropout, batch_first=True
#         )
#         self.norm1 = nn.LayerNorm(self.fusion_dim); self.scale_weights = nn.Parameter(torch.ones(2))
#         self.refine_mlp = nn.Sequential(
#             nn.Linear(self.fusion_dim, self.fusion_dim * 4), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(self.fusion_dim * 4, self.fusion_dim), nn.Dropout(dropout)
#         )
#         self.norm2 = nn.LayerNorm(self.fusion_dim)
#     def forward(self, cls_fine, cls_coarse):
#         cls_fine_proj = self.fine_proj(cls_fine).unsqueeze(1); cls_coarse_uns = cls_coarse.unsqueeze(1)
#         cls_seq = torch.cat([cls_fine_proj, cls_coarse_uns], dim=1) 
#         attn_out, _ = self.cross_attn(cls_seq, cls_seq, cls_seq); cls_seq = self.norm1(cls_seq + attn_out)
#         weights = F.softmax(self.scale_weights, dim=0); weighted = weights[0] * cls_seq[:, 0] + weights[1] * cls_seq[:, 1]
#         fused = self.norm2(weighted + self.refine_mlp(weighted)); return fused

# # =========================================================================
# # 🔥 G_MMNet V24 (EffNetB1-Stem + ASYMMETRIC Mamba + BỎ Aux)
# # =========================================================================
# class G_MMNet(nn.Module):
#     def __init__(self, num_classes, cat_dims, num_continuous, use_cross_scale=True):
#         super().__init__()
#         self.num_continuous = num_continuous
#         self.use_cross_scale = use_cross_scale
        
#         print(f"\n{'='*60}")
#         print(f"🔥 BUILDING G_MMNet V24 (EffNetB1-Stem + Asymmetric Mamba)")
#         print(f"{'='*60}")
        
#         # 🔥 Đọc Dims Bất Đối Xứng từ cfg
#         self.dim_fine = cfg.IMG_EMBED_DIM_FINE     # 192
#         self.dim_coarse = cfg.IMG_EMBED_DIM_COARSE # 768
#         self.mlp_dim_fine = cfg.FUSION_MLP_DIM_FINE     # 768
#         self.mlp_dim_coarse = cfg.FUSION_MLP_DIM_COARSE # 3072
        
#         print(f"  [Dims] Fine: {self.dim_fine}D | Coarse: {self.dim_coarse}D")
        
#         # 1. Metadata (Giữ nguyên) & Stem (🔥 Nâng cấp)
#         self.meta_encoder = OptimizedMetadataEncoder(cat_dims, num_continuous, output_dim=cfg.META_DIM)
#         print(f"  [Meta] Encoder: {num_continuous} num + {len(cat_dims)} cat → {cfg.META_DIM}D")
        
#         self.stem = PretrainedHybridStem(
#             fine_dim=self.dim_fine, 
#             coarse_dim=self.dim_coarse, 
#             pretrained=True
#         )
        
#         # 2. CLS & Pos Embed (🔥 Nâng cấp)
#         self.cls_token_fine = nn.Parameter(torch.zeros(1, 1, self.dim_fine))
#         self.pos_embed_fine = nn.Parameter(torch.zeros(1, self.stem.num_patches_fine + 1, self.dim_fine))
#         self.cls_token_coarse = nn.Parameter(torch.zeros(1, 1, self.dim_coarse))
#         self.pos_embed_coarse = nn.Parameter(torch.zeros(1, self.stem.num_patches_coarse + 1, self.dim_coarse))
#         nn.init.trunc_normal_(self.pos_embed_fine, std=0.02); nn.init.trunc_normal_(self.cls_token_fine, std=0.02)
#         nn.init.trunc_normal_(self.pos_embed_coarse, std=0.02); nn.init.trunc_normal_(self.cls_token_coarse, std=0.02)
        
#         # 3. Dual Towers (🔥 Nâng cấp)
#         NUM_LAYERS = cfg.FUSION_NUM_LAYERS
#         print(f"  [Towers] Building {NUM_LAYERS} layers × 2 streams (Adaptive FiLM)...\\n")
#         self.tower_fine = nn.ModuleList([
#             AdaptiveFiLMedMambaBlock(self.dim_fine, self.mlp_dim_fine, cfg.META_DIM, cfg.FUSION_DROPOUT)
#             for _ in range(NUM_LAYERS)
#         ])
#         self.tower_coarse = nn.ModuleList([
#             AdaptiveFiLMedMambaBlock(self.dim_coarse, self.mlp_dim_coarse, cfg.META_DIM, cfg.FUSION_DROPOUT)
#             for _ in range(NUM_LAYERS)
#         ])
        
#         # 4. Cross-Attention (🔥 Nâng cấp)
#         if self.use_cross_scale:
#             self.cross_attns_f2c = nn.ModuleList([
#                 CrossScaleAttention(self.dim_fine, self.dim_coarse, num_heads=cfg.FUSION_NUM_HEADS, dropout=cfg.FUSION_DROPOUT) 
#                 for _ in range(NUM_LAYERS)
#             ])
#             self.cross_attns_c2f = nn.ModuleList([
#                 CrossScaleAttention(self.dim_coarse, self.dim_fine, num_heads=cfg.FUSION_NUM_HEADS, dropout=cfg.FUSION_DROPOUT)
#                 for _ in range(NUM_LAYERS)
#             ])
#             print(f"  [Cross-Attn] Dense cross-scale (192D/768D): {NUM_LAYERS} layers")
        
#         # 5. Fusion Nâng cao (🔥 Nâng cấp)
#         self.fusion = AdvancedMultiScaleFusion(
#             fine_dim=self.dim_fine, 
#             coarse_dim=self.dim_coarse, 
#             num_heads=cfg.FUSION_NUM_HEADS, 
#             dropout=cfg.FUSION_DROPOUT
#         )
        
#         # 6. Main Head (🔥 Nâng cấp)
#         fused_cls_dim = self.dim_coarse # Input là 768D
#         self.head = nn.Sequential( 
#             nn.LayerNorm(fused_cls_dim), nn.Dropout(cfg.FUSION_DROPOUT),
#             nn.Linear(fused_cls_dim, 512), nn.GELU(),
#             nn.LayerNorm(512), nn.Dropout(cfg.FUSION_DROPOUT),
#             nn.Linear(512, 256), nn.GELU(),
#             nn.LayerNorm(256), nn.Dropout(cfg.FUSION_DROPOUT),
#             nn.Linear(256, num_classes)
#         )
#         print(f"  [Head] 3-layer: {fused_cls_dim}→512→256→{num_classes}")
        
#         # 7. ❌ XÓA AUX HEADS
#         print(f"  [Aux Heads] Đã TẮT (Thiết kế V24)")
        
#         print(f"\n{'='*60}")
#         print(f"✅ G_MMNet V24 (Asymmetric Pre-trained) READY")
#         print(f"{'='*60}")

#     def forward(self, img, meta):
#         B = img.shape[0]
        
#         # Dropout (Giữ nguyên)
#         if self.training:
#             if cfg.META_FEATURE_DROPOUT_RATE > 0:
#                 meta_num = meta[:, :self.num_continuous]; meta_cat = meta[:, self.num_continuous:]
#                 keep_prob = 1.0 - cfg.META_FEATURE_DROPOUT_RATE
#                 if self.num_continuous > 0:
#                     mask = torch.bernoulli(torch.full((1, meta_num.shape[1]), keep_prob, device=meta.device))
#                     if keep_prob > 0: mask = mask / keep_prob
#                     meta_num = meta_num * mask
#                 if meta_cat.shape[1] > 0:
#                     mask = torch.bernoulli(torch.full((1, meta_cat.shape[1]), keep_prob, device=meta.device))
#                     meta_cat = meta_cat * mask
#                 meta = torch.cat([meta_num, meta_cat], dim=1)
#             if torch.rand(1) < cfg.MODALITY_DROPOUT_RATE:
#                 meta = torch.zeros_like(meta)
        
#         # 1-3. Encode & Towers
#         meta_vec = self.meta_encoder(meta)
#         x_fine, x_coarse = self.stem(img) 
        
#         x_fine = torch.cat([self.cls_token_fine.expand(B, -1, -1), x_fine], dim=1) + self.pos_embed_fine
#         x_coarse = torch.cat([self.cls_token_coarse.expand(B, -1, -1), x_coarse], dim=1) + self.pos_embed_coarse
        
#         for i in range(len(self.tower_fine)):
#             x_fine = self.tower_fine[i](x_fine, meta_vec)
#             x_coarse = self.tower_coarse[i](x_coarse, meta_vec)
#             if self.use_cross_scale:
#                 x_fine_attn = self.cross_attns_f2c[i](x_fine, x_coarse)
#                 x_coarse_attn = self.cross_attns_c2f[i](x_coarse, x_fine)
#                 x_fine = x_fine_attn
#                 x_coarse = x_coarse_attn
        
#         # 4. CLS extraction
#         cls_fine = x_fine[:, 0]     # (B, 192)
#         cls_coarse = x_coarse[:, 0] # (B, 768)
            
#         # 5. Fusion
#         cls_fused = self.fusion(cls_fine, cls_coarse) # (B, 768)
        
#         # 6. Classification
#         main_logits = self.head(cls_fused)

#         # 7. 🔥 CHỈ RETURN 1 GIÁ TRỊ
#         return main_logits

# print("="*70)
# print("✅ MODEL V24 READY - (EffNetB1-Stem + Asymmetric Mamba)")
# print("="*70 + "\n")
# =========================================================================
# CELL 7: G_MMNet V36 (🔥 Single-Stream Bidirectional Mamba - Lite)
# =========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 
from mamba_ssm import Mamba
from src.config import cfg 

print("="*70)
print(f"MODEL V36 - 🔥 Single-Stream Bidirectional Mamba (Optimization)")
print("="*70)

# ------------------------------------------------------------------
# 1. META ENCODER & FILM (Giữ nguyên)
# ------------------------------------------------------------------
class OptimizedMetadataEncoder(nn.Module):
    def __init__(self, cat_dims, num_continuous, embed_dim=32, output_dim=cfg.META_DIM):
        super().__init__()
        self.num_continuous = num_continuous
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
        x_num = meta_tensor[:, :self.num_continuous]; x_cat = meta_tensor[:, self.num_continuous:].long()
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]; x_cat_proc = torch.cat(cat_embeds, dim=1)
        x_num_proc = self.num_processor(x_num); x_combined = torch.cat([x_num_proc, x_cat_proc], dim=1)
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
# 2. 🔥 NEW: BIDIRECTIONAL MAMBA BLOCK (Quét 2 chiều)
# ------------------------------------------------------------------
class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model, condition_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.film = AdaptiveFiLMLayer(d_model, condition_dim)
        
        # Mamba Core (Vẫn dùng thư viện chuẩn)
        self.mamba_forward = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_backward = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        
        # Projection để gộp 2 chiều lại
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, meta_vec):
        # 1. Norm + FiLM (Điều biến bởi Metadata)
        x_norm = self.norm(x)
        x_mod = self.film(x_norm, meta_vec)
        
        # 2. Bidirectional Scan (Quét Xuôi + Quét Ngược)
        # Xuôi
        out_fwd = self.mamba_forward(x_mod)
        
        # Ngược (Lật chuỗi -> Mamba -> Lật lại)
        x_rev = torch.flip(x_mod, dims=[1])
        out_bwd = self.mamba_backward(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        # 3. Gộp + Residual
        out_mixed = self.out_proj(out_fwd + out_bwd)
        return x + self.dropout(out_mixed)

# ------------------------------------------------------------------
# 3. SINGLE STREAM MODEL (V36)
# ------------------------------------------------------------------
class G_MMNet(nn.Module):
    def __init__(self, num_classes, cat_dims, num_continuous, use_cross_scale=True):
        super().__init__()
        self.num_continuous = num_continuous
        
        print(f"🔥 BUILDING G_MMNet V36 (Single-Stream Bidirectional)")
        
        # 1. Backbone (DenseNet-121)
        self.backbone = timm.create_model('densenet121', pretrained=True, features_only=True, out_indices=(2, 3))
        dims = self.backbone.feature_info.channels() # [512, 1024]
        dim_fine, dim_coarse = dims[0], dims[1]
        
        print(f"  [Stem] DenseNet-121: Fine={dim_fine}, Coarse={dim_coarse}")
        
        # 2. Early Fusion & Projection
        # Mục tiêu: D_MODEL = 512 (Nhẹ nhàng, hiệu quả)
        self.D_MODEL = 512 
        
        self.proj_fine = nn.Conv2d(dim_fine, 256, kernel_size=1)   # 512 -> 256
        self.proj_coarse = nn.Conv2d(dim_coarse, 256, kernel_size=1) # 1024 -> 256
        
        self.fusion_conv = nn.Conv2d(512, self.D_MODEL, kernel_size=1) # Gộp 256+256 -> 512
        
        # 3. Meta Encoder
        self.meta_encoder = OptimizedMetadataEncoder(cat_dims, num_continuous, output_dim=cfg.META_DIM)
        
        # 4. Main Mamba Tower (1 Tháp duy nhất - Sâu hơn)
        # Dùng 6 lớp (thay vì 4+4=8 lớp cũ)
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(self.D_MODEL, cfg.META_DIM, cfg.FUSION_DROPOUT)
            for _ in range(6) 
        ])
        print(f"  [Encoder] Single Tower: {len(self.layers)} Bidirectional Layers (Dim={self.D_MODEL})")
        
        # 5. Classifier Head
        self.head = nn.Sequential(
            nn.LayerNorm(self.D_MODEL),
            nn.Dropout(cfg.FUSION_DROPOUT),
            nn.Linear(self.D_MODEL, num_classes)
        )

    def forward(self, img, meta):
        B = img.shape[0]
        
        # --- A. Stem & Early Fusion ---
        feats = self.backbone(img)
        f_fine, f_coarse = feats[0], feats[1] 
        
        # Chiếu về cùng channel
        f_fine = self.proj_fine(f_fine)     
        f_coarse = self.proj_coarse(f_coarse) 
        
        # Upsample Coarse cho bằng Fine
        f_coarse_up = F.interpolate(f_coarse, size=f_fine.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate (Gộp)
        f_fused = torch.cat([f_fine, f_coarse_up], dim=1) 
        f_fused = self.fusion_conv(f_fused) 
        
        # Flatten thành chuỗi (Sequence)
        # (B, C, H, W) -> (B, H*W, C)
        x_seq = f_fused.flatten(2).transpose(1, 2) 
        
        # --- B. Metadata ---
        if self.training and cfg.META_FEATURE_DROPOUT_RATE > 0:
            meta_num = meta[:, :self.num_continuous]; meta_cat = meta[:, self.num_continuous:]
            keep_prob = 1.0 - cfg.META_FEATURE_DROPOUT_RATE
            if self.num_continuous > 0:
                mask = torch.bernoulli(torch.full((1, meta_num.shape[1]), keep_prob, device=meta.device))
                if keep_prob > 0: mask = mask / keep_prob
                meta_num = meta_num * mask
            if meta_cat.shape[1] > 0:
                mask = torch.bernoulli(torch.full((1, meta_cat.shape[1]), keep_prob, device=meta.device))
                meta_cat = meta_cat * mask
            meta = torch.cat([meta_num, meta_cat], dim=1)
        if torch.rand(1) < cfg.MODALITY_DROPOUT_RATE:
            meta = torch.zeros_like(meta)
            
        meta_vec = self.meta_encoder(meta)
        
        # --- C. Single Mamba Tower ---
        for layer in self.layers:
            x_seq = layer(x_seq, meta_vec)
            
        # --- D. Pooling & Head ---
        x_pool = x_seq.mean(dim=1) 
        logits = self.head(x_pool)
        
        return logits

print("="*70)
print("✅ MODEL V36 READY (Single-Stream / Bi-Directional)")
print("="*70 + "\n")