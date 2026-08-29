import torch
import torch.nn as nn
import torch.nn.functional as F
from ..SubNets.FeatureNets import BERTEncoder
from .tcs_mamba import CoSSM

__all__ = ['TCS_Mamba']


class TuckerSharedLayer(nn.Module):
    """
    基于 Tucker 分解的共享子空间提取层。
    Equation: Z_shared = X x_2 U_mod x_3 U_feat
    """
    def __init__(self, input_dim, shared_dim, num_modes=3):
        super(TuckerSharedLayer, self).__init__()
        # U_mod: 模态混合矩阵 (M x M)
        self.U_mod = nn.Parameter(torch.eye(num_modes))
        # U_feat: 特征压缩矩阵 (d x d')
        self.U_feat = nn.Sequential(
            nn.Linear(input_dim, shared_dim, bias=False),
            nn.Dropout(0.3)
        )
        self.U_feat_inv = nn.Linear(shared_dim, input_dim, bias=False)

    def forward(self, x_tensor):
        # x_tensor: [B_total, Modes, Input_Dim] (Flattened Batch*Seq)

        # 1. Feature Projection
        z_step1 = self.U_feat(x_tensor) # [B_t, M, d']

        # 2. Modality Mixing
        # [B_t, M, d'] x [M, M] ->[B_t, M, d']
        z_shared = torch.einsum('bmd, nm -> bnd', z_step1, self.U_mod)
        return z_shared

    def reconstruct(self, z_shared):
        # Inverse operation for reconstruction loss
        z_rec_step1 = torch.einsum('bnd, nm -> bmd', z_shared, self.U_mod.t())
        x_hat = self.U_feat_inv(z_rec_step1)
        return x_hat


class ConstrainedCPLayer(nn.Module):
    """
    基于CP 分解的私有子空间提取层。
    每个模态有独立的投影矩阵，Rank-1 CP 分解。
    """
    def __init__(self, input_dim, private_dim, num_modes=3):
        super(ConstrainedCPLayer, self).__init__()
        self.num_modes = num_modes

        # 独立的投影层
        self.cp_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, private_dim),
                nn.LayerNorm(private_dim),
                nn.ReLU()
            ) for _ in range(num_modes)
        ])

        # 独立的重构层
        self.cp_inverse = nn.ModuleList([
            nn.Linear(private_dim, input_dim, bias=False)
            for _ in range(num_modes)
        ])

    def forward(self, x_tensor):
        # x_tensor: [B_total, Modes, Input_Dim]
        z_privates =[]
        for i in range(self.num_modes):
            x_slice = x_tensor[:, i, :]
            z_p = self.cp_projections[i](x_slice)
            z_privates.append(z_p)
        return torch.stack(z_privates, dim=1)

    def reconstruct(self, z_private):
        x_hats =[]
        for i in range(self.num_modes):
            z_slice = z_private[:, i, :]
            x_h = self.cp_inverse[i](z_slice)
            x_hats.append(x_h)
        return torch.stack(x_hats, dim=1)


class HLBF(nn.Module):
    """
    分层低秩双线性融合 (Hierarchical Low-rank Bilinear Fusion)
    """
    def __init__(self, shared_dim, private_dim, fusion_dim, dropout=0.3):
        super(HLBF, self).__init__()
        self.proj_sh = nn.Sequential(
            nn.Linear(shared_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_sp = nn.Sequential(
            nn.Linear(private_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.res_proj = nn.Linear(shared_dim, fusion_dim) if shared_dim != fusion_dim else nn.Identity()

    def forward(self, h_shared, h_private):
        z_sh = self.proj_sh(h_shared)
        z_sp = self.proj_sp(h_private)
        # Hadamard Product (Bilinear)
        z = z_sh * z_sp
        # Residual
        out = self.layer_norm(z + self.res_proj(h_shared))
        return out


class GatedModulationCoSSM(nn.Module):
    """
    GM-Manba: Gated Modulation Bidirectional Mamba
    采用确定性的特征调制 (FiLM) 和信息门控 (Gating) 来深度融合文本引导信息。
    """
    def __init__(self, base_cossm_module, hidden_dim):
        super().__init__()
        self.cossm = base_cossm_module

        # === 特征调制模块 (FiLM) ===
        self.fc_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.fc_beta = nn.Linear(hidden_dim, hidden_dim)

        # === 信息流门控模块 (Gating) ===
        self.fc_gate = nn.Linear(hidden_dim, hidden_dim)

        # 稳定特征
        self.layer_norm = nn.LayerNorm(hidden_dim)

        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)
        nn.init.zeros_(self.fc_gate.weight)
        # bias 初始化为负数或0，让早期偏向于残差
        nn.init.constant_(self.fc_gate.bias, 0.0)

    def forward(self, text_feat, other_feat):
        """
        text_feat: [B, L, D] (引导模态)
        other_feat: [B, L, D] (被修正模态)
        """
        # 1. 基础 Mamba 跨模态扫描
        _, mamba_out = self.cossm(text_feat, other_feat)

        # 2. 特征线性调制 (Modulation)
        gamma = self.fc_gamma(mamba_out)
        beta = self.fc_beta(mamba_out)
        # (1 + gamma)，保证初始状态为恒等映射，防止特征一开始就坍塌
        modulated_feat = (1.0 + gamma) * other_feat + beta

        # 3. 跨模态门控残差 (Gated Residual)
        gate = torch.sigmoid(self.fc_gate(mamba_out))

        # 动态融合，这里用 gate 来动态平衡 "Mamba深层特征" 和 "被调制后的原始特征"
        refined_feat = gate * mamba_out + (1.0 - gate) * modulated_feat
        refined_feat = self.layer_norm(refined_feat)

        return refined_feat


# ==============================================================================
# TCS_Mamba 主模型 (MIntRec 适配版)
# ==============================================================================

class TCS_Mamba(nn.Module):
    def __init__(self, args):
        super(TCS_Mamba, self).__init__()

        # -------------------【消融实验开关】-------------------
        self.use_tucker = getattr(args, 'use_tucker', True)      # 是否使用 Tucker 提取共享特征
        self.use_cp     = getattr(args, 'use_cp', True)          # 是否使用 CP 提取私有特征

        # mamba_type 支持:
        # 'gated': 完整版的 Gated Modulation Mamba
        # 'vanilla': 消融Gating与FiLM，仅使用纯基础版的 CoSSM Mamba
        # 'none': 完全消融 Mamba，不进行文本引导特征修正
        self.mamba_type = getattr(args, 'mamba_type', 'gated')

        self.use_hlbf   = getattr(args, 'use_hlbf', True)        # 是否使用 HLBF 融合模块
        # ------------------------------------------------------

        # --- 【模态组合开关】 (如: 'TAV', 'TA', 'TV', 'T', 'A', 'V') ---
        self.modalities = getattr(args, 'modalities', 'TAV').upper()
        self.use_t = 'T' in self.modalities
        self.use_a = 'A' in self.modalities
        self.use_v = 'V' in self.modalities

        # 1. 文本编码器 (BERT, MIntRec 的 BERTEncoder, token 级输出 [B, L, 768])
        self.use_bert = getattr(args, 'use_bert', True)
        self.use_finetune = getattr(args, 'use_finetune', True)
        if self.use_bert:
            self.text_model = BERTEncoder.from_pretrained(args.text_backbone, cache_dir = args.cache_path)

        # 维度配置 (MIntRec 的 DataManager 注入三个独立字段; 兼容 M-SENA 的 feature_dims 元组)
        feature_dims = getattr(args, 'feature_dims', None) or (args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim)
        self.orig_d_l, self.orig_d_a, self.orig_d_v = feature_dims
        self.d_model = args.dst_feature_dim_nheads[0]

        # 2. 基础特征编码 (Conv1d)
        self.conv_l = nn.Conv1d(self.orig_d_l, self.d_model, kernel_size=args.kernel_size_l, padding=1, bias=False)
        self.conv_a = nn.Conv1d(self.orig_d_a, self.d_model, kernel_size=args.kernel_size_a, padding=1, bias=False)
        self.conv_v = nn.Conv1d(self.orig_d_v, self.d_model, kernel_size=args.kernel_size_v, padding=1, bias=False)

        # 3. 张量分解层 (Tucker & CP)
        if self.use_tucker:
            self.tucker_shared = TuckerSharedLayer(input_dim=self.d_model, shared_dim=self.d_model)

        if self.use_cp:
            self.cp_private = ConstrainedCPLayer(input_dim=self.d_model, private_dim=self.d_model)

        # 4. 文本引导的精炼模块 (Mamba Variations)
        if self.mamba_type in ['vanilla', 'gated']:
            mamba_config = {
                'd_state': args.d_state,
                'd_conv': 4,
                'expand': 2,
                'bidirectional': True
            }
            # 基础版 Vanilla CoSSM
            self.base_cossm_a = CoSSM(
                num_layers=1, input_size=self.d_model, output_sizes=[self.d_model],
                d_ffn=self.d_model * 2, dropout=args.text_dropout, mamba_config=mamba_config
            )
            self.base_cossm_v = CoSSM(
                num_layers=1, input_size=self.d_model, output_sizes=[self.d_model],
                d_ffn=self.d_model * 2, dropout=args.text_dropout, mamba_config=mamba_config
            )

            # 进阶版 Gated CoSSM
            if self.mamba_type == 'gated':
                self.gated_mamba_a = GatedModulationCoSSM(self.base_cossm_a, self.d_model)
                self.gated_mamba_v = GatedModulationCoSSM(self.base_cossm_v, self.d_model)

        # 5. 融合与预测 (MIntRec: 分类任务, output_dim = num_labels)
        self.output_dim = args.num_labels

        # 融合层：HLBF / Simple Concat
        if self.use_hlbf:
            self.hlbf = HLBF(shared_dim=self.d_model * 3, private_dim=self.d_model, fusion_dim=self.d_model)
        else:
            self.simple_fusion = nn.Sequential(
                nn.Linear(self.d_model * 3 + self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU()
            )

        # 分类头
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.d_model // 2, self.output_dim)
        )
        self.head_shared = nn.Linear(self.d_model * 3, self.output_dim)
        self.head_private = nn.Linear(self.d_model, self.output_dim)


    def forward(self, text_feats, video_feats, audio_feats):
        # === 基础特征提取 ===
        if self.use_bert:
            if self.use_finetune:
                text = self.text_model(text_feats)
            else:
                with torch.no_grad():
                    text = self.text_model(text_feats)

        x_l_t = self.conv_l(text.transpose(1, 2))
        x_a_t = self.conv_a(audio_feats.transpose(1, 2))
        x_v_t = self.conv_v(video_feats.transpose(1, 2))

        x_l = x_l_t.transpose(1, 2)
        x_a = x_a_t.transpose(1, 2)
        x_v = x_v_t.transpose(1, 2)

        # 序列对齐
        target_len = x_l.shape[1]
        if x_a.shape[1] != target_len: x_a = F.interpolate(x_a.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
        if x_v.shape[1] != target_len: x_v = F.interpolate(x_v.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)

        # =================================================================
        # 模态消融 (Zero-Imputation)
        # =================================================================
        if not self.use_t: x_l = torch.zeros_like(x_l)
        if not self.use_a: x_a = torch.zeros_like(x_a)
        if not self.use_v: x_v = torch.zeros_like(x_v)

        X_tensor = torch.stack([x_l, x_v, x_a], dim=1) # [B, 3, L, D]
        B, M, L, D = X_tensor.shape
        X_flat = X_tensor.permute(0, 2, 1, 3).reshape(B * L, M, D)

        # =================================================================
        # 模块 1：张量分解 (Tucker & CP Ablation Support)
        # =================================================================
        X_hat_shared = 0
        X_hat_private = 0

        # --- 1.A: Shared Subspace (Tucker) ---
        if self.use_tucker:
            Z_shared_flat = self.tucker_shared(X_flat)
            Z_shared = Z_shared_flat.view(B, L, M, D)
            X_hat_shared = self.tucker_shared.reconstruct(Z_shared_flat)
        else:
            # 均值作为兜底的 Shared 特征
            X_mean = torch.mean(X_tensor, dim=1, keepdim=True).expand(-1, 3, -1, -1)
            Z_shared = X_mean.permute(0, 2, 1, 3)
            X_hat_shared = Z_shared.reshape(B * L, M, D) / 2 # Dummy value

        # --- 1.B: Private Subspace (CP) ---
        if self.use_cp:
            Z_private_flat = self.cp_private(X_flat)
            Z_private = Z_private_flat.view(B, L, M, D)
            h_sp_l = Z_private[:, :, 0, :]
            h_sp_v = Z_private[:, :, 1, :]
            h_sp_a = Z_private[:, :, 2, :]
            X_hat_private = self.cp_private.reconstruct(Z_private_flat)
        else:
            # 原始特征作为兜底的 Private 特征
            Z_private = X_tensor.permute(0, 2, 1, 3) #[B, L, M, D]
            h_sp_l, h_sp_v, h_sp_a = x_l, x_v, x_a
            X_hat_private = X_flat / 2 # Dummy value

        # 重构Loss的处理
        if not self.use_tucker and not self.use_cp:
            X_hat = X_tensor # 都被消融时，保证重构损失天然为 0
        else:
            X_hat = (X_hat_shared + X_hat_private).view(B, L, M, D).permute(0, 2, 1, 3)

        # =================================================================
        # 模块 2：跨模态 Mamba 提炼 (Mamba Ablation Support)
        # =================================================================
        if self.mamba_type == 'gated':
            # Gated 提炼版 (带门控和特征调制)
            refined_a_seq = self.gated_mamba_a(h_sp_l, h_sp_a)
            refined_v_seq = self.gated_mamba_v(h_sp_l, h_sp_v)

        elif self.mamba_type == 'vanilla':
            # Vanilla CoSSM 版 (仅使用 Mamba 的输出，不加 Gated Modulation)
            _, refined_a_seq = self.base_cossm_a(h_sp_l, h_sp_a)
            _, refined_v_seq = self.base_cossm_v(h_sp_l, h_sp_v)

        else: # 'none'
            # 完全消融版：直接跳过提炼过程
            refined_a_seq, refined_v_seq = h_sp_a, h_sp_v

        # =================================================================
        # 特征池化
        # =================================================================
        h_shared_seq = Z_shared.reshape(B, L, -1)
        h_shared_vec = torch.mean(h_shared_seq, dim=1) # [B, 3*d]

        h_sp_l_vec = torch.mean(h_sp_l, dim=1)
        h_sp_a_vec = torch.mean(refined_a_seq, dim=1)
        h_sp_v_vec = torch.mean(refined_v_seq, dim=1)

        h_private_enhanced = h_sp_l_vec + h_sp_a_vec + h_sp_v_vec #[B, d]

        # =================================================================
        # 模块 3：特征融合 (HLBF Ablation Support)
        # =================================================================
        if self.use_hlbf:
            h_final = self.hlbf(h_shared=h_shared_vec, h_private=h_private_enhanced)
        else:
            # 消融 HLBF: 退化为最简单的拼接特征 + MLP
            concat_feat = torch.cat([h_shared_vec, h_private_enhanced], dim=-1)
            h_final = self.simple_fusion(concat_feat)

        # 最终预测
        output = self.regressor(h_final)
        # 辅助输出暂存为属性(MISA 同款模式),供 manager 计算辅助损失
        self.aux_outputs = {
            'pred_shared': self.head_shared(h_shared_vec),
            'pred_private': self.head_private(h_private_enhanced),
            'X_origin': X_tensor,
            'X_recon': X_hat,
            'H_shared': h_shared_vec,
            'H_private': h_private_enhanced,
        }
        return output
