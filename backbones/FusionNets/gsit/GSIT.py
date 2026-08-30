"""
GSIT backbone ("Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs").
Adapted from M-SENA: forward follows the MIntRec modality order
(text_feats, video_feats, audio_feats), the text encoder is MIntRec's
BERTEncoder, and the output head is `num_labels` classes. The CrossModalGraph
hierarchical graph attention (xformers block-diagonal memory-efficient
attention) is reused verbatim under gsit/modules/.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from ...SubNets.FeatureNets import BERTEncoder
from .modules import CrossModalGraph

__all__ = ['GSIT']


class GSIT(nn.Module):
    def __init__(self, args):
        super(GSIT, self).__init__()
        self.use_bert = getattr(args, 'use_bert', True)
        self.use_finetune = getattr(args, 'use_finetune', True)
        if self.use_bert:
            self.text_model = BERTEncoder.from_pretrained(args.text_backbone, cache_dir=args.cache_path)

        # Flatten the cmg params into an EasyDict for CrossModalGraph.
        cmg_cfg = EasyDict({
            'bidirectional': args.bidirectional,
            'attn_dropout': args.attn_dropout,
            'attn_dropout_a': args.attn_dropout_a,
            'attn_dropout_v': args.attn_dropout_v,
            'relu_dropout': args.relu_dropout,
            'embed_dropout': args.embed_dropout,
            'res_dropout': args.res_dropout,
            'dst_feature_dim_nheads': args.dst_feature_dim_nheads,
            'nlevels': args.nlevels,
            'conv1d_kernel_size_l': args.conv1d_kernel_size_l,
            'conv1d_kernel_size_a': args.conv1d_kernel_size_a,
            'conv1d_kernel_size_v': args.conv1d_kernel_size_v,
            'text_dropout': args.text_dropout,
            'output_dropout': args.output_dropout,
            'grad_clip': args.grad_clip,
            'attn_mask': args.attn_mask,
        })
        self.text_dropout = cmg_cfg.text_dropout

        feature_dims = getattr(args, 'feature_dims', None) or (args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim)
        orig_d_l, orig_d_a, orig_d_v = feature_dims

        # TEMPORAL CONVOLUTION LAYERS
        proj_dim = cmg_cfg.dst_feature_dim_nheads[0]
        self.proj_l = nn.Conv1d(orig_d_l, proj_dim, kernel_size=cmg_cfg.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(orig_d_a, proj_dim, kernel_size=cmg_cfg.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(orig_d_v, proj_dim, kernel_size=cmg_cfg.conv1d_kernel_size_v, padding=0, bias=False)

        # GRAPH MODAL GRAPH
        self.cross_modal_graph = CrossModalGraph(EasyDict({'cmg_cfg': cmg_cfg}))

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(6 * proj_dim, proj_dim)
        self.post_fusion_layer_2 = nn.Linear(proj_dim, args.num_labels)

    def forward(self, text_feats, video_feats, audio_feats):
        if self.use_bert:
            if self.use_finetune:
                text = self.text_model(text_feats)
            else:
                with torch.no_grad():
                    text = self.text_model(text_feats)
        text_embedding = text

        x_l = F.dropout(text_embedding.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio_feats.transpose(1, 2)
        x_v = video_feats.transpose(1, 2)

        proj_x_l = self.proj_l(x_l).permute(2, 0, 1)
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        proj_x_v = self.proj_v(x_v).permute(2, 0, 1)

        cat_list = [proj_x_l, proj_x_v, proj_x_a]
        cat_seq = torch.cat(cat_list, dim=0)
        cat_split = self.get_seq_split(cat_list)

        fused_output = self.cross_modal_graph(cat_seq=cat_seq, split=cat_split, plot_map=False)
        split_output = fused_output.split

        fusion_h = torch.concat([split.permute(1, 0, 2)[-1] for split in split_output], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)

        logits = self.post_fusion_layer_2(fusion_h)
        return logits

    def get_seq_split(self, seq_list):
        return [seq.shape[0] for seq in seq_list]
