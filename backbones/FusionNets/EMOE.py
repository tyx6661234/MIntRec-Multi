"""
EMOE backbone (Modality-Specific Enhanced Dynamic Emotion Experts, CVPR 2025).
Adapted from M-SENA: forward follows the MIntRec modality order
(text_feats, video_feats, audio_feats), the text encoder is MIntRec's
BERTEncoder, and all prediction heads output `num_labels` classes.
MIntRec sequences are unaligned (30/230/480), so the router always takes the
transferred (a/v -> text length) branch. Auxiliary tensors for the manager's
losses are stored in `self.aux_outputs`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.transformers_encoder.transformer import TransformerEncoder

__all__ = ['EMOE']


class Router(nn.Module):
    """Per-sample modality weight predictor (EMOE paper, kept as-is)."""

    def __init__(self, dim, channel_num, t):
        super().__init__()
        self.l1 = nn.Linear(dim, int(dim / 8))
        self.l2 = nn.Linear(int(dim / 8), channel_num)
        self.t = t

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l2(F.relu(F.normalize(self.l1(x), p=2, dim=1))) / self.t
        return torch.softmax(x, dim=1)


class EMOE(nn.Module):
    def __init__(self, args):
        super(EMOE, self).__init__()
        self.use_bert = getattr(args, 'use_bert', True)
        self.use_finetune = getattr(args, 'use_finetune', True)
        if self.use_bert:
            self.text_model = BERTEncoder.from_pretrained(args.text_backbone, cache_dir=args.cache_path)
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        # Sequence lengths: MIntRec's DataManager injects per-modality seq lens.
        if hasattr(args, 'text_seq_len'):
            self.len_l, self.len_v, self.len_a = args.text_seq_len, args.video_seq_len, args.audio_seq_len
        elif args.dataset_name == 'mosi':
            self.len_l, self.len_v, self.len_a = (50, 50, 50) if args.need_data_aligned else (50, 500, 375)
        elif args.dataset_name == 'mosei':
            self.len_l, self.len_v, self.len_a = (50, 50, 50) if args.need_data_aligned else (50, 500, 500)
        else:
            raise ValueError(f'Unsupported dataset: {args.dataset_name}')
        feature_dims = getattr(args, 'feature_dims', None) or (args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim)
        self.orig_d_l, self.orig_d_a, self.orig_d_v = feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.fusion_method = args.fusion_method
        output_dim = args.num_labels
        self.args = args

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        self.self_attentions_l = self.get_network(self_type='l')
        self.self_attentions_v = self.get_network(self_type='v')
        self.self_attentions_a = self.get_network(self_type='a')

        self.proj1_l = nn.Linear(self.d_l, self.d_l)
        self.proj2_l = nn.Linear(self.d_l, self.d_l)
        self.out_layer_l = nn.Linear(self.d_l, output_dim)
        self.proj1_v = nn.Linear(self.d_l, self.d_l)
        self.proj2_v = nn.Linear(self.d_l, self.d_l)
        self.out_layer_v = nn.Linear(self.d_l, output_dim)
        self.proj1_a = nn.Linear(self.d_l, self.d_l)
        self.proj2_a = nn.Linear(self.d_l, self.d_l)
        self.out_layer_a = nn.Linear(self.d_l, output_dim)

        if self.fusion_method == "sum":
            self.proj1_c = nn.Linear(self.d_l, self.d_l)
            self.proj2_c = nn.Linear(self.d_l, self.d_l)
            self.out_layer_c = nn.Linear(self.d_l, output_dim)
        elif self.fusion_method == "concat":
            self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
            self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
            self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)
        else:
            raise ValueError(f'Unknown fusion method: {self.fusion_method}')

        # Router sees BERT text + a/v transferred to the text length, flattened.
        router_dim = (self.orig_d_l + self.orig_d_v + self.orig_d_a) * self.len_l
        self.Router = Router(router_dim, 3, self.args.temperature)
        self.transfer_a_ali = nn.Linear(self.len_a, self.len_l)
        self.transfer_v_ali = nn.Linear(self.len_v, self.len_l)

    def get_network(self, self_type='l', layers=-1):
        if self_type == 'l':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type == 'v':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text_feats, video_feats, audio_feats):
        if self.use_bert:
            if self.use_finetune:
                text = self.text_model(text_feats)
            else:
                with torch.no_grad():
                    text = self.text_model(text_feats)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio_feats.transpose(1, 2)
        x_v = video_feats.transpose(1, 2)

        # MIntRec is unaligned: bring a/v to the text length for the router.
        audio_ = self.transfer_a_ali(audio_feats.permute(0, 2, 1)).permute(0, 2, 1)
        video_ = self.transfer_v_ali(video_feats.permute(0, 2, 1)).permute(0, 2, 1)
        m_i = torch.cat((text, video_, audio_), dim=2)
        m_w = self.Router(m_i)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        c_l_att = self.self_attentions_l(c_l)
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]
        c_v_att = self.self_attentions_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]
        c_a_att = self.self_attentions_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        l_proj = self.proj2_l(
            F.dropout(F.relu(self.proj1_l(c_l_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        l_proj += c_l_att
        logits_l = self.out_layer_l(l_proj)
        v_proj = self.proj2_v(
            F.dropout(F.relu(self.proj1_v(c_v_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        v_proj += c_v_att
        logits_v = self.out_layer_v(v_proj)
        a_proj = self.proj2_a(
            F.dropout(F.relu(self.proj1_a(c_a_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        a_proj += c_a_att
        logits_a = self.out_layer_a(a_proj)

        if self.fusion_method == "sum":
            c_fusion = c_l_att * m_w[:, 0].view(-1, 1) + c_v_att * m_w[:, 1].view(-1, 1) + c_a_att * m_w[:, 2].view(-1, 1)
        elif self.fusion_method == "concat":
            c_fusion = torch.cat([c_l_att * m_w[:, 0].view(-1, 1),
                                  c_v_att * m_w[:, 1].view(-1, 1),
                                  c_a_att * m_w[:, 2].view(-1, 1)], dim=1) * 3

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)

        self.aux_outputs = {
            'logits_l': logits_l,
            'logits_v': logits_v,
            'logits_a': logits_a,
            'channel_weight': m_w,
            'c_proj': c_proj,
            'l_proj': l_proj,
            'v_proj': v_proj,
            'a_proj': a_proj,
        }
        return logits_c
