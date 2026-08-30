class Param():

    def __init__(self, args):

        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):

        common_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': False,
            'eval_monitor': 'f1',
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        """Smoke config: 2 epochs, paper's original bs8 + accumulation 8."""

        hyper_parameters = {
            'num_train_epochs': 2,
            'use_bert': True,
            'use_finetune': True,
            'dst_feature_dim_nheads': [128, 4],
            'nlevels': 4,
            'bidirectional': False,
            'attn_mask': True,
            'conv1d_kernel_size_l': 5,
            'conv1d_kernel_size_a': 5,
            'conv1d_kernel_size_v': 5,
            'text_dropout': 0.5,
            'attn_dropout': 0.3,
            'attn_dropout_a': 0.2,
            'attn_dropout_v': 0.0,
            'relu_dropout': 0.0,
            'embed_dropout': 0.2,
            'res_dropout': 0.0,
            'output_dropout': 0.5,
            'post_fusion_dropout': 0.0,
            'grad_clip': 0.6,
            'lr_bert': 0.00005,
            'lr_other': 0.0005,
            'weight_decay_bert': 0.001,
            'weight_decay_other': 0.001,
            'update_epochs': 8,
        }
        return hyper_parameters
