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
            'train_batch_size': 32,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        # Smoke-test config: identical to ddse.py but with 2 epochs.
        hyper_parameters = {
            'num_train_epochs': 2,
            'use_bert': True,
            'use_finetune': True,
            'dst_feature_dim_nheads': [50, 10],
            'nlevels': 4,
            'attn_dropout': 0.3,
            'attn_dropout_a': 0.2,
            'attn_dropout_v': 0.0,
            'relu_dropout': 0.0,
            'embed_dropout': 0.2,
            'res_dropout': 0.0,
            'output_dropout': 0.5,
            'text_dropout': 0.1,
            'attn_mask': True,
            'conv1d_kernel_size_l': 5,
            'conv1d_kernel_size_a': 5,
            'conv1d_kernel_size_v': 5,
            'lr': 0.0001,
            'grad_clip': 0.6,
        }
        return hyper_parameters
