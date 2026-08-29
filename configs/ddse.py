class Param():

    def __init__(self, args):

        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        """
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end').
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation.
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
        """
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
        """
        Hyper-parameters taken from the DDSE paper config (mosi section).
            dst_feature_dim_nheads (list): [0] unified projection dim (d), [1] attention heads.
            nlevels (int): Transformer encoder layers.
            conv1d_kernel_size_l/a/v (int): Conv1d kernel width per modality (no padding).
            attn_mask (bool): Whether to apply the causal future mask inside encoders.
        """
        hyper_parameters = {
            'num_train_epochs': 100,
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
