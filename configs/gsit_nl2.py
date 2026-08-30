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
            'train_batch_size': 64,
            'eval_batch_size': 64,
            'test_batch_size': 64,
            'wait_patience': 8
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        """
        Hyper-parameters taken from the GSIT paper config (mosi section).
            dst_feature_dim_nheads (list): [0] graph node dim (d), [1] attention heads.
            nlevels (int): Graph transformer encoder layers.
            conv1d_kernel_size_l/a/v (int): Conv1d kernel width per modality (no padding).
            bidirectional (bool): False -> forward + backward cross encoders + self encoder.
            lr_bert / lr_other (float): Per-group learning rates (BERT vs the rest).
            weight_decay_bert / weight_decay_other (float): Per-group weight decay.
            update_epochs (int): Gradient accumulation window; 1 at bs64 equals the
                                 paper's effective batch (bs8 x 8).
            post_fusion_dropout (float): Dropout before the post-fusion MLP.
            use_bert (bool): Whether to use the BERT text encoder.
            use_finetune (bool): Whether to fine-tune the BERT text encoder.
        """
        hyper_parameters = {
            'num_train_epochs': 100,
            'use_bert': True,
            'use_finetune': True,
            'dst_feature_dim_nheads': [128, 4],
            'nlevels': 2,
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
            'update_epochs': 1,
        }
        return hyper_parameters
