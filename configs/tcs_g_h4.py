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
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            use_tucker (bool): Whether to use the Tucker decomposition for the shared subspace.
            use_cp (bool): Whether to use the CP decomposition for the private subspace.
            mamba_type (str): 'gated' (GM-Mamba), 'vanilla' (plain CoSSM) or 'none' (no Mamba refinement).
            use_hlbf (bool): Whether to use the hierarchical low-rank bilinear fusion.
            modalities (str): The combination of modalities, e.g., 'TAV'.
            use_finetune (bool): Whether to fine-tune the BERT text encoder.
            dst_feature_dim_nheads (list): [0] is the unified projection dimension (d_model).
            kernel_size_l/a/v (int): The Conv1d kernel width for the three modalities.
            d_state (int): The SSM state dimension of Mamba.
            text_dropout (float): The dropout used inside CoSSM.
            lambda_rec (float): The weight of the reconstruction loss.
            lambda_ort (float): The weight of the orthogonality loss.
            gamma_1 (float): The weight of the shared-level auxiliary classification loss.
            gamma_2 (float): The weight of the private-level auxiliary classification loss.
            lr (float): The learning rate.
            grad_clip (float): The gradient clip value (-1.0 to disable).
        """
        hyper_parameters = {
            'num_train_epochs': 100,
            'use_tucker': True,
            'use_cp': True,
            'mamba_type': 'gated',
            'use_hlbf': True,
            'modalities': 'TAV',
            'use_finetune': True,
            'dst_feature_dim_nheads': [128, 4],
            'kernel_size_l': 3,
            'kernel_size_a': 3,
            'kernel_size_v': 3,
            'd_state': 8,
            'text_dropout': 0.3,
            'lambda_rec': 0.1,
            'lambda_ort': 0.05,
            'gamma_1': 0.1,
            'gamma_2': 0.1,
            'lr': 0.00003,
            'update_epochs': 8,
            'grad_clip': 0.8,
        }
        return hyper_parameters
