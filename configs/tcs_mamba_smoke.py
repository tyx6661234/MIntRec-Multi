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
        # Smoke-test config: identical to tcs_mamba.py but with 2 epochs.
        hyper_parameters = {
            'num_train_epochs': 2,
            'use_tucker': True,
            'use_cp': True,
            'mamba_type': 'gated',
            'use_hlbf': True,
            'modalities': 'TAV',
            'use_finetune': True,
            'dst_feature_dim_nheads': [128, 1],
            'kernel_size_l': 3,
            'kernel_size_a': 3,
            'kernel_size_v': 3,
            'd_state': 8,
            'text_dropout': 0.3,
            'lambda_rec': 0.1,
            'lambda_ort': 0.05,
            'gamma_1': 0.1,
            'gamma_2': 0.1,
            'lr': 0.00002,
            'grad_clip': 0.8,
        }
        return hyper_parameters
