import torch
import torch.nn.functional as F
import logging
from torch import nn, optim
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics

__all__ = ['TCS_Mamba']


class TCS_Mamba:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)

        self.device, self.model = model.device, model.model

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=args.wait_patience)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']

        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def _train(self, args):

        early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()

            # optional gradient accumulation (update_epochs > 1), as in DLF / DDSE
            update_epochs = getattr(args, 'update_epochs', 1)
            left_epochs = update_epochs
            self.optimizer.zero_grad()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    logits = self.model(text_feats, video_feats, audio_feats)
                    aux = self.model.model.aux_outputs

                    # 主任务损失 + 层级辅助分类损失 (TCS_Mamba: gamma_1 = gamma_2 = 0.1)
                    loss_task = self.criterion(logits, label_ids)
                    loss_shared = self.criterion(aux['pred_shared'], label_ids)
                    loss_private = self.criterion(aux['pred_private'], label_ids)
                    loss_cls = loss_task + args.gamma_1 * loss_shared + args.gamma_2 * loss_private

                    # 重构损失 (归一化后 MSE) 与共享/私有正交损失
                    loss_recon = F.mse_loss(
                        F.normalize(aux['X_recon'], dim=-1),
                        F.normalize(aux['X_origin'], dim=-1),
                    )
                    h_shared_mean = aux['H_shared'].view(aux['H_shared'].size(0), 3, -1).mean(dim=1)
                    loss_orth = torch.abs(F.cosine_similarity(h_shared_mean, aux['H_private'], dim=1)).mean()

                    loss = loss_cls + args.lambda_rec * loss_recon + args.lambda_ort * loss_orth

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    left_epochs -= 1
                    if not left_epochs:
                        if args.grad_clip != -1.0:
                            nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        left_epochs = update_epochs

            if left_epochs != update_epochs:
                if args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            outputs = self._get_outputs(args, mode = 'eval')
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4)
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            early_stopping(eval_score, self.model)
            self.scheduler.step(outputs['eval_loss'])

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)

    def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False):

        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

        loss_record = AverageMeter()

        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)

            with torch.set_grad_enabled(False):

                logits = self.model(text_feats, video_feats, audio_feats)

                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))

                loss = self.criterion(logits, label_ids)
                loss_record.update(loss.item(), label_ids.size(0))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'eval_loss': loss_record.avg})

        if return_sample_results:

            outputs.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )

        return outputs

    def _test(self, args):

        test_results = self._get_outputs(args, mode = 'test', return_sample_results=True, show_results = True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)

        return test_results
