import torch
import torch.nn.functional as F
import logging
from torch import nn, optim
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics
from utils.hinge_loss import HingeLoss

__all__ = ['DLF']


class DLF:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)

        self.device, self.model = model.device, model.model

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=args.wait_patience)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']

        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = Metrics(args)
        self.sim_loss = HingeLoss()

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def _train(self, args):

        early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    logits = self.model(text_feats, video_feats, audio_feats)
                    res = self.model.model.aux_outputs

                    # task loss (language-focused head weighted 3x, as in the paper)
                    loss_task_all = self.criterion(logits, label_ids)
                    loss_task_l_hetero = self.criterion(res['logits_l_hetero'], label_ids)
                    loss_task_v_hetero = self.criterion(res['logits_v_hetero'], label_ids)
                    loss_task_a_hetero = self.criterion(res['logits_a_hetero'], label_ids)
                    loss_task_c = self.criterion(res['logits_c'], label_ids)
                    loss_task = 1 * (1 * loss_task_all + 1 * loss_task_c + 3 * loss_task_l_hetero
                                     + 1 * loss_task_v_hetero + 1 * loss_task_a_hetero)

                    # reconstruction loss L_r
                    loss_recon = F.mse_loss(res['recon_l'], res['origin_l']) \
                               + F.mse_loss(res['recon_v'], res['origin_v']) \
                               + F.mse_loss(res['recon_a'], res['origin_a'])

                    # specific-feature consistency loss L_s
                    loss_s_sr = F.mse_loss(res['s_l'].permute(1, 2, 0), res['s_l_r']) \
                              + F.mse_loss(res['s_v'].permute(1, 2, 0), res['s_v_r']) \
                              + F.mse_loss(res['s_a'].permute(1, 2, 0), res['s_a_r'])

                    # orthogonality loss L_o (reshape(-1, d) always divides)
                    num = self.model.model.d_l
                    ort_target = torch.tensor([-1.0], device=self.device)
                    loss_ort = self.cosine(res['s_l'].reshape(-1, num), res['c_l'].reshape(-1, num), ort_target) \
                             + self.cosine(res['s_v'].reshape(-1, num), res['c_v'].reshape(-1, num), ort_target) \
                             + self.cosine(res['s_a'].reshape(-1, num), res['c_a'].reshape(-1, num), ort_target)

                    # triplet margin loss L_m
                    c_l, c_v, c_a = res['c_l_sim'], res['c_v_sim'], res['c_a_sim']
                    ids, feats = [], []
                    for i in range(label_ids.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(label_ids[i].view(1, -1))
                        ids.append(label_ids[i].view(1, -1))
                        ids.append(label_ids[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    # overall loss L_DLF
                    loss = loss_task + (loss_s_sr + loss_recon + (loss_sim + loss_ort) * 0.1) * 0.1

                    self.optimizer.zero_grad()
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.step()

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
