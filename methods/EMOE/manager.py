import torch
import torch.nn.functional as F
import logging
from torch import nn, optim
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics

__all__ = ['EMOE']


def uni_distill(proj1, proj2):
    # EMOE unimodal distillation: MSE between softmax-normalized projections.
    prob1 = torch.softmax(proj1, dim=-1)
    prob2 = torch.softmax(proj2, dim=-1)
    return torch.mean((prob1 - prob2) ** 2, dim=-1).mean()


def entropy_balance(probs):
    # Negative entropy: minimizing it pushes the router weights toward uniform.
    probs = torch.clamp(probs, min=1e-9)
    n = probs.size(1)
    return torch.mean(n * torch.sum(probs * torch.log(probs), dim=1))


class EMOE:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)

        self.device, self.model = model.device, model.model

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
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

    def _router_supervision(self, res, label_ids):
        # Per-sample classification error of each unimodal head, as the
        # classification analogue of EMOE's squared regression error.
        errors = []
        for key in ['logits_l', 'logits_v', 'logits_a']:
            p_true = torch.softmax(res[key], dim=-1).gather(1, label_ids.view(-1, 1)).squeeze(1)
            errors.append((1.0 - p_true) ** 2)
        l_err, v_err, a_err = errors
        s = 1 / (l_err + 0.1) + 1 / (v_err + 0.1) + 1 / (a_err + 0.1)
        dist = torch.stack([1 / (l_err + 0.1) / s,
                            1 / (v_err + 0.1) / s,
                            1 / (a_err + 0.1) / s], dim=1)
        loss_sim = torch.mean(torch.mean((dist.detach() - res['channel_weight']) ** 2, dim=-1))
        return loss_sim

    def _train(self, args):

        early_stopping = EarlyStopping(args)
        update_epochs = getattr(args, 'update_epochs', 1)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()

            left_epochs = update_epochs
            self.optimizer.zero_grad()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    logits = self.model(text_feats, video_feats, audio_feats)
                    res = self.model.model.aux_outputs
                    w = res['channel_weight']

                    # task losses: fused head + unimodal heads
                    loss_task_m = self.criterion(logits, label_ids)
                    loss_task_l = self.criterion(res['logits_l'], label_ids)
                    loss_task_v = self.criterion(res['logits_v'], label_ids)
                    loss_task_a = self.criterion(res['logits_a'], label_ids)

                    # router supervision: match weights to inverse per-sample error
                    loss_sim = self._router_supervision(res, label_ids)

                    # entropy balance on router weights
                    loss_ety = entropy_balance(w)

                    # unimodal distillation into the fused representation
                    loss_ud = uni_distill(res['c_proj'],
                                          (res['l_proj'] * w[:, 0].view(-1, 1)
                                           + res['v_proj'] * w[:, 1].view(-1, 1)
                                           + res['a_proj'] * w[:, 2].view(-1, 1)).detach())

                    loss = loss_task_m + (loss_task_l + loss_task_v + loss_task_a) / 3 \
                         + 0.1 * (loss_ety + 0.1 * loss_sim) + 0.1 * loss_ud

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

            outputs = self._get_outputs(args, mode='eval')
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

    def _get_outputs(self, args, mode='eval', return_sample_results=False, show_results=False):

        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
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
        _, total_preds = total_probs.max(dim=1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'eval_loss': loss_record.avg})

        if return_sample_results:
            outputs.update({'y_true': y_true, 'y_pred': y_pred})

        return outputs

    def _test(self, args):

        test_results = self._get_outputs(args, mode='test', return_sample_results=True, show_results=True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)

        return test_results
