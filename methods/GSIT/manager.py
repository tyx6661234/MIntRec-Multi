import torch
import torch.nn.functional as F
import logging
from torch import nn, optim
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics

__all__ = ['GSIT']


class GSIT:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)

        self.device, self.model = model.device, model.model

        # Optimizer groups follow the original GSIT trainer: BERT gets a lower
        # lr (5e-5) with a no-decay split; everything else uses lr_other.
        backbone = self.model.model
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(backbone.text_model.named_parameters()) if hasattr(backbone, 'text_model') else []
        other_params = [p for n, p in backbone.named_parameters() if 'text_model' not in n]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay_bert, 'lr': args.lr_bert},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.lr_bert},
            {'params': other_params, 'weight_decay': args.weight_decay_other, 'lr': args.lr_other},
        ]
        self.optimizer = optim.Adam(optimizer_grouped_parameters)

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

                    loss = self.criterion(logits, label_ids)
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    left_epochs -= 1
                    if not left_epochs:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        left_epochs = update_epochs

            if left_epochs != update_epochs:
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
