from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import logging

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Metrics(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """
    def __init__(self, args):

        self.logger = logging.getLogger(args.logger_name)
        self.eval_metrics = ['acc', 'f1', 'prec', 'rec', 'f1_w', 'prec_w', 'rec_w']

    def __call__(self, y_true, y_pred, show_results = False):

        acc_score = self._acc_score(y_true, y_pred)
        macro_f1 = self._f1_score(y_true, y_pred)
        macro_prec = self._precision_score(y_true, y_pred)
        macro_rec = self._recall_score(y_true, y_pred)
        weighted_f1 = self._f1_score_w(y_true, y_pred)
        weighted_prec = self._precision_score_w(y_true, y_pred)
        weighted_rec = self._recall_score_w(y_true, y_pred)

        eval_results = {
            'acc': acc_score,
            'f1': macro_f1,
            'prec': macro_prec,
            'rec': macro_rec,
            'f1_w': weighted_f1,
            'prec_w': weighted_prec,
            'rec_w': weighted_rec,
        }

        if show_results:

            self._show_confusion_matrix(y_true, y_pred)

            self.logger.info("***** Evaluation results *****")
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(round(eval_results[key], 4)))

        return eval_results

    def _acc_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def _f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')

    def _precision_score(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro')

    def _recall_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average='macro')

    def _f1_score_w(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

    def _precision_score_w(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='weighted')

    def _recall_score_w(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average='weighted')

    def _show_confusion_matrix(self, y_true, y_pred):

        cm = confusion_matrix(y_true, y_pred)
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))