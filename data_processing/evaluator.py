import torch
from statistics import mean


class Evaluator:
    """
    """
    def __init__(self,
                 dataloader,
                 model,
                 device,
                 tb,
                 num_batches=500):
        """
        """
        self._dataloader = dataloader
        self._device = device
        self._model = model.to(device)

        self._tb = tb
        self._num_batches = num_batches

        self._epoch = 0

    def evaluate(self):
        """
        """
        self._model.eval()

        precision_list = []
        recall_list = []
        for batch, (input, target) in enumerate(self._dataloader):
            
            if batch == self._num_batches:
                break

            input  = input.to(self._device)
            target = target.to(self._device)

            with torch.no_grad():
                output = self._model(input)
                output = torch.sigmoid(output)
                output = torch.round(output)

            precision_list.append(self._compute_precision(output, target))
            recall_list.append(self._compute_recall(output, target))
        
        precision = mean(precision_list)
        self._tb.add_scalar('Precision',
                            precision,
                            self._epoch)
        recall = mean(recall_list)
        self._tb.add_scalar('Recall',
                            recall,
                            self._epoch)

        f1 = self._compute_f1_score(precision, recall)
        self._tb.add_scalar('F1 Score',
                            f1,
                            self._epoch)
        
        self._epoch += 1

    def _compute_precision(self, output, target):
        output = output.view(-1)
        target = target.view(-1)
        true_negative  = ((output == 0) & (target == 0)).sum().item()
        false_negative = ((output == 0) & (target == 1)).sum().item()
        if (true_negative + false_negative) == 0:
            return 0
        return true_negative / (true_negative + false_negative) 

    def _compute_recall(self, output, target):
        output = output.view(-1)
        target = target.view(-1)
        true_negative  = ((output == 0) & (target == 0)).sum().item()
        false_positive = ((output == 1) & (target == 0)).sum().item()
        if (true_negative + false_positive) == 0:
            return 0
        return true_negative / (true_negative + false_positive)

    def _compute_f1_score(self, precision, recall):
        if (precision + recall) == 0:
            return 0
        return 2 * ((precision * recall) / (precision + recall))