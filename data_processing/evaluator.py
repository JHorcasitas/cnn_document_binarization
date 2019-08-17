import torch
from statistics import mean


class Evaluator:
    def __init__(self,
                 dataloader,
                 model,
                 device,
                 tensorboard,
                 epoch,
                 num_batches=1000,
                 specifity=True,
                 NPV=True):
        
        self._epoch = epoch

        self._NPV = NPV
        self._specifity = specifity

        self._num_batches = num_batches
        self._model = model
        self._dataloader = dataloader
        self._device = device
        self._tb = tensorboard

        self._model = self._model.to(self._device).eval()


    def evaluate(self):

        NPV_list = []
        specifity_list = []
        for batch, (input, target) in enumerate(self._dataloader):
            

            if batch == self._num_batches:
                break

            input  = input.to(self._device)
            target = target.to(self._device)

            with torch.no_grad():
                output = self._model(input)
                output = torch.sigmoid(output)
                output = torch.round(output)

            if self._NPV:
                NPV_list.append(self._compute_NPV(output, target))

            if self._specifity:
                specifity = self._compute_specifity(output, target)
                specifity_list.append(specifity)

        self._tb.add_scalar('NPV',
                            mean(NPV_list),
                            self._epoch)
        self._tb.add_scalar('Specifity',
                            mean(specifity_list),
                            self._epoch)
    
    def _compute_NPV(self, output, target):
        output = output.view(-1)
        target = target.view(-1)
        true_negative  = ((output == 0) & (target == 0)).sum().item()
        false_negative = ((output == 0) & (target == 1)).sum().item()
        return true_negative / (true_negative + false_negative) 

    
    def _compute_specifity(self, output, target):
        output = output.view(-1)
        target = target.view(-1)
        true_negative  = ((output == 0) & (target == 0)).sum().item()
        false_positive = ((output == 1) & (target == 0)).sum().item()
        return true_negative / (true_negative + false_positive)