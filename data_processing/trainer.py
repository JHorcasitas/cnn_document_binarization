import torch.nn.functional as F


class Trainer:
    """
    """
    def __init__(self,
                 dataloader,
                 model,
                 device,
                 optimizer,
                 tb=None,
                 num_batches=500):
        """
        """ 
        self._dataloader = dataloader
        self._model      = model.to(device)
        self._device     = device
        self._optim      = optimizer
        
        self._tb = tb
        self._num_batches = num_batches
        
        self._epoch = 0

    def train(self):
        """
        """
        self._model.train()
        
        avg_loss = 0
        for batch, (input, target) in enumerate(self._dataloader):

            if batch == self._num_batches:
                break

            input  = input.to(self._device)
            target = target.to(self._device)
            
            self._optim.zero_grad()
            output = self._model(input)
            loss = F.binary_cross_entropy_with_logits(output,
                                                      target,
                                                      reduction='mean')
            avg_loss += loss.item()
            self._tb.add_scalar('Batch Loss',
                                loss.item(),
                                batch + (self._epoch * self._num_batches))
            
            loss.backward()
            self._optim.step()
        
        self._epoch += 1