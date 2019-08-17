import torch.optim as optim
import torch.nn.functional as F


class Trainer:

    def __init__(self,
                 dataloader,
                 model,
                 device,
                 tensorboard,
                 epoch,
                 num_batches=1000):
        
        self._epoch = epoch
        
        self._tb          = tensorboard 
        self._model       = model
        self._device      = device
        self._dataloader  = dataloader
        self._num_batches = num_batches

        self._model     = self._model.to(self._device).train()
        self._optimizer = optim.Adam(self._model.parameters())


    def train(self):
        avg_loss = 0
        for batch, (input, target) in enumerate(self._dataloader):

            if batch == self._num_batches:
                break

            input  = input.to(self._device)
            target = target.to(self._device)
            
            self._optimizer.zero_grad()
            output = self._model(input)
            loss = F.binary_cross_entropy_with_logits(output,
                                                      target,
                                                      reduction='mean')
            avg_loss += loss.item()
            self._tb.add_scalar('Batch Loss',
                                loss.item(),
                                batch + (self._epoch * self._num_batches))
            
            loss.backward()
            self._optimizer.step()