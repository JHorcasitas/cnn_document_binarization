import torch.optim as optim
import torch.nn.functional as F


class Trainer:

    def __init__(self, dataloader, model, device, tensorboard):
        self._tb         = tensorboard 
        self._model      = model
        self._device     = device
        self._dataloader = dataloader
    
        self._model     = self._model.to(self._device).train()
        self._optimizer = optim.Adam(self._model.parameters())

    def train(self):
        avg_loss = 0
        for batch, sample in enumerate(self._dataloader):
            if batch > 500:
                break
            input  = sample['image'].to(self._device)
            target = sample['target'].to(self._device)
            
            self._optimizer.zero_grad()
            output = self._model(input)
            loss = F.binary_cross_entropy_with_logits(output,
                                                      target,
                                                      reduction='mean')
            self._tb.add_scalar('Batch Loss', loss.item(), batch)
            loss.backward()
            self._optimizer.step()
            avg_loss += loss.item()