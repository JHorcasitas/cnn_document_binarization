import os
import configparser

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter

import config
import models
import data_ingestion
import data_processing
from constants import CONFIG_PATH, MODELS_PATH
from data_ingestion.transform import Normalize, Tensorize


# Tensorboard
runs_dir = '/home/jorge/Documents/Blog/img_bin/runs'
tb = SummaryWriter(log_dir=os.path.join(runs_dir, 'adam-bs1500'))

# Data Ingestion
transform = Compose([Normalize(mean=0.733, std=0.129), Tensorize()])
dataset_factory = data_ingestion.DatasetFactory()
dataset = dataset_factory.get_dataset(kind='train', transform=transform)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=4)

# Load Model
model = models.BinNet()
#tb.add_graph(model)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = data_processing.Trainer(dataloader=dataloader,
                                  model=model,
                                  device=device,
                                  tensorboard=tb)
trainer.train()

# Save Model
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
write_path = os.path.join(MODELS_PATH, config['MODEL SERIALIZATION']['name'])
torch.save(trainer._model.state_dict(), write_path)