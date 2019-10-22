import configparser
from torchvision import transforms
from torchvision.transforms import Compose


config = configparser.ConfigParser()
config.read('config.ini')

mean = config['data_ingestion'].getfloat('mean')
std = config['data_ingestion'].getfloat('std')
input_transform = Compose([transforms.ToTensor(),
                           transforms.Normalize(mean=[mean], std=[std])])
target_transform = Compose([transforms.ToTensor()])
