import os
import configparser

import torch

import models
from definitions import MODELS_PATH, CONFIG_PATH


config = configparser.ConfigParser()
config.read(CONFIG_PATH)


def get_model(kind, device, cache=False):
    if kind == 'sw':
        if cache:
            return load_window_weights()
        else:
            return models.SlideNet()
        return models.SlideNet()
    elif kind == 'sk':
        if cache:
            return load_strided_weights()
        else:
            return models.StridedNet() 
    else:
        raise ValueError(f'Invalid kind value: \'kind\'')


def load_window_weights():
    model = get_model(kind='sw', device=torch.device('cpu'), cache=False)
    path  = os.path.join(MODELS_PATH, config['MODEL SERIALIZATION']['name'])
    model.load_state_dict(torch.load(path))
    return model


def load_strided_weights():
    sk_model = get_model(kind='sk', device=torch.device('cpu'), cache=False)
    sw_model = get_model(kind='sw', device=torch.device('cpu'), cache=True)
    sk_model = transfer_parameters(sw_model, sk_model)
    return sk_model


def transfer_parameters(src_model, dst_model):
    """
    Transfer parameter from src_model to dst_model
    """
    def process_fc_weights(array, shape):
        flat = array.view(-1, array.shape[0] * array.shape[1])
        
        m = 0
        out = torch.empty(shape).float()
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        out[i][j][k][l] = flat[0][m]
                        m += 1
        return torch.nn.Parameter(out)
    
    dst_model.conv1.weight = src_model.conv1.weight
    dst_model.conv1.bias   = src_model.conv1.bias
    
    dst_model.conv2.weight = src_model.conv2.weight
    dst_model.conv2.bias   = src_model.conv2.bias
    
    dst_model.fc1.weight   = process_fc_weights(src_model.fc1.weight,
                                                dst_model.fc1.weight.shape)
    dst_model.fc1.bias     = src_model.fc1.bias
    
    dst_model.fc2.weight   = process_fc_weights(src_model.fc2.weight,
                                                dst_model.fc2.weight.shape)
    dst_model.fc2.bias     = src_model.fc2.bias
    
    dst_model.fc3.weight   = process_fc_weights(src_model.fc3.weight,
                                                dst_model.fc3.weight.shape)
    dst_model.fc3.bias     = src_model.fc3.bias
    return dst_model