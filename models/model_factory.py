import os
from typing import Union, Optional

import torch

from models.model_definition.sw_network import SlideNet
from models.model_definition.sk_network import StridedNet


Model = Union[SlideNet, StridedNet]


def get_model(kind: str, model_name: Optional[str] = None) -> Model:
    """
    Retrieves specified model loaded in the CPU
    Args:
        kind: 'sw'(sliding windows) or 'sk' (strided kernel)
        cache: Wether to use a pretrained model
    """
    if kind == 'sw':
        return load_sw_model(model_name=model_name)
    elif kind == 'sk':
        # TODO: refactor this part
        if model_name:
            return load_strided_weights()
        else:
            return StridedNet()
    else:
        raise ValueError(f'Invalid kind value: {kind}')


def load_sw_model(model_name: Optional[str] = None) -> SlideNet:
    model = SlideNet()
    if model_name:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(file_dir, 'pretrained', model_name)
        model.load_state_dict(torch.load(model_path,
                                         map_location=torch.device('CPU')))
    return model


def load_strided_weights():
    sk_model = get_model(kind='sk', device=torch.device('cpu'), cache=False)
    sw_model = get_model(kind='sw', device=torch.device('cpu'), cache=True)
    sk_model = transfer_parameters(sw_model, sk_model)
    return sk_model


def transfer_parameters(src_model, dst_model):
    """ Transfer parameter from src_model to dst_model """
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
    dst_model.conv1.bias = src_model.conv1.bias

    dst_model.conv2.weight = src_model.conv2.weight
    dst_model.conv2.bias = src_model.conv2.bias

    dst_model.fc1.weight = process_fc_weights(src_model.fc1.weight,
                                              dst_model.fc1.weight.shape)
    dst_model.fc1.bias = src_model.fc1.bias

    dst_model.fc2.weight = process_fc_weights(src_model.fc2.weight,
                                              dst_model.fc2.weight.shape)
    dst_model.fc2.bias = src_model.fc2.bias

    dst_model.fc3.weight = process_fc_weights(src_model.fc3.weight,
                                              dst_model.fc3.weight.shape)
    dst_model.fc3.bias = src_model.fc3.bias
    return dst_model
