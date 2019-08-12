import os
import configparser
from definitions import CONFIG_PATH


def read_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    radius    = config['DATA INGESTION'].getint('radius')
    data_path = config['DATA INGESTION']['data_path']

    validate_radius(radius)
    validate_data_path(data_path)

    return {'radius':radius, 'data_path':data_path}

def validate_radius(radius):
    if radius % 2 == 0:
        raise IOError('Configuration variable radius must be an odd number')

def validate_data_path(path):
    if os.path.isdir(path) == False:
        raise IOError('Configuration variable data_path must be a directory')