import configparser


def create_configuration():
    config = configparser.ConfigParser()
    config['DATA INGESTION'] = {}
    config['DATA INGESTION']['radius']   = '19'
    config['DATA INGESTION']['data_path'] = '../../data'

    with open('./config/config.ini', 'w') as configfile:
        config.write(configfile)

if __name__ == '__main__':
    create_configuration()