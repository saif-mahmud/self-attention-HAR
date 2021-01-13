import datetime
import os
import sys
import time
import zipfile

import requests
import yaml


def get_dataset(url: str, data_directory: str, file_name: str, unzip: bool):
    if not os.path.exists('data/'):
        os.mkdir('data/')

    print(datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    if not os.path.exists(os.path.join(data_directory, file_name)):
        print(f'GETTING DATASET [{file_name}] ...')

        response = requests.get(url, stream=True)
        data_file = open(os.path.join(data_directory, file_name), 'wb')

        for chunk in response.iter_content(chunk_size=1024):
            data_file.write(chunk)

        data_file.close()

        if unzip:
            print(f'Unzipping [{file_name}] ...')
            with zipfile.ZipFile(os.path.join(data_directory, file_name), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(
                    data_directory, file_name.split('.')[0]))
                os.remove(os.path.join(data_directory, file_name))

        print('\n---DATASET DOWNLOAD COMPLETE---')

    else:
        print(f'Requested dataset exists in {data_directory}')


if __name__ == "__main__":
    config_file = open('configs/data.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    dataset = str(sys.argv[1])

    get_dataset(url=config[dataset]['source'], data_directory=config['data_dir']['raw'],
                file_name=config[dataset]['destination'], unzip=True)
