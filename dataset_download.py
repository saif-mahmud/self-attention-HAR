import argparse
import datetime
import os
import time
import zipfile
from tqdm import tqdm

import requests
import yaml


def get_dataset(url: str, data_directory: str, file_name: str, unzip: bool):
    if not os.path.exists('data/'):
        os.mkdir('data/')

    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    if not os.path.exists(os.path.join(data_directory, file_name)):
        print(f'GETTING DATASET [{file_name}] ...')

        response = requests.get(url, stream=True)
        data_file = open(os.path.join(data_directory, file_name), 'wb')

        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024

        for chunk in tqdm(iterable=response.iter_content(chunk_size=chunk_size), total=total_size / chunk_size,
                          unit='B', unit_scale=True, unit_divisor=chunk_size):
            data_file.write(chunk)

        data_file.close()

        if unzip:
            print(f'Unzipping [{file_name}] ...')
            with zipfile.ZipFile(os.path.join(data_directory, file_name), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(data_directory, file_name.split('.')[0]))
                os.remove(os.path.join(data_directory, file_name))

        print('\n---DATASET DOWNLOAD COMPLETE---')

    else:
        print(f'Requested dataset exists in {data_directory}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self Attention Based HAR Model Training')
    parser.add_argument('-d', '--dataset', default='pamap2', type=str, help='Name of Dataset for Model Training')
    parser.add_argument('-z', '--unzip', action='store_true', help='Unzip downloaded dataset')
    args = parser.parse_args()

    config_file = open('configs/data.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    get_dataset(url=config[args.dataset]['source'], data_directory=config['data_dir']['raw'],
                file_name=config[args.dataset]['destination'], unzip=args.unzip)
