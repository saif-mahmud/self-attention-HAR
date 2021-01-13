import argparse
import warnings

import tensorflow as tf
import yaml

from model.har_model import get_model
from preprocess.pamap2.data_loader import get_pamap2_data
from preprocess.skoda.data_loader import get_skoda_data
from preprocess.opp.data_loader import get_opp_data
from preprocess.uschad.data_loader import get_uschad_data

tf.keras.backend.clear_session()
warnings.filterwarnings("ignore")


def get_data(dataset: str):
    if dataset == 'pamap2':
        (train_x, train_y), (val_x, val_y), (test_x, test_y), y_test = get_pamap2_data(input_width=33, n_sensor_val=18,
                                                                                       overlap=.5, print_debug=True)

        return train_x, train_y, val_x, val_y, test_x, test_y

    elif dataset == 'skoda':
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_skoda_data()
        return train_x, train_y, val_x, val_y, test_x, test_y

    elif dataset == 'opp':
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_opp_data()
        return train_x, train_y, val_x, val_y, test_x, test_y

    elif dataset == 'uschad':
        return get_uschad_data()


def train_model(dataset: str, model_config, train_x, train_y, val_x, val_y):
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    model = get_model(n_timesteps, n_features, n_outputs,
                      d_model=model_config[dataset]['d_model'])

    model.compile(**model_config['training'])
    model.summary()

    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, verbose=1, mode='max')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.1,
                                                          patience=4,
                                                          verbose=1,
                                                          min_delta=1e-4,
                                                          mode='min')

    model.fit(train_x, train_y,
              epochs=model_config[dataset]['epochs'],
              batch_size=model_config[dataset]['batch_size'],
              verbose=1,
              validation_data=(val_x, val_y),
              callbacks=[reduce_lr_loss, earlyStopping])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Self Attention Based HAR Model Training')
    parser.add_argument('-d', '--dataset', default='pamap2',
                        type=str, help='Name of Dataset for Model Training')
    args = parser.parse_args()

    model_config_file = open('configs/model.yaml', mode='r')
    model_config = yaml.load(model_config_file, Loader=yaml.FullLoader)

    train_x, train_y, val_x, val_y, test_x, test_y = get_data(
        dataset=args.dataset)
    train_model(dataset=args.dataset, model_config=model_config, train_x=train_x, train_y=train_y, val_x=val_x,
                val_y=val_y)
