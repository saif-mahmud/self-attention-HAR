import argparse
import warnings

import tensorflow as tf
import yaml

from model.har_model import get_model
from preprocess.pamap2.data_loader import get_pamap2_data

tf.keras.backend.clear_session()
warnings.filterwarnings("ignore")


def get_data(dataset: str):
    if dataset == 'pamap2':
        (train_x, train_y), (val_x, val_y), (test_x, test_y), y_test = get_pamap2_data(input_width=33, n_sensor_val=18,
                                                                                       overlap=.5, print_debug=True)

        return train_x, train_y, val_x, val_y, test_x, test_y


def train_model(dataset: str, exp_config, train_x, train_y, val_x, val_y):
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    model = get_model(n_timesteps, n_features, n_outputs, d_model=exp_config[dataset]['d_model'])

    model.compile(**exp_config['training'])
    model.summary()

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.1,
                                                          patience=4,
                                                          verbose=1,
                                                          min_delta=1e-4,
                                                          mode='min')

    model.fit(train_x, train_y,
              epochs=exp_config[dataset]['epochs'],
              batch_size=exp_config[dataset]['batch_size'],
              verbose=1,
              validation_data=(val_x, val_y),
              callbacks=[reduce_lr_loss, earlyStopping])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Attention Based HAR Model Training')
    parser.add_argument('-d', '--dataset', default='pamap2', type=str, help='Name of Dataset for Model Training')
    args = parser.parse_args()

    exp_config_file = open('configs/experiment.yaml', mode='r')
    exp_config = yaml.load(exp_config_file, Loader=yaml.FullLoader)

    train_x, train_y, val_x, val_y, test_x, test_y = get_data(dataset=args.dataset)
    train_model(dataset=args.dataset, exp_config=exp_config, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
