import yaml
import pandas as pd

from ._data_reader import read_uschad
from ._sliding_window import sliding_window


def get_uschad_data(downsample=True, verbose=False):
    config_file = open('configs/data.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)['uschad']
    window_size = config['window_size']

    df = read_uschad()
    df = df.apply(pd.to_numeric)

    train_df = df.loc[df['subject'].isin(config['train_subject'])]
    x_train = train_df[config['feature_column']].values
    y_train = train_df[config['label_column']].values

    validation_df = df.loc[df['subject'].isin(config['validation_subject'])]
    x_validation = validation_df[config['feature_column']].values
    y_validation = validation_df[config['label_column']].values

    test_df = df.loc[df['subject'].isin(config['test_subject'])]
    x_test = test_df[config['feature_column']].values
    y_test = test_df[config['label_column']].values

    if downsample:
        x_train = x_train[::3, :]
        y_train = y_train[::3]
        x_validation = x_validation[::3, :]
        y_validation = y_validation[::3]
        x_test = x_test[::3, :]
        y_test = y_test[::3]

        if verbose:
            print("x_train shape(downsampled) = ", x_train.shape)
            print("y_train shape(downsampled) =", y_train.shape)
            print("x_val shape(downsampled) = ", x_validation.shape)
            print("y_val shape(downsampled) =", y_validation.shape)
            print("x_test shape(downsampled) =", x_test.shape)
            print("y_test shape(downsampled) =", y_test.shape)

    train_x, train_y, val_x, val_y, test_x, test_y = sliding_window(
        x_train, y_train, x_validation, y_validation, x_test, y_test, window_size,
        n_sensor_val=len(config['feature_column']))

    return train_x, train_y, val_x, val_y, test_x, test_y
