import os

import h5py
import numpy as np
import tensorflow as tf

from .sliding_window import segment_pa2, segment_window_all


def preprocess(input_width=33, n_sensor_val=18, overlap=.5, print_debug=True):
    path = os.path.join('pamap2_106.h5')
    f = h5py.File(path, 'r')

    x_train = f.get('train').get('inputs')[()]
    y_train = f.get('train').get('targets')[()]

    x_val = f.get('validation').get('inputs')[()]
    y_val = f.get('validation').get('targets')[()]

    x_test = f.get('test').get('inputs')[()]
    y_test = f.get('test').get('targets')[()]

    if print_debug:
        print("x_train shape = ", x_train.shape)
        print("y_train shape =", y_train.shape)

        print("x_val shape = ", x_val.shape)
        print("y_val shape =", y_val.shape)

        print("x_test shape =", x_test.shape)
        print("y_test shape =", y_test.shape)

    x_train = x_train[::3, :]
    y_train = y_train[::3]
    x_val = x_val[::3, :]
    y_val = y_val[::3]
    x_test = x_test[::3, :]
    y_test = y_test[::3]

    if print_debug:
        print("x_train shape(downsampled) = ", x_train.shape)
        print("y_train shape(downsampled) =", y_train.shape)
        print("x_val shape(downsampled) = ", x_val.shape)
        print("y_val shape(downsampled) =", y_val.shape)
        print("x_test shape(downsampled) =", x_test.shape)
        print("y_test shape(downsampled) =", y_test.shape)

    # replace nan with mean
    x_train = np.where(np.isnan(x_train), np.ma.array(x_train, mask=np.isnan(x_train)).mean(axis=0), x_train)
    x_val = np.where(np.isnan(x_val), np.ma.array(x_val, mask=np.isnan(x_val)).mean(axis=0), x_val)
    x_test = np.where(np.isnan(x_test), np.ma.array(x_test, mask=np.isnan(x_test)).mean(axis=0), x_test)

    n_sensor_val = len(use_columns) - 1
    input_width = input_width
    print("segmenting signal...")
    train_x, train_y = segment_pa2(x_train, y_train, input_width, n_sensor_val)
    val_x, val_y = segment_pa2(x_val, y_val, input_width, n_sensor_val)
    test_x, test_y = segment_window_all(x_test, y_test, input_width, n_sensor_val)
    print("signal segmented.")

    if print_debug:
        print("train_x shape =", train_x.shape)
        print("train_y shape =", train_y.shape)
        print('train_y distribution', np.unique(train_y, return_counts=True))

        print("val_x shape =", val_x.shape)
        print("val_y shape =", val_y.shape)
        print('val_y distribution', np.unique(val_y, return_counts=True))

        print("test_x shape =", test_x.shape)
        print("test_y shape =", test_y.shape)
        print('test_y distribution', np.unique(test_y, return_counts=True))

    train_y = tf.keras.utils.to_categorical(train_y, num_classes=19)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=19)
    val_y = tf.keras.utils.to_categorical(val_y, num_classes=19)

    if print_debug:
        print("unique test_y", np.unique(test_y))
        print("unique train_y", np.unique(train_y))
        print("test_y[1]=", test_y[1])

        print("train_y shape(1-hot) =", train_y.shape)
        print("val_y shape(1-hot) =", val_y.shape)
        print("test_y shape(1-hot) =", test_y.shape)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y), y_test
