import h5py
import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd

from ._data_reader import data_reader
from ._sliding_window import *

def get_opp_data():
    cols = [
        38,  39,  40,  41,  42,  43,  44,  45,  46,  51,  52,  53,  54,
        55,  56,  57,  58,  59,  64,  65,  66,  67,  68,  69,  70,  71,
        72,  77,  78,  79,  80,  81,  82,  83,  84,  85,  90,  91,  92,
        93,  94,  95,  96,  97,  98, 103, 104, 105, 106, 107, 108, 109,
       110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
       123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 250]
    cols = np.array(cols)-1

    train_test_split = {
                'train': ['S1-ADL1.dat','S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat', 'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-Drill.dat', 'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-Drill.dat', 'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'],
                'test': ['S2-ADL4.dat','S2-ADL5.dat','S3-ADL4.dat', 'S3-ADL5.dat'],
                'validation':['S1-ADL2.dat']
            }
    if not os.path.exists('data/processed/opportunity.h5'):
        r = data_reader(train_test_split,cols)
    return preprocess(input_width=24, n_sensor_val=len(cols)-1)


def preprocess(input_width=64, n_sensor_val=77, verbose=False):
    path = os.path.join('data/processed/opportunity.h5')
    f = h5py.File(path, 'r')

    x_train = f.get('train').get('inputs')[()]
    y_train = f.get('train').get('targets')[()]

    x_val = f.get('validation').get('inputs')[()]
    y_val = f.get('validation').get('targets')[()]

    x_test = f.get('test').get('inputs')[()]
    y_test = f.get('test').get('targets')[()]


    if verbose:
        print("x_train shape = ", x_train.shape)
        print("y_train shape =",y_train.shape)

        print("x_val shape = ", x_val.shape)
        print("y_val shape =",y_val.shape)

        print("x_test shape =" ,x_test.shape)
        print("y_test shape =",y_test.shape)

    #replace nan with mean
    # x_train = np.where(np.isnan(x_train), np.ma.array(x_train, mask=np.isnan(x_train)).mean(axis=0), x_train)
    # x_val = np.where(np.isnan(x_val), np.ma.array(x_val, mask=np.isnan(x_val)).mean(axis=0), x_val)
    # x_test = np.where(np.isnan(x_test), np.ma.array(x_test, mask=np.isnan(x_test)).mean(axis=0), x_test)

    input_width = input_width
    if verbose:
        print("segmenting signal...")
    train_x, train_y = segment_opp(x_train,y_train,input_width,n_sensor_val)
    val_x, val_y = segment_opp(x_val,y_val,input_width,n_sensor_val)
    test_x, test_y = segment_opp_test(x_test,y_test,input_width,n_sensor_val)
    if verbose:
        print("signal segmented.")

    if verbose:
        print("train_x shape =",train_x.shape)
        print("train_y shape =",train_y.shape)
        print('train_y distribution', np.unique(train_y,return_counts=True))

        print("val_x shape =",val_x.shape)
        print("val_y shape =",val_y.shape)
        print('val_y distribution', np.unique(val_y,return_counts=True))

        print("test_x shape =",test_x.shape)
        print("test_y shape =",test_y.shape)
        print('test_y distribution', np.unique(test_y,return_counts=True))

    n_classes = len(np.unique(train_y))

    train_y = tf.keras.utils.to_categorical(train_y-1, num_classes=n_classes)
    val_y = tf.keras.utils.to_categorical(val_y-1, num_classes=n_classes)
    test_y = tf.keras.utils.to_categorical(test_y-1, num_classes=n_classes)

    if verbose:
        print("unique test_y",np.unique(test_y))
        print("unique train_y",np.unique(train_y))
        print("test_y[1]=",test_y[1])

        print("train_y shape(1-hot) =",train_y.shape)
        print("val_y shape(1-hot) =",val_y.shape)
        print("test_y shape(1-hot) =",test_y.shape)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)