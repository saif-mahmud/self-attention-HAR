import numpy as np
import tensorflow as tf
from scipy import stats


def windowz(data, size, use_overlap=True):
    start = 0
    while start < len(data):
        yield start, start + size
        if use_overlap:
            start += (size // 2)
        else:
            start += size


def segment_window(x_train, y_train, window_size, n_sensor_val):
    segments = np.zeros(((len(x_train) // (window_size // 2)) - 1, window_size, n_sensor_val))
    labels = np.zeros(((len(y_train) // (window_size // 2)) - 1))
    i_segment = 0
    i_label = 0
    for (start, end) in windowz(x_train, window_size):
        if (len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1
    return segments, labels


def segment_window_all(x_train, y_train, window_size, n_sensor_val):
    window_segments = np.zeros((len(x_train), window_size, n_sensor_val))
    labels = np.zeros((len(y_train),))

    total_len = len(x_train)

    for i in range(total_len):
        end = i + window_size

        if end > total_len:
            pad_len = end - total_len
            window_segments[i] = x_train[i - pad_len:end]
            labels[i] = y_train[total_len - 1]
        else:
            window_segments[i] = x_train[i:end]
            labels[i] = y_train[end - 1]

    return window_segments, labels


def down_sample(x_train, y_train, x_test, y_test, x_validation, y_validation, verbose=False):
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
    return x_train, y_train, x_test, y_test, x_validation, y_validation


def segment_data_window(x_train, y_train, x_test, y_test, x_validation, y_validation, input_width=20, verbose=False,
                        shuffle=True):
    n_sensor_val = x_train.shape[1]

    # window
    print("segmenting signal...")
    train_x, train_y = segment_window(x_train, y_train, input_width, n_sensor_val)
    val_x, val_y = segment_window(x_validation, y_validation, input_width, n_sensor_val)
    test_x, test_y = segment_window_all(x_test, y_test, input_width, n_sensor_val)
    print("signal segmented.")

    # sample

    # print("segmenting signal...")
    # train_x, train_y = segment_window_sample(x_train,y_train,input_width,n_sensor_val)
    # val_x, val_y = segment_window_sample(x_validation,y_validation,input_width,n_sensor_val)
    # test_x, test_y = segment_window_test_sample(x_test,y_test,input_width,n_sensor_val)
    # print("signal segmented.")

    if verbose:
        print("train_x shape =", train_x.shape)
        print("train_y shape =", train_y.shape)
        print('train_y distribution', np.unique(train_y, return_counts=True))

        print("val_x shape =", val_x.shape)
        print("val_y shape =", val_y.shape)
        print('val_y distribution', np.unique(val_y, return_counts=True))

        print("test_x shape =", test_x.shape)
        print("test_y shape =", test_y.shape)
        print('test_y distribution', np.unique(test_y, return_counts=True))

    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    val_y = tf.keras.utils.to_categorical(val_y)

    if verbose:
        print("unique test_y", np.unique(test_y))
        print("unique train_y", np.unique(train_y))
        print("test_y[1]=", test_y[1])

        print("train_y shape(1-hot) =", train_y.shape)
        print("val_y shape(1-hot) =", val_y.shape)
        print("test_y shape(1-hot) =", test_y.shape)
    if shuffle:
        from sklearn.utils import shuffle
        train_x, train_y = shuffle(train_x, train_y)
        val_x, val_y = shuffle(val_x, val_y)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)
