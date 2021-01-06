import numpy as np
from scipy import stats


def windowz(data, size, use_overlap=True):
    start = 0
    while start < len(data):
        yield start, start + size
        if use_overlap:
            start += (size // 2)
        else:
            start += size


def segment_pa2_test(x_test, y_test, window_size, n_sensor_val):
    segments = np.zeros(((len(x_test) // (window_size)) + 1, window_size, n_sensor_val))
    labels = np.zeros(((len(y_test) // (window_size)) + 1))
    i_segment = 0
    i_label = 0
    for (start, end) in windowz(x_test, window_size, use_overlap=False):
        if end >= x_test.shape[0]:
            pad_len = window_size - len(x_test[start:end])
            segments[i_segment] = x_test[start - pad_len:end]
            m = stats.mode(y_test[start - pad_len:end])
            labels[i_label] = m[0]
        else:
            m = stats.mode(y_test[start:end])
            segments[i_segment] = x_test[start:end]
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1

    return segments, labels


def segment_pa2(x_train, y_train, window_size, n_sensor_val):
    segments = np.zeros(((len(x_train) // (window_size // 2)) - 1, window_size, n_sensor_val))
    labels = np.zeros(((len(y_train) // (window_size // 2)) - 1))
    i_segment = 0
    i_label = 0
    for (start, end) in windowz(x_train, window_size):
        if len(x_train[start:end]) == window_size:
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1
    return segments, labels


def unsegment_pa2_test(y_preds, total_length, window_size):
    unsegmented_preds = np.zeros((total_length,))
    start = 0
    end = window_size
    for element in y_preds:
        if end >= total_length:
            end = total_length
        for i in range(start, end):
            unsegmented_preds[i] = element
        start = end
        end += window_size
        # print(start, end)
    return unsegmented_preds


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
