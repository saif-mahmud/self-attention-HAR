import numpy as np
from scipy import stats
from sklearn import metrics


def windowz(data, size, use_overlap=True):
    start = 0
    while start < len(data):
        yield start, start + size
        if use_overlap:
            start += (size // 2)
        else:
            start += size

def segment_opp(x_train,y_train,window_size,n_sensor_val):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,n_sensor_val))
    labels = np.zeros(((len(y_train)//(window_size//2))-1,window_size))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            segments[i_segment] = x_train[start:end]
            labels[i_label] = y_train[start:end]
            i_label+=1
            i_segment+=1
    return segments, labels

def segment_opp_test(x_test,y_test,window_size,n_sensor_val):
    segments = np.zeros(((len(x_test)//(window_size))+1,window_size,n_sensor_val))
    labels = np.zeros(((len(y_test)//(window_size))+1,window_size))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_test,window_size,use_overlap=False):
        if end >= x_test.shape[0]:
            pad_len = window_size - len(x_test[start:end])
            segments[i_segment] = x_test[start-pad_len:end]
            labels[i_label] = y_test[start-pad_len:end]
        else:
            m = stats.mode(y_test[start:end])
            segments[i_segment] = x_test[start:end]
            labels[i_label] = y_test[start:end]
            i_label+=1
            i_segment+=1      

    return segments, labels