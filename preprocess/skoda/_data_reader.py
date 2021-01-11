import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


def standardize(mat):
    """ standardize each sensor data columnwise"""
    for i in range(mat.shape[1]):
        mean = np.mean(mat[:, [i]])
        std = np.std(mat[:, [i]])
        mat[:, [i]] -= mean
        mat[:, [i]] /= std

    return mat


def normalize(data):
    """ l2 normalization can be used"""

    y = data[:, 0].reshape(-1, 1)
    X = np.delete(data, 0, axis=1)
    transformer = Normalizer(norm='l2', copy=True).fit(X)
    X = transformer.transform(X)

    return np.concatenate((y, X), 1)


def label_count_from_zero(all_data):
    """ start all labels from 0 to total number of activities"""

    labels = {32: 'null class', 48: 'write on notepad', 49: 'open hood', 50: 'close hood',
              51: 'check gaps on the front door', 52: 'open left front door',
              53: 'close left front door', 54: 'close both left door', 55: 'check trunk gaps',
              56: 'open and close trunk', 57: 'check steering wheel'}

    a = np.unique(all_data[:, 0])

    for i in range(len(a)):
        all_data[:, 0][all_data[:, 0] == a[i]] = i
    #         print(i, labels[a[i]])

    return all_data


def split(data):
    """ get 80% train, 10% test and 10% validation data from each activity """

    y = data[:, 0]  # .reshape(-1, 1)
    X = np.delete(data, 0, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

    return X_train, y_train, X_test, y_test, X_val, y_val


def get_train_val_test(data):
    # removing sensor ids
    for i in range(1, 60, 6):
        data = np.delete(data, i, 1)

    # data = data[data[:, 0] != 32]  # remove null class activity

    data = label_count_from_zero(data)
    data = normalize(data)

    activity_id = np.unique(data[:, 0])
    number_of_activity = len(activity_id)

    for i in range(number_of_activity):

        data_for_a_single_activity = data[np.where(data[:, 0] == activity_id[i])]
        trainx, trainy, testx, testy, valx, valy = split(data_for_a_single_activity)

        if i == 0:
            x_train, y_train, x_test, y_test, x_val, y_val = trainx, trainy, testx, testy, valx, valy

        else:
            x_train = np.concatenate((x_train, trainx))
            y_train = np.concatenate((y_train, trainy))

            x_test = np.concatenate((x_test, testx))
            y_test = np.concatenate((y_test, testy))

            x_val = np.concatenate((x_val, valx))
            y_val = np.concatenate((y_val, valy))

    return x_train, y_train, x_test, y_test, x_val, y_val
