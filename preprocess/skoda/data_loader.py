import scipy.io as sio
import yaml

from ._data_reader import get_train_val_test
from ._sliding_window import down_sample, segment_data_window


def get_skoda_data():
    data_config_file = open('configs/data.yaml', mode='r')
    data_config = yaml.load(data_config_file, Loader=yaml.FullLoader)

    data_dict = sio.loadmat(file_name=data_config['skoda']['data_file'], squeeze_me=True)
    all_data = data_dict[list(data_dict.keys())[3]]

    x_train, y_train, x_test, y_test, x_validation, y_validation = get_train_val_test(all_data)
    x_train, y_train, x_test, y_test, x_validation, y_validation = down_sample(x_train, y_train, x_test, y_test,
                                                                               x_validation, y_validation, True)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = segment_data_window(x_train, y_train,
                                                                               x_test, y_test,
                                                                               x_validation, y_validation,
                                                                               input_width=data_config['skoda'][
                                                                                   'window_size'],
                                                                               verbose=False,
                                                                               shuffle=True)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)
