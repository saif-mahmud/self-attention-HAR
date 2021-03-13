import sys
import warnings

from preprocess.opp.data_loader import get_opp_data
from preprocess.pamap2.data_loader import get_pamap2_data
from preprocess.skoda.data_loader import get_skoda_data
from preprocess.uschad.data_loader import get_uschad_data

sys.path.append("../")
warnings.filterwarnings("ignore")


def get_data(dataset: str):
    print(f'[Loading {dataset} data]')

    if dataset == 'pamap2':
        (train_x, train_y), (val_x, val_y), (test_x, test_y), y_test = get_pamap2_data()

        return train_x, train_y, val_x, val_y, test_x, test_y

    elif dataset == 'skoda':
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_skoda_data()
        return train_x, train_y, val_x, val_y, test_x, test_y

    elif dataset == 'opp':
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_opp_data()
        return train_x, train_y, val_x, val_y, test_x, test_y

    elif dataset == 'uschad':
        return get_uschad_data()
