import csv

import h5py
import numpy as np
from tqdm import tqdm


class data_reader:
    def __init__(self, train_test_split, cols):
        self.data, self.idToLabel = self.readOpportunity(
            train_test_split, cols)
        self.save_data()

    def save_data(self):
        f = h5py.File('data/processed/opportunity.h5')
        for key in self.data:
            f.create_group(key)
            for field in self.data[key]:
                f[key].create_dataset(field, data=self.data[key][field])
        f.close()
        print('Done.')

    @property
    def train(self):
        return self.data['train']

    @property
    def test(self):
        return self.data['test']

    @property
    def validation(self):
        return self.data['validation']

    def readOpportunity(self, train_test_split, cols):
        files = train_test_split
        # names are from label_legend.txt of Opportunity dataset
        # except 0-ie Other, which is an additional label
        label_map = [
            (0, 'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        cols = cols

        data = {dataset: self.readOpportunityFiles(files[dataset], cols, labelToId)
                for dataset in ('train', 'test', 'validation')}

        return data, idToLabel

    def readOpportunityFiles(self, filelist, cols, labelToId):
        data = []
        labels = []
        print('[READING OPPORTUNITY DATASET FILES]')
        for filename in tqdm(filelist, total=len(filelist)):
            with open('data/raw/opp/OpportunityUCIDataset/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) < 5:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int) + 1}
