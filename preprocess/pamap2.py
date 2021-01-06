import numpy as np
import pandas as pd
import csv
import sys
import os
import h5py
import sqlite3
import copy
from scipy import stats
import time
from sklearn import metrics
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm
import seaborn as sns


class data_reader:
    def __init__(self, train_test_files, use_columns, output_file_name):
        self.data, self.idToLabel = self.readPamap2(train_test_files, use_columns)
        self.save_data(output_file_name)

    def save_data(self, output_file_name):
        f = h5py.File(output_file_name)
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

    def readPamap2(self, train_test_files, use_columns):
        files = train_test_files
        label_map = [
            # (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'Nordic walking'),
            (9, 'watching TV'),
            (10, 'computer work'),
            (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            (18, 'folding laundry'),
            (19, 'house cleaning'),
            (20, 'playing soccer'),
            (24, 'rope jumping')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        # print "label2id=",labelToId
        idToLabel = [x[1] for x in label_map]
        # print "id2label=",idToLabel
        cols = use_columns
        # print "cols",cols
        data = {dataset: self.readPamap2Files(files[dataset], cols, labelToId)
                for dataset in ('train', 'test', 'validation')}
        return data, idToLabel

    def readPamap2Files(self, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open('PAMAP2_Dataset/Protocol/%s' % filename, 'r') as f:
                # print "f",f
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    # print "line=",line
                    elem = []
                    # not including the non related activity
                    if line[1] == "0":
                        continue
                    # if line[10] == "0":
                    #     continue
                    for ind in cols:
                        # print "ind=",ind
                        # if ind == 10:
                        #     # print "line[ind]",line[ind]
                        #     if line[ind] == "0":
                        #         continue
                        elem.append(line[ind])
                    # print "elem =",elem
                    # print "elem[:-1] =",elem[:-1]
                    # print "elem[0] =",elem[0]
                    if sum([x == 'NaN' for x in elem]) < 9:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[0]])
                        # print "[x for x in elem[:-1]]=",[x for x in elem[:-1]]
                        # print "[float(x) / 1000 for x in elem[:-1]]=",[float(x) / 1000 for x in elem[:-1]]
                        # print "labelToId[elem[0]]=",labelToId[elem[0]]
                        # print "labelToId[elem[-1]]",labelToId[elem[-1]]
                        # sys.exit(0)

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int) + 1}


def read_dataset(train_test_files, use_columns, output_file_name):
    print('Reading pamap2')
    dr = data_reader(train_test_files, use_columns, output_file_name)
