import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def generate_result(dataset, ground_truth, prediction):
    activity_map = json.load(open(os.path.join('configs', 'activity_maps', dataset + '.json')))
    activity_names = list(activity_map.values())

    print('\n[CLASSIFICATION REPORT]')
    print(classification_report(np.argmax(ground_truth, axis=1), np.argmax(prediction, axis=1),
                                labels=range(len(activity_names)), target_names=activity_names, zero_division=1))

    confm = confusion_matrix(np.argmax(ground_truth, axis=1), np.argmax(prediction, axis=1),
                             labels=range(len(activity_names)), normalize='true')

    df_cm = pd.DataFrame(confm, index=activity_names, columns=activity_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, annot=True, fmt='.3f', cmap="YlGnBu")
    out_fig = dataset + '_confusion_matrix.png'
    plt.savefig(os.path.join('results', out_fig))

    print(f'\nConfusion matrix plot generated for {dataset}: Check "./results" direcotry')
