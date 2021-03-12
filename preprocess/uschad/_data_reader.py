import scipy.io
import pandas as pd
import os

from sklearn import metrics


def read_dir(directory):
    subject = []
    act_num = []
    sensor_readings = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if name.endswith('.mat'):
                mat = scipy.io.loadmat(os.path.join(path, name))
                subject.extend(mat['subject'])
                sensor_readings.append(mat['sensor_readings'])

                if mat.get('activity_number') is None:
                    act_num.append('11')
                else:
                    act_num.append(mat['activity_number'])
    return subject, act_num, sensor_readings


def read_uschad(save_csv=False):
    subject, act_num, sensor_readings = read_dir('data/raw/uschad/USC-HAD')

    acc_x = []
    acc_y = []
    acc_z = []
    gyr_x = []
    gyr_y = []
    gyr_z = []

    act_label = []
    subject_id = []

    for i in range(840):
        for j in sensor_readings[i]:
            acc_x.append(j[0])  # acc_x
            acc_y.append(j[1])  # acc_y
            acc_z.append(j[2])  # acc_z
            gyr_x.append(j[3])  # gyr_x
            gyr_y.append(j[4])  # gyr_y
            gyr_z.append(j[5])  # gyr_z
            act_label.append(act_num[i])
            subject_id.append(subject[i])

    df = pd.DataFrame({'subject': subject_id, 'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
                       'gyr_x': gyr_x, 'gyr_y': gyr_y, 'gyr_z': gyr_z, 'activity': act_label})
    df = df[['subject', 'acc_x', 'acc_y', 'acc_z',
             'gyr_x', 'gyr_y', 'gyr_z', 'activity']]

    df['activity'] = df['activity'].astype(int)

    if save_csv:
        df.to_csv('data/processed/usc-had.csv', index=False)
    return df
