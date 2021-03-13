import os
import sys
import warnings

import tensorflow as tf

from model.har_model import create_model

tf.keras.backend.clear_session()
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append("../")


def train_model(dataset: str, model_config, train_x, train_y, val_x, val_y, epochs, save_model=True):
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    model = create_model(n_timesteps, n_features, n_outputs,
                         d_model=model_config[dataset]['d_model'],
                         nh=model_config[dataset]['n_head'],
                         dropout_rate=model_config[dataset]['dropout'])

    model.compile(**model_config['training'])
    model.summary()

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.1,
                                                          patience=4,
                                                          verbose=1,
                                                          min_delta=1e-4,
                                                          mode='min')

    model.fit(train_x, train_y,
              epochs=epochs,
              batch_size=model_config[dataset]['batch_size'],
              verbose=1,
              validation_data=(val_x, val_y),
              callbacks=[reduce_lr_loss, earlyStopping])

    if save_model:
        print(f'Saving trained model for {dataset}')
        model.save(os.path.join(model_config['dirs']['saved_models'], dataset))
