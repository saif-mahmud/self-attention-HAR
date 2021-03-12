import os
import sys
import warnings

import tensorflow as tf

sys.path.append("../")
warnings.filterwarnings("ignore")


def test_model(dataset: str, model_config, test_x):
    if os.path.exists(os.path.join(model_config['dirs']['saved_models'], dataset)):
        model = tf.keras.models.load_model(os.path.join(model_config['dirs']['saved_models'], dataset))
    else:
        print('PLEASE, TRAIN THE MODEL FIRST OR PUT PRETRAINED MODEL IN "saved_model" DIRECTORY')
        return

    pred = model.predict(test_x, batch_size=model_config[dataset]['batch_size'], verbose=1)

    return pred
