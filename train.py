from model.har_model import get_model
import tensorflow as tf
import argparse
import warnings

tf.keras.backend.clear_session()
warnings.filterwarnings("ignore")


def train_model(dataset: str):
    # n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    if dataset == 'pamap2':
        n_timesteps, n_features, n_outputs = 33, 18, 19
        d_model = 128

    model = get_model(n_timesteps, n_features, n_outputs, d_model=d_model)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # verbose, epochs, batch_size = 1, 15, 128
    # earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
    # # mcp_save = ModelCheckpoint('test_3_best.hdf5', save_best_only=True, monitor='val_acc', mode='max')
    # reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4, mode='min')
    #
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(val_x, val_y), callbacks=[reduce_lr_loss, earlyStopping])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Attention Based HAR Model Training')
    parser.add_argument('-d', '--dataset', default='pamap2', type=str, help='Name of Dataset for Model Training')
    args = parser.parse_args()

    train_model(args.dataset)
