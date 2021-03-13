import tensorflow as tf


class SensorAttention(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_size, dilation_rate):
        super(SensorAttention, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(n_filters, kernel_size=kernel_size,
                                             dilation_rate=dilation_rate, padding='same', activation='relu')
        self.conv_f = tf.keras.layers.Conv2D(1, kernel_size=1, padding='same')
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.ln(x)
        x1 = tf.expand_dims(x, axis=3)
        x1 = self.conv_1(x1)
        x1 = self.conv_f(x1)
        x1 = tf.keras.activations.softmax(x1, axis=2)
        x1 = tf.keras.layers.Reshape(x.shape[-2:])(x1)
        return tf.math.multiply(x, x1), x1
