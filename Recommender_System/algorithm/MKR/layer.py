import tensorflow as tf


class CrossLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        v, e = inputs  # (batch, dim)
        v = tf.expand_dims(v, axis=2)  # (batch, dim, 1)
        e = tf.expand_dims(e, axis=1)  # (batch, 1, dim)
        c_matrix = tf.matmul(v, e)  # (batch, dim, dim)
        c_matrix_t = tf.transpose(c_matrix, perm=[0, 2, 1])  # (batch, dim, dim)
        return c_matrix, c_matrix_t


class CompressLayer(tf.keras.layers.Layer):
    def __init__(self, weight_regularizer, **kwargs):
        super(CompressLayer, self).__init__(**kwargs)
        self.weight_regularizer = tf.keras.regularizers.get(weight_regularizer)

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.weight = self.add_weight(shape=(self.dim, 1), regularizer=self.weight_regularizer, name='weight')
        self.weight_t = self.add_weight(shape=(self.dim, 1), regularizer=self.weight_regularizer, name='weight_t')
        self.bias = self.add_weight(shape=self.dim, initializer='zeros', name='bias')

    def call(self, inputs):
        c_matrix, c_matrix_t = inputs  # (batch, dim, dim)

        c_matrix = tf.reshape(c_matrix, shape=[-1, self.dim])  # (batch * dim, dim)
        c_matrix_t = tf.reshape(c_matrix_t, shape=[-1, self.dim])  # (batch * dim, dim)

        return tf.reshape(tf.matmul(c_matrix, self.weight) + tf.matmul(c_matrix_t, self.weight_t),
                          shape=[-1, self.dim]) + self.bias  # (batch, dim)


def cross_compress_unit(inputs, weight_regularizer):
    cross_feature_matrix = CrossLayer()(inputs)

    v_out = CompressLayer(weight_regularizer)(cross_feature_matrix)
    e_out = CompressLayer(weight_regularizer)(cross_feature_matrix)

    return v_out, e_out
