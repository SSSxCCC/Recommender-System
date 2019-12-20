import tensorflow as tf
from tensorflow.python.ops import embedding_ops


class Embedding2D(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 output_width,
                 output_height,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        dtype = kwargs.pop('dtype', tf.keras.backend.floatx())
        super(Embedding2D, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_width = output_width
        self.output_height = output_height
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_width, self.output_height),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            name='embeddings2d',)

    def call(self, inputs):
        return embedding_ops.embedding_lookup(self.embeddings, inputs)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_width': self.output_width,
            'output_height': self.output_height,
            'embeddings_initializer': tf.keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': tf.keras.regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': tf.keras.constraints.serialize(self.embeddings_constraint),
        }
        base_config = super(Embedding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
