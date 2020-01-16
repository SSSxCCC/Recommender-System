import tensorflow as tf


class LabelAggregator(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        self_labels, neighbor_labels, neighbor_relations, user_embeddings, masks = inputs

        neighbor_size = kwargs['neighbor_size']
        neighbor_iter = self_labels.shape[1]
        dim = user_embeddings.shape[1]

        neighbor_labels = tf.reshape(neighbor_labels, shape=(-1, neighbor_iter, neighbor_size))
        neighbor_relations = tf.reshape(neighbor_relations, shape=(-1, neighbor_iter, neighbor_size, dim))
        user_embeddings = tf.reshape(user_embeddings, shape=(-1, 1, 1, dim))

        # [batch, neighbor_iter, neighbor_size]
        user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
        user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, axis=-1)

        # [batch, neighbor_iter]
        neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_labels, axis=-1)
        output = tf.cast(masks, tf.keras.backend.floatx()) * self_labels + tf.cast(
            tf.logical_not(masks), tf.keras.backend.floatx()) * neighbors_aggregated

        return output


class HashLookupWrapper(tf.keras.layers.Layer):
    def __init__(self, hashtable, **kwargs):
        super(HashLookupWrapper, self).__init__(**kwargs)
        self.hashtable = hashtable

    def call(self, inputs, **kwargs):
        return self.hashtable.lookup(inputs)
