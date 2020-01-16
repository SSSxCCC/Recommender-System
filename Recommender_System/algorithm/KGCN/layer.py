from abc import abstractmethod
import tensorflow as tf


class Aggregator(tf.keras.layers.Layer):
    def __init__(self, activation='relu', kernel_regularizer=None, **kwargs):
        super(Aggregator, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def call(self, inputs, **kwargs):
        self_vectors, neighbor_vectors, neighbor_relations, user_embeddings = inputs

        _, neighbor_iter, dim = self_vectors.shape
        neighbor_size = kwargs['neighbor_size']

        neighbor_vectors = tf.reshape(neighbor_vectors, shape=(-1, neighbor_iter, neighbor_size, dim))
        neighbor_relations = tf.reshape(neighbor_relations, shape=(-1, neighbor_iter, neighbor_size, dim))

        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, **kwargs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, **kwargs):
        # self_vectors: [batch, neighbor_iter, dim]
        # neighbor_vectors: [batch, neighbor_iter, neighbor_size, dim]
        # neighbor_relations: [batch, neighbor_iter, neighbor_size, dim]
        # user_embeddings: [batch, dim]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        dim = user_embeddings.shape[-1]
        avg = False
        if not avg:
            user_embeddings = tf.reshape(user_embeddings, shape=(-1, 1, 1, dim))  # [batch, 1, 1, dim]

            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)  # [batch, neighbor_iter, neighbor_size]
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, axis=-1)  # [batch, neighbor_iter, neighbor_size]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)  # [batch, neighbor_iter, neighbor_size, 1]

            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)  # [batch, neighbor_iter, dim]
        else:
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)  # [batch, neighbor_iter, dim]

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def build(self, input_shape):
        dim = input_shape[-1][-1]
        self.kernel = self.add_weight('kernel', shape=(dim, dim), initializer='glorot_uniform', regularizer=self.kernel_regularizer)
        self.bias = self.add_weight('bias', shape=(dim,), initializer='zeros')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, **kwargs):
        _, neighbor_iter, dim = self_vectors.shape
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)  # [batch, neighbor_iter, dim]

        output = tf.reshape(self_vectors + neighbors_agg, shape=(-1, dim))  # [batch * neighbor_iter, dim]
        #if kwargs['training']:
        #    output = tf.nn.dropout(output, rate=0.2)
        output = tf.nn.bias_add(tf.matmul(output, self.kernel), self.bias)  # [batch * neighbor_iter, dim]

        return tf.reshape(output, shape=(-1, neighbor_iter, dim))  # [batch, neighbor_iter, dim]


class ConcatAggregator(Aggregator):
    def build(self, input_shape):
        dim = input_shape[-1][-1]
        self.kernel = self.add_weight('kernel', shape=(dim * 2, dim), initializer='glorot_uniform', regularizer=self.kernel_regularizer)
        self.bias = self.add_weight('bias', shape=(dim,), initializer='zeros')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, **kwargs):
        _, neighbor_iter, dim = self_vectors.shape
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)  # [batch, neighbor_iter, dim]

        output = tf.concat([self_vectors, neighbors_agg], axis=2)  # [batch, neighbor_iter, dim * 2]
        output = tf.reshape(output, shape=(-1, dim * 2))  # [batch * neighbor_iter, dim * 2]
        #if kwargs['training']:
        #    output = tf.nn.dropout(output, rate=0.2)
        output = tf.nn.bias_add(tf.matmul(output, self.kernel), self.bias)  # [batch * neighbor_iter, dim]

        return tf.reshape(output, shape=(-1, neighbor_iter, dim))  # [batch, neighbor_iter, dim]


class NeighborAggregator(Aggregator):
    def build(self, input_shape):
        dim = input_shape[-1][-1]
        self.kernel = self.add_weight('kernel', shape=(dim, dim), initializer='glorot_uniform', regularizer=self.kernel_regularizer)
        self.bias = self.add_weight('bias', shape=(dim,), initializer='zeros')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, **kwargs):
        _, neighbor_iter, dim = self_vectors.shape
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)  # [batch, neighbor_iter, dim]

        output = tf.reshape(neighbors_agg, shape=(-1, dim))  # [batch * neighbor_iter, dim]
        #if kwargs['training']:
        #    output = tf.nn.dropout(output, rate=0.2)
        output = tf.nn.bias_add(tf.matmul(output, self.kernel), self.bias)  # [batch * neighbor_iter, dim]

        return tf.reshape(output, shape=(-1, neighbor_iter, dim))  # [batch, neighbor_iter, dim]
