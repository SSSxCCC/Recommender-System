import tensorflow as tf
from tensorflow.keras.regularizers import l2 as reg_l2


class LFM(tf.keras.Model):
    def __init__(self, n_user: int, n_item: int, dim=100, l2=1e-6):
        super(LFM, self).__init__()
        print('初始化LFM模型：n_user=', n_user, ', n_item=', n_item, ', dim=', dim, ', l2=', l2, sep='')
        self.P = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=reg_l2(l2))
        self.Q = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=reg_l2(l2))

    def call(self, inputs):  # user_id, item_id
        score = tf.reduce_sum(self.P(inputs['user_id']) * self.Q(inputs['item_id']), 1)
        score = tf.where(score < 0., 0., score)
        return tf.where(score > 1., 1., score)[..., tf.newaxis]


def LFM_model(n_user: int, n_item: int, dim=100, l2=1e-6) -> tf.keras.Model:
    print('初始化LFM模型：n_user=', n_user, ', n_item=', n_item, ', dim=', dim, ', l2=', l2, sep='')
    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    u = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=reg_l2(l2))(user_id)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
    i = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=reg_l2(l2))(item_id)
    x = tf.reduce_sum(u * i, 1)
    x = tf.where(x < 0., 0., x)
    x = tf.where(x > 1., 1., x)
    x = x[..., tf.newaxis]
    return tf.keras.Model(inputs=[user_id, item_id], outputs=x)


if __name__ == '__main__':
    tf.keras.utils.plot_model(LFM_model(1, 1), 'graph.png', show_shapes=True)
