import tensorflow as tf
from Recommender_System.utility.decorator import logger


@logger('初始化LFM模型：', ('n_user', 'n_item', 'dim', 'l2'))
def LFM_model(n_user: int, n_item: int, dim=100, l2=1e-6) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(l2)
    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    u = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=l2)(user_id)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
    i = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=l2)(item_id)
    x = tf.reduce_sum(u * i, axis=1)
    x = tf.where(x < 0., 0., x)
    x = tf.where(x > 1., 1., x)
    x = x[..., tf.newaxis]
    return tf.keras.Model(inputs=[user_id, item_id], outputs=x)


if __name__ == '__main__':
    tf.keras.utils.plot_model(LFM_model(1, 1), 'graph.png', show_shapes=True)
