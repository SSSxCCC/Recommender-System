import tensorflow as tf
from tensorflow.keras.regularizers import l2 as reg_l2
from Recommender_System.utility.decorator import logger


@logger('初始化FM模型：', ('n_user', 'n_item', 'dim', 'l2'))
def FM_model(n_user: int, n_item: int, dim=8, l2=1e-6) -> tf.keras.Model:
    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    user_embedding = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=reg_l2(l2))(user_id)
    user_bias = tf.keras.layers.Embedding(n_user, 1, embeddings_initializer='zeros')(user_id)

    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
    item_embedding = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=reg_l2(l2))(item_id)
    item_bias = tf.keras.layers.Embedding(n_item, 1, embeddings_initializer='zeros')(item_id)

    x = tf.reduce_sum(user_embedding * item_embedding, axis=1, keepdims=True) + user_bias + item_bias
    out = tf.keras.activations.sigmoid(x)
    return tf.keras.Model(inputs=[user_id, item_id], outputs=out)


if __name__ == '__main__':
    tf.keras.utils.plot_model(FM_model(1, 1), 'graph.png', show_shapes=True)
