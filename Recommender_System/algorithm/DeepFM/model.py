import tensorflow as tf
from Recommender_System.utility.decorator import logger


@logger('初始化DeepFM模型：', ('n_user', 'n_item', 'dim', 'layers', 'l2'))
def DeepFM_model(n_user: int, n_item: int, dim=8, layers=[16, 16, 16], l2=1e-6) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(l2)

    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    user_embedding = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=l2)(user_id)

    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
    item_embedding = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=l2)(item_id)

    user_bias = tf.keras.layers.Embedding(n_user, 1, embeddings_initializer='zeros')(user_id)
    item_bias = tf.keras.layers.Embedding(n_item, 1, embeddings_initializer='zeros')(item_id)
    fm = tf.reduce_sum(user_embedding * item_embedding, axis=1, keepdims=True) + user_bias + item_bias

    deep = tf.concat([user_embedding, item_embedding], axis=1)
    for layer in layers:
        deep = tf.keras.layers.Dense(layer, activation='relu', kernel_regularizer=l2)(deep)
    deep = tf.keras.layers.Dense(1, kernel_regularizer=l2)(deep)

    out = tf.keras.activations.sigmoid(fm + deep)
    return tf.keras.Model(inputs=[user_id, item_id], outputs=out)


if __name__ == '__main__':
    tf.keras.utils.plot_model(DeepFM_model(1, 1), 'graph.png', show_shapes=True)
