import tensorflow as tf
from Recommender_System.utility.decorator import logger


@logger('初始化MLP模型：', ('n_user', 'n_item', 'dim', 'layers', 'l2', 'dropout'))
def MLP_model(n_user: int, n_item: int, dim=32, layers=[32, 16, 8], l2=1e-6, dropout=0.2) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(l2)
    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    u = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=l2)(user_id)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
    i = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=l2)(item_id)
    x = tf.concat([u, i], axis=1)
    for n in layers:
        x = tf.keras.layers.Dropout(rate=dropout)(x)
        x = tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(rate=dropout)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2)(x)
    return tf.keras.Model(inputs=[user_id, item_id], outputs=x)


if __name__ == '__main__':
    tf.keras.utils.plot_model(MLP_model(1, 1), 'graph.png', show_shapes=True)
