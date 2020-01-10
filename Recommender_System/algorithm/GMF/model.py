import tensorflow as tf
from Recommender_System.utility.decorator import logger


@logger('初始化GMF模型：', ('n_user', 'n_item', 'dim', 'l2'))
def GMF_model(n_user, n_item, dim=8, l2=1e-6) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(l2)

    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    u = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=l2)(user_id)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
    i = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=l2)(item_id)

    x = u * i
    x = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2)(x)
    return tf.keras.Model(inputs=[user_id, item_id], outputs=x)


if __name__ == '__main__':
    model = GMF_model(3, 3)
    model.summary()
    tf.keras.utils.plot_model(model, 'graph.png', show_shapes=True)
