from typing import Tuple
import tensorflow as tf
from Recommender_System.utility.decorator import logger


@logger('初始化NeuMF模型：', ('n_user', 'n_item', 'gmf_dim', 'mlp_dim', 'layers', 'l2'))
def NeuMF_model(n_user: int, n_item: int, gmf_dim=8, mlp_dim=32, layers=[32, 16, 8], l2=1e-6) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    l2 = tf.keras.regularizers.l2(l2)

    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)

    u = tf.keras.layers.Embedding(n_user, gmf_dim, embeddings_regularizer=l2)(user_id)
    i = tf.keras.layers.Embedding(n_item, gmf_dim, embeddings_regularizer=l2)(item_id)
    gmf = u * i
    gmf_out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2, name='gmf_out')(gmf)

    u = tf.keras.layers.Embedding(n_user, mlp_dim, embeddings_regularizer=l2)(user_id)
    i = tf.keras.layers.Embedding(n_item, mlp_dim, embeddings_regularizer=l2)(item_id)
    mlp = tf.concat([u, i], axis=1)
    for n in layers:
        mlp = tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=l2)(mlp)
    mlp_out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2, name='mlp_out')(mlp)

    x = tf.concat([gmf, mlp], axis=1)
    out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2, name='out')(x)
    return tf.keras.Model(inputs=[user_id, item_id], outputs=out),\
           tf.keras.Model(inputs=[user_id, item_id], outputs=gmf_out),\
           tf.keras.Model(inputs=[user_id, item_id], outputs=mlp_out)


if __name__ == '__main__':
    tf.keras.utils.plot_model(NeuMF_model(1, 1)[0], 'graph.png', show_shapes=True, rankdir='BT')
