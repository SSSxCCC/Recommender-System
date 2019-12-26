from typing import Tuple
import tensorflow as tf
from tensorflow.keras.regularizers import l2 as reg_l2
from Recommender_System.algorithm.MKR.layer import cross_compress_unit
from Recommender_System.utility.decorator import logger


@logger('初始化MKR模型：', ('n_user', 'n_item', 'n_entity', 'n_relation', 'dim', 'L', 'H', 'l2'))
def MKR_model(n_user: int, n_item: int, n_entity: int, n_relation: int, dim=8, L=1, H=1, l2=1e-6) -> Tuple[tf.keras.Model, tf.keras.Model]:
    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
    head_id = tf.keras.Input(shape=(), name='head_id', dtype=tf.int32)
    relation_id = tf.keras.Input(shape=(), name='relation_id', dtype=tf.int32)
    tail_id = tf.keras.Input(shape=(), name='tail_id', dtype=tf.int32)

    user_embedding = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=reg_l2(l2))
    item_embedding = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=reg_l2(l2))
    entity_embedding = tf.keras.layers.Embedding(n_entity, dim, embeddings_regularizer=reg_l2(l2))
    relation_embedding = tf.keras.layers.Embedding(n_relation, dim, embeddings_regularizer=reg_l2(l2))

    u = user_embedding(user_id)
    i = item_embedding(item_id)
    h = entity_embedding(head_id)
    r = relation_embedding(relation_id)
    t = entity_embedding(tail_id)

    for _ in range(L):
        u = tf.keras.layers.Dense(dim, activation='relu', kernel_regularizer=reg_l2(l2))(u)
        i, h = cross_compress_unit(inputs=(i, h), weight_regularizer=reg_l2(l2))
        t = tf.keras.layers.Dense(dim, activation='relu', kernel_regularizer=reg_l2(l2))(t)

    #rs = tf.concat([u, i], axis=1)
    rs = tf.keras.activations.sigmoid(tf.reduce_sum(u * i, axis=1, keepdims=True))
    kge = tf.concat([h, r], axis=1)
    for _ in range(H - 1):
        #rs = tf.keras.layers.Dense(dim * 2, activation='relu', kernel_regularizer=reg_l2(l2))(rs)
        kge = tf.keras.layers.Dense(dim * 2, activation='relu', kernel_regularizer=reg_l2(l2))(kge)
    #rs = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_l2(l2))(rs)
    kge = tf.keras.layers.Dense(dim, activation='sigmoid', kernel_regularizer=reg_l2(l2))(kge)
    kge = -tf.keras.activations.sigmoid(tf.reduce_sum(t * kge, axis=1))
    return tf.keras.Model(inputs=[user_id, item_id, head_id], outputs=rs),\
           tf.keras.Model(inputs=[item_id, head_id, relation_id, tail_id], outputs=kge)


if __name__ == '__main__':
    rs_model, kge_model = MKR_model(2, 2, 2, 2)
    u = tf.constant([0, 1])
    i = tf.constant([1, 0])
    h = tf.constant([0, 1])
    r = tf.constant([1, 0])
    t = tf.constant([0, 1])
    print(rs_model({'user_id': u, 'item_id': i, 'head_id': h}))
    print(kge_model({'item_id': i, 'head_id': h, 'relation_id': r, 'tail_id': t}))

    ds = tf.data.Dataset.from_tensor_slices(({'item_id': i, 'head_id': h, 'relation_id': r, 'tail_id': t}, tf.constant([0] * 2))).batch(2)
    kge_model.compile(optimizer='adam', loss=lambda y_true, y_pre: y_pre)
    kge_model.fit(ds, epochs=3)

    #ds = tf.data.Dataset.from_tensor_slices(({'user_id': u, 'item_id': i, 'head_id': h}, tf.constant([0., 1.]))).batch(2)
    #rs_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())
    #rs_model.fit(ds, epochs=3)
