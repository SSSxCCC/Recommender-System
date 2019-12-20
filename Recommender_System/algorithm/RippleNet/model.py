import tensorflow as tf
from tensorflow.keras.regularizers import l2 as reg_l2
from Recommender_System.algorithm.RippleNet.layer import Embedding2D


def RippleNet_model(n_entity: int, n_relation: int, dim=16, hop_size=2, ripple_size=32, kge_weight=0.01, l2=1e-7,
                    item_update_mode='plus_transform', use_all_hops=True) -> tf.keras.Model:
    print('初始化RippleNet模型：n_entity=', n_entity, ', n_relation=', n_relation, ', dim=', dim,
          ', hop_size=', hop_size, ', ripple_size=', ripple_size, ', kge_weight=', kge_weight, ', l2=', l2, sep='')

    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
    ripple_h, ripple_r, ripple_t = [], [], []
    for hop in range(hop_size):
        ripple_h.append(tf.keras.Input(shape=(ripple_size,), name='ripple_h_' + str(hop), dtype=tf.int32))
        ripple_r.append(tf.keras.Input(shape=(ripple_size,), name='ripple_r_' + str(hop), dtype=tf.int32))
        ripple_t.append(tf.keras.Input(shape=(ripple_size,), name='ripple_t_' + str(hop), dtype=tf.int32))

    entity_embedding = tf.keras.layers.Embedding(n_entity, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=reg_l2(l2))
    relation_embedding = Embedding2D(n_relation, dim, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=reg_l2(l2))
    transform_matrix = tf.keras.layers.Dense(dim, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=reg_l2(l2))

    i = entity_embedding(item_id)  # batch, dim
    h, r, t = [], [], []
    for hop in range(hop_size):
        h.append(entity_embedding(ripple_h[hop]))  # batch, ripple_size, dim
        r.append(relation_embedding(ripple_r[hop]))  # batch, ripple_size, dim, dim
        t.append(entity_embedding(ripple_t[hop]))  # batch, ripple_size, dim

    def update_item(i, o):
        if item_update_mode == 'replace':
            i = o
        elif item_update_mode == 'plus':
            i = i + o
        elif item_update_mode == 'replace_transform':
            i = transform_matrix(o)
        elif item_update_mode == 'plus_transform':
            i = transform_matrix(i + o)
        else:
            raise Exception("Unknown item updating mode: " + item_update_mode)
        return i

    o_list = []
    for hop in range(hop_size):
        h_expanded = tf.expand_dims(h[hop], axis=3)  # batch, ripple_size, dim, 1
        Rh = tf.squeeze(tf.matmul(r[hop], h_expanded), axis=3)  # batch, ripple_size, dim
        v = tf.expand_dims(i, axis=2)  # batch, dim, 1
        probs = tf.squeeze(tf.matmul(Rh, v), axis=2)  # batch, ripple_size
        probs_normalized = tf.keras.activations.softmax(probs)  # batch, ripple_size
        probs_expanded = tf.expand_dims(probs_normalized, axis=2)  # batch, ripple_size, 1
        o = tf.reduce_sum(t[hop] * probs_expanded, axis=1)  # batch, dim
        i = update_item(i, o)
        o_list.append(o)

    u = sum(o_list) if use_all_hops else o_list[-1]

    score = tf.keras.layers.Activation('sigmoid', name='score')(tf.reduce_sum(i * u, axis=1))  # batch

    kge_loss = 0
    for hop in range(hop_size):
        h_expanded = tf.expand_dims(h[hop], axis=2)  # batch, ripple_size, 1, dim
        t_expanded = tf.expand_dims(t[hop], axis=3)  # batch, ripple_size, dim, 1
        hRt = tf.squeeze(h_expanded @ r[hop] @ t_expanded)  # batch, ripple_size
        kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
    kge_loss = tf.keras.layers.Layer(name='kge_loss')(kge_weight * -kge_loss)

    l2_loss = 0  # tf.reduce_sum(tf.square(transform_matrix.kernel)) if item_update_mode in {'replace_transform', 'plus_transform'} else 0
    for hop in range(hop_size):
        l2_loss += tf.reduce_sum(tf.square(h[hop]))
        l2_loss += tf.reduce_sum(tf.square(r[hop]))
        l2_loss += tf.reduce_sum(tf.square(t[hop]))
    l2_loss = tf.keras.layers.Layer(name='l2_loss')(l2 * l2_loss)

    return tf.keras.Model(inputs=[item_id] + ripple_h + ripple_r + ripple_t, outputs=[score, kge_loss, l2_loss])


if __name__ == '__main__':
    pass
