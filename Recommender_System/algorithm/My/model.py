import tensorflow as tf
from tensorflow.keras.regularizers import l2 as reg_l2
from Recommender_System.algorithm.RippleNet.layer import Embedding2D
from Recommender_System.algorithm.KGCN.layer import SumAggregator
from Recommender_System.utility.decorator import logger


@logger('初始化My模型：', )
def My_model(n_user: int, n_entity: int, n_relation: int, adj_entity, adj_relation, dim=16, iter_size=2, hop_size=2,
             ripple_size=32, l2=1e-7, item_update_mode='plus_transform', use_all_hops=True) -> tf.keras.Model:
    neighbor_size = len(adj_entity[0])
    print('n_user=', n_user, ', n_entity=', n_entity, ', n_relation=', n_relation, ', dim=', dim,
          ', iter_size=', iter_size, ', neighbor_size=', neighbor_size,
          ', hop_size=', hop_size, ', ripple_size=', ripple_size, ', l2=', l2, sep='')

    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)

    user_embedding = tf.keras.layers.Embedding(n_user, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=reg_l2(l2))
    entity_embedding = tf.keras.layers.Embedding(n_entity, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=reg_l2(l2))
    relation_embedding_agg = tf.keras.layers.Embedding(n_relation, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=reg_l2(l2))

    u_agg = user_embedding(user_id)

    flatten = tf.keras.layers.Flatten()
    entities = [tf.expand_dims(item_id, axis=1)]  # [(batch, 1), (batch, n_neighbor), (batch, n_neighbor^2), ..., (batch, n_neighbor^n_iter)]
    relations = []  # [(batch, n_neighbor), (batch, n_neighbor^2), ..., (batch, n_neighbor^n_iter)]
    for _ in range(iter_size):
        neighbor_entities = flatten(tf.gather(adj_entity, entities[-1]))
        neighbor_relations = flatten(tf.gather(adj_relation, entities[-1]))
        entities.append(neighbor_entities)
        relations.append(neighbor_relations)

    entity_vectors = [entity_embedding(entity) for entity in entities]  # [(batch, 1, dim), (batch, n_neighbor, dim), (batch, n_neighbor^2, dim), ..., (batch, n_neighbor^n_iter, dim)]
    relation_vectors = [relation_embedding_agg(relation) for relation in relations]  # [(batch, n_neighbor, dim), (batch, n_neighbor^2, dim), ..., (batch, n_neighbor^n_iter, dim)]
    for it in range(iter_size):
        aggregator = SumAggregator(activation='relu' if it < iter_size - 1 else 'tanh', kernel_regularizer=reg_l2(l2))
        entities_next = []
        for hop in range(iter_size - it):
            inputs = (entity_vectors[hop], entity_vectors[hop + 1], relation_vectors[hop], u_agg)
            vector = aggregator(inputs, neighbor_size=neighbor_size)
            entities_next.append(vector)
        entity_vectors = entities_next
    i_agg = flatten(entity_vectors[0])  # batch, dim

    #score_agg = tf.reduce_sum(u_agg * i_agg, axis=1, keepdims=True)  # batch, 1

    ####################################################################################################################

    ripple_h, ripple_r, ripple_t = [], [], []
    for hop in range(hop_size):
        ripple_h.append(tf.keras.Input(shape=(ripple_size,), name='ripple_h_' + str(hop), dtype=tf.int32))
        ripple_r.append(tf.keras.Input(shape=(ripple_size,), name='ripple_r_' + str(hop), dtype=tf.int32))
        ripple_t.append(tf.keras.Input(shape=(ripple_size,), name='ripple_t_' + str(hop), dtype=tf.int32))

    #entity_embedding = tf.keras.layers.Embedding(n_entity, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=reg_l2(l2))
    relation_embedding_prop = Embedding2D(n_relation, dim, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=reg_l2(l2))
    transform_matrix = tf.keras.layers.Dense(dim, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=reg_l2(l2))

    i_prop = entity_embedding(item_id)  # batch, dim
    h, r, t = [], [], []
    for hop in range(hop_size):
        h.append(entity_embedding(ripple_h[hop]))  # batch, ripple_size, dim
        r.append(relation_embedding_prop(ripple_r[hop]))  # batch, ripple_size, dim, dim
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
        v = tf.expand_dims(i_prop, axis=2)  # batch, dim, 1
        probs = tf.squeeze(tf.matmul(Rh, v), axis=2)  # batch, ripple_size
        probs_normalized = tf.keras.activations.softmax(probs)  # batch, ripple_size
        probs_expanded = tf.expand_dims(probs_normalized, axis=2)  # batch, ripple_size, 1
        o = tf.reduce_sum(t[hop] * probs_expanded, axis=1)  # batch, dim
        i_prop = update_item(i_prop, o)
        o_list.append(o)

    u_prop = sum(o_list) if use_all_hops else o_list[-1]

    score_prop = tf.reduce_sum(i_prop * u_prop, axis=1, keepdims=True)  # batch, 1

    extra_loss = 0
    for hop in range(hop_size):
        extra_loss += tf.reduce_sum(tf.square(h[hop]))
        extra_loss += tf.reduce_sum(tf.square(r[hop]))
        extra_loss += tf.reduce_sum(tf.square(t[hop]))
    extra_loss *= l2

    ####################################################################################################################

    #score_mix_low = tf.reduce_sum(u_agg * i_prop, axis=1, keepdims=True)  # batch, 1
    score_mix_high = tf.reduce_sum(i_agg * u_prop, axis=1, keepdims=True)  # batch, 1
    #score_mix = tf.reduce_sum(u_agg * entity_embedding(item_id), axis=1, keepdims=True)  # batch, 1

    #attention = entity_embedding(item_id) * u_agg  # batch, dim
    #attention = tf.keras.layers.Dense(4, 'softmax', kernel_regularizer=reg_l2(l2))(attention)  # batch, 4

    score = tf.concat([score_prop, score_mix_high], axis=1)  # batch, 2
    #score = tf.sigmoid(tf.reduce_sum(score * attention, axis=1))  # batch
    #score = tf.squeeze(tf.keras.layers.Dense(1, 'sigmoid', kernel_initializer=tf.keras.initializers.Constant(0.25), kernel_regularizer=reg_l2(l2), name='score')(score), axis=1)  # batch
    #score = tf.sigmoid(tf.squeeze(score_mix_high, axis=1))  # batch
    score = tf.sigmoid(tf.reduce_sum(score, axis=1))  # batch

    return tf.keras.Model(inputs=[user_id, item_id] + ripple_h + ripple_r + ripple_t, outputs=[score, extra_loss])


if __name__ == '__main__':
    adj = [[1, 2], [0, 2], [0, 1]]
    model = My_model(3, 3, 3, adj, adj)
    print(model.get_layer('score').variables)
