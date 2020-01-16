from typing import List
import tensorflow as tf
from Recommender_System.algorithm.KGCN.layer import SumAggregator, ConcatAggregator, NeighborAggregator
from Recommender_System.algorithm.KGNNLS.layer import LabelAggregator, HashLookupWrapper
from Recommender_System.utility.decorator import logger


@logger('初始化KGNNLS模型：', ('n_user', 'n_entity', 'n_relation', 'neighbor_size', 'iter_size', 'dim', 'l2', 'ls', 'aggregator'))
def KGNNLS_model(n_user: int, n_entity: int, n_relation: int, adj_entity: List[List[int]], adj_relation: List[List[int]],
                 interaction_table: tf.lookup.StaticHashTable, neighbor_size: int, iter_size=2, dim=16, l2=1e-7, ls=1.,
                 aggregator='sum') -> tf.keras.Model:
    assert neighbor_size == len(adj_entity[0]) == len(adj_relation[0])
    l2 = tf.keras.regularizers.l2(l2)

    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)

    user_embedding = tf.keras.layers.Embedding(n_user, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=l2)
    entity_embedding = tf.keras.layers.Embedding(n_entity, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=l2)
    relation_embedding = tf.keras.layers.Embedding(n_relation, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=l2)

    u = user_embedding(user_id)

    flatten = tf.keras.layers.Flatten()
    entities = [tf.expand_dims(item_id, axis=1)]  # [(batch, 1), (batch, n_neighbor), (batch, n_neighbor^2), ..., (batch, n_neighbor^n_iter)]
    relations = []  # [(batch, n_neighbor), (batch, n_neighbor^2), ..., (batch, n_neighbor^n_iter)]
    for _ in range(iter_size):
        neighbor_entities = flatten(tf.gather(adj_entity, entities[-1]))
        neighbor_relations = flatten(tf.gather(adj_relation, entities[-1]))
        entities.append(neighbor_entities)
        relations.append(neighbor_relations)

    if aggregator == 'sum':
        aggregator_class = SumAggregator
    elif aggregator == 'concat':
        aggregator_class = ConcatAggregator
    elif aggregator == 'neighbor':
        aggregator_class = NeighborAggregator
    else:
        raise Exception("Unknown aggregator: " + aggregator)

    entity_vectors = [entity_embedding(entity) for entity in entities]  # [(batch, 1, dim), (batch, n_neighbor, dim), (batch, n_neighbor^2, dim), ..., (batch, n_neighbor^n_iter, dim)]
    relation_vectors = [relation_embedding(relation) for relation in relations]  # [(batch, n_neighbor, dim), (batch, n_neighbor^2, dim), ..., (batch, n_neighbor^n_iter, dim)]
    for it in range(iter_size):
        aggregator = aggregator_class(activation='relu' if it < iter_size - 1 else 'tanh', kernel_regularizer=l2)
        entities_next = []
        for hop in range(iter_size - it):
            inputs = (entity_vectors[hop], entity_vectors[hop + 1], relation_vectors[hop], u)
            vector = aggregator(inputs, neighbor_size=neighbor_size)
            entities_next.append(vector)
        entity_vectors = entities_next
    i = tf.reshape(entity_vectors[0], shape=(-1, dim))  # batch, dim
    score = tf.sigmoid(tf.reduce_sum(u * i, axis=1))  # batch

    # calculate initial labels; calculate updating masks for label propagation
    entity_labels = []
    reset_masks = []  # True means the label of this item is reset to initial value during label propagation
    holdout_item_for_user = None
    offset = tf.constant(10 ** len(str(n_entity)), dtype=tf.int64)
    interaction_table_lookup = HashLookupWrapper(interaction_table)

    for entities_per_iter in entities:
        users = tf.cast(tf.expand_dims(user_id, axis=1), dtype=tf.int64)  # [batch_size, 1]
        user_entity_concat = users * offset + tf.cast(entities_per_iter, dtype=tf.int64)  # [batch_size, n_neighbor^i]

        if holdout_item_for_user is None:  # the first one in entities is the items to be held out
            holdout_item_for_user = user_entity_concat  # [batch, 1]

        initial_label = interaction_table_lookup(user_entity_concat)  # [batch_size, n_neighbor^i]
        holdout_mask = tf.cast(holdout_item_for_user - user_entity_concat, tf.bool)  # False if the item is held out
        reset_mask = tf.cast(initial_label - tf.constant(0.5), tf.bool)  # True if the entity is a labeled item
        reset_mask = tf.logical_and(reset_mask, holdout_mask)  # remove held-out items
        initial_label = tf.cast(holdout_mask, tf.keras.backend.floatx()) * initial_label + tf.cast(
            tf.logical_not(holdout_mask), tf.keras.backend.floatx()) * tf.constant(0.5)  # label initialization

        reset_masks.append(reset_mask)
        entity_labels.append(initial_label)
    reset_masks = reset_masks[:-1]  # we do not need the reset_mask for the last iteration

    # label propagation
    aggregator = LabelAggregator()
    for it in range(iter_size):
        entity_labels_next = []
        for hop in range(iter_size - it):
            inputs = (entity_labels[hop], entity_labels[hop + 1], relation_vectors[hop], u, reset_masks[hop])
            vector = aggregator(inputs, neighbor_size=neighbor_size)
            entity_labels_next.append(vector)
        entity_labels = entity_labels_next
    predicted_labels = tf.squeeze(entity_labels[0], axis=1)  # batch

    label_keys = tf.cast(user_id, dtype=tf.int64) * offset + tf.cast(item_id, dtype=tf.int64)  # batch
    labels = interaction_table_lookup(label_keys)  # batch
    ls_loss = tf.keras.losses.binary_crossentropy(labels, predicted_labels)

    model = tf.keras.Model(inputs=[user_id, item_id], outputs=score)
    model.add_loss(ls_loss * ls)
    return model


if __name__ == '__main__':
    pass
