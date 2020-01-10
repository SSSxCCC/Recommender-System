from Recommender_System.utility.decorator import logger
import tensorflow as tf


@logger('根据训练集数据构建用户物品-标签哈希表，', ('n_entity',))
def get_interaction_table(train_data, n_entity) -> tf.lookup.StaticHashTable:
    offset = 10 ** len(str(n_entity))
    keys = tf.constant([d[0] * offset + d[1] for d in train_data], dtype=tf.int64)
    values = tf.constant([d[2] for d in train_data], dtype=tf.keras.backend.floatx())

    interaction_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int64, value_dtype=tf.keras.backend.floatx()),
        default_value=0.5)

    return interaction_table
