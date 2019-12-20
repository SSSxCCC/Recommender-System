import tensorflow as tf
from tensorflow.keras.regularizers import l2 as reg_l2


class MLP(tf.keras.Model):
    def __init__(self, n_user: int, n_item: int, dim=32, layers=[32, 16, 8], l2=0.00001):
        super(MLP, self).__init__()
        print('初始化MLP模型：n_user=', n_user, ', n_item=', n_item, ', dim=', dim, ', layers=', layers, ', l2=', l2, sep='')
        self.user_embedding = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=reg_l2(l2))
        self.item_embedding = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=reg_l2(l2))
        self.dense_layers = [tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=reg_l2(l2)) for n in layers]
        self.out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_l2(l2))

    def call(self, inputs):  # user_id, item_id
        x = tf.concat([self.user_embedding(inputs['user_id']), self.item_embedding(inputs['item_id'])], axis=1)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        return self.out(x)


def MLP_model(n_user: int, n_item: int, dim=32, layers=[32, 16, 8], l2=1e-6) -> tf.keras.Model:
    print('初始化MLP模型：n_user=', n_user, ', n_item=', n_item, ', dim=', dim, ', layers=', layers, ', l2=', l2, sep='')
    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    u = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=reg_l2(l2))(user_id)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
    i = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=reg_l2(l2))(item_id)
    x = tf.concat([u, i], axis=1)
    for n in layers:
        x = tf.keras.layers.Dropout(rate=0.3)(x)
        x = tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=reg_l2(l2))(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_l2(l2))(x)
    return tf.keras.Model(inputs=[user_id, item_id], outputs=x)


if __name__ == '__main__':
    tf.keras.utils.plot_model(MLP_model(1, 1), 'graph.png', show_shapes=True)
