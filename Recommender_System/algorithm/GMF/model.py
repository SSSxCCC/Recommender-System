import tensorflow as tf
from tensorflow.keras.regularizers import l2 as reg_l2


class GMF(tf.keras.Model):
    def __init__(self, n_user, n_item, dim=8, l2=1e-6):
        super(GMF, self).__init__()
        print('初始化GMF模型：n_user=', n_user, ', n_item=', n_item, ', dim=', dim, ', l2=', l2, sep='')
        self.user_embedding = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=reg_l2(l2))
        self.item_embedding = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=reg_l2(l2))
        self.out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_l2(l2))

    def call(self, inputs):  # user_id, item_id
        x = self.user_embedding(inputs['user_id']) * self.item_embedding(inputs['item_id'])
        return self.out(x)


def GMF_model(n_user, n_item, dim=8, l2=1e-6) -> tf.keras.Model:
    print('初始化GMF模型：n_user=', n_user, ', n_item=', n_item, ', dim=', dim, ', l2=', l2, sep='')
    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    u = tf.keras.layers.Embedding(n_user, dim, embeddings_regularizer=reg_l2(l2))(user_id)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
    i = tf.keras.layers.Embedding(n_item, dim, embeddings_regularizer=reg_l2(l2))(item_id)
    x = u * i
    x = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_l2(l2))(x)
    return tf.keras.Model(inputs=[user_id, item_id], outputs=x)


if __name__ == '__main__':
    model = GMF_model(2, 10)

    @tf.function(experimental_relax_shapes=True)
    def fast_model(inputs):
        print('!!!!!!!!!!!!')
        return tf.squeeze(model(inputs))

    inputs0 = {'user_id': tf.constant([0]), 'item_id': tf.constant([5])}
    inputs1 = {'user_id': tf.constant([0, 1]), 'item_id': tf.constant([5, 7])}
    inputs2 = {'user_id': tf.constant([0, 1, 0]), 'item_id': tf.constant([5, 7, 3])}
    inputs3 = {'user_id': tf.constant([0, 1, 0, 1]), 'item_id': tf.constant([5, 7, 3, 2])}
    inputs4 = {'user_id': tf.constant([0, 1, 0, 1, 0]), 'item_id': tf.constant([5, 7, 3, 2, 0])}
    print(fast_model(inputs0).numpy())
    print(fast_model(inputs1).numpy())
    print(fast_model(inputs2).numpy())
    print(fast_model(inputs3).numpy())
    print(fast_model(inputs4).numpy())

    #model.summary()
    #tf.keras.utils.plot_model(model, 'graph.png', show_shapes=True)
