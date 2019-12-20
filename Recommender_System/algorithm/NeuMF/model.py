from typing import Tuple
import tensorflow as tf
from tensorflow.keras.regularizers import l2 as reg_l2


class NeuMF(tf.keras.Model):
    def __init__(self, n_user: int, n_item: int, gmf_dim=8, mlp_dim=32, layers=[32, 16, 8], l2=1e-6):
        super(NeuMF, self).__init__()
        print('初始化NeuMF模型：n_user=', n_user, ', n_item=', n_item, ', gmf_dim=', gmf_dim,
              ', mlp_dim=', mlp_dim, ', layers=', layers, ', l2=', l2, sep='')
        self.gmf_user_embedding = tf.keras.layers.Embedding(n_user, gmf_dim, embeddings_regularizer=reg_l2(l2))
        self.gmf_item_embedding = tf.keras.layers.Embedding(n_item, gmf_dim, embeddings_regularizer=reg_l2(l2))

        self.mlp_user_embedding = tf.keras.layers.Embedding(n_user, mlp_dim, embeddings_regularizer=reg_l2(l2))
        self.mlp_item_embedding = tf.keras.layers.Embedding(n_item, mlp_dim, embeddings_regularizer=reg_l2(l2))
        self.dense_layers = [tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=reg_l2(l2)) for n in layers]

        self.out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_l2(l2))

    def call(self, inputs):  # user_id, item_id
        gmf_layer = self.gmf_user_embedding(inputs['user_id']) * self.gmf_item_embedding(inputs['item_id'])

        mlp_layer = tf.concat([self.mlp_user_embedding(inputs['user_id']), self.mlp_item_embedding(inputs['item_id'])], axis=1)
        for dense_layer in self.dense_layers:
            mlp_layer = dense_layer(mlp_layer)

        neumf_layer = tf.concat([gmf_layer, mlp_layer], axis=1)
        return self.out(neumf_layer)


def NeuMF_model(n_user: int, n_item: int, gmf_dim=8, mlp_dim=32, layers=[32, 16, 8], l2=1e-6) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    print('初始化NeuMF模型：n_user=', n_user, ', n_item=', n_item, ', gmf_dim=', gmf_dim,
          ', mlp_dim=', mlp_dim, ', layers=', layers, ', l2=', l2, sep='')
    user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
    item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)

    u = tf.keras.layers.Embedding(n_user, gmf_dim, embeddings_regularizer=reg_l2(l2))(user_id)
    i = tf.keras.layers.Embedding(n_item, gmf_dim, embeddings_regularizer=reg_l2(l2))(item_id)
    gmf = u * i
    gmf_out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_l2(l2), name='gmf_out')(gmf)

    u = tf.keras.layers.Embedding(n_user, mlp_dim, embeddings_regularizer=reg_l2(l2))(user_id)
    i = tf.keras.layers.Embedding(n_item, mlp_dim, embeddings_regularizer=reg_l2(l2))(item_id)
    mlp = tf.concat([u, i], axis=1)
    for n in layers:
        mlp = tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=reg_l2(l2))(mlp)
    mlp_out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_l2(l2), name='mlp_out')(mlp)

    x = tf.concat([gmf, mlp], axis=1)
    out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_l2(l2), name='out')(x)
    return tf.keras.Model(inputs=[user_id, item_id], outputs=out),\
           tf.keras.Model(inputs=[user_id, item_id], outputs=gmf_out),\
           tf.keras.Model(inputs=[user_id, item_id], outputs=mlp_out)


if __name__ == '__main__':
    # tf.keras.utils.plot_model(NeuMF_model(1, 1)[0], 'graph.png', show_shapes=True, rankdir='BT')

    input1 = tf.keras.Input(shape=(3,), name='in1')
    input2 = tf.keras.Input(shape=(3,), name='in2')

    gmf_out = tf.keras.layers.Dense(1, bias_initializer='glorot_uniform', name='gmf_out')(input1)
    mlp_out = tf.keras.layers.Dense(1, bias_initializer='glorot_uniform', name='mlp_out')(input2)

    concat = tf.concat([input1, input2], axis=1)
    neumf_out = tf.keras.layers.Dense(1, bias_initializer='glorot_uniform', name='out')(concat)
    another_out = (gmf_out + mlp_out) * 0.5

    gmf_model = tf.keras.Model(inputs=input1, outputs=gmf_out)
    mlp_model = tf.keras.Model(inputs=input2, outputs=mlp_out)
    neumf_model = tf.keras.Model(inputs=[input1, input2], outputs=neumf_out)
    another_model = tf.keras.Model(inputs=[input1, input2], outputs=another_out)

    out_kernel = tf.concat((gmf_model.get_layer('gmf_out').get_weights()[0], mlp_model.get_layer('mlp_out').get_weights()[0]), 0)
    out_bias = gmf_model.get_layer('gmf_out').get_weights()[1] + mlp_model.get_layer('mlp_out').get_weights()[1]
    neumf_model.get_layer('out').set_weights([out_kernel * 0.5, out_bias * 0.5])

    in1 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    in2 = tf.constant([[4, 2, 5], [6, 3, 1]], dtype=tf.float32)
    inp = {'in1': in1, 'in2': in2}

    print(neumf_model(inp))
    print(another_model(inp))
