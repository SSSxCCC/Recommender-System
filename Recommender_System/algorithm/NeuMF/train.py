from Recommender_System.algorithm.GMF.model import GMF
from Recommender_System.algorithm.MLP.model import MLP
from Recommender_System.algorithm.NeuMF.model import NeuMF, NeuMF_model
from Recommender_System.algorithm.train import train, test
import tensorflow as tf


def combine_weights_(neumf_model, gmf_model, mlp_model):
    neumf_model.gmf_user_embedding.set_weights(gmf_model.user_embedding.get_weights())
    neumf_model.gmf_item_embedding.set_weights(gmf_model.item_embedding.get_weights())

    neumf_model.mlp_user_embedding.set_weights(mlp_model.user_embedding.get_weights())
    neumf_model.mlp_item_embedding.set_weights(mlp_model.item_embedding.get_weights())

    for neumf_layer, mlp_layer in zip(neumf_model.dense_layers, mlp_model.dense_layers):
        neumf_layer.set_weights(mlp_layer.get_weights())

    out_kernel = tf.concat((gmf_model.out.get_weights()[0], mlp_model.out.get_weights()[0]), 0)
    out_bias = gmf_model.out.get_weights()[1] + mlp_model.out.get_weights()[1]
    neumf_model.out.set_weights([out_kernel * 0.5, out_bias * 0.5])


def train_with_pretrain_(n_user, n_item, train_data, test_data, topk_data, gmf_dim, mlp_dim, layers, l2):
    gmf_model = GMF(n_user, n_item, dim=gmf_dim, l2=l2)
    train(gmf_model, train_data, test_data, topk_data, epochs=10, batch=512, execution='graph')

    mlp_model = MLP(n_user, n_item, dim=mlp_dim, layers=layers, l2=l2)
    train(mlp_model, train_data, test_data, topk_data, epochs=10, batch=512, execution='graph')

    neumf_model = NeuMF(n_user, n_item, gmf_dim=gmf_dim, mlp_dim=mlp_dim, layers=layers, l2=l2)
    neumf_model({'user_id': tf.constant([0]), 'item_id': tf.constant([0])})
    combine_weights_(neumf_model, gmf_model, mlp_model)
    test(neumf_model, train_data, test_data, topk_data, batch=512)
    train(neumf_model, train_data, test_data, topk_data, epochs=10, batch=512, execution='graph')


def train_with_pretrain(n_user, n_item, train_data, test_data, topk_data, gmf_dim, mlp_dim, layers, l2):
    neumf_model, gmf_model, mlp_model = NeuMF_model(n_user, n_item, gmf_dim=gmf_dim, mlp_dim=mlp_dim, layers=layers, l2=l2)
    train(gmf_model, train_data, test_data, topk_data, epochs=10, batch=512)
    train(mlp_model, train_data, test_data, topk_data, epochs=10, batch=512)

    out_kernel = tf.concat((gmf_model.get_layer('gmf_out').get_weights()[0], mlp_model.get_layer('mlp_out').get_weights()[0]), 0)
    out_bias = gmf_model.get_layer('gmf_out').get_weights()[1] + mlp_model.get_layer('mlp_out').get_weights()[1]
    neumf_model.get_layer('out').set_weights([out_kernel * 0.5, out_bias * 0.5])

    test(neumf_model, train_data, test_data, topk_data, batch=512)
    train(neumf_model, train_data, test_data, topk_data, optimizer=tf.keras.optimizers.SGD(0.0001), epochs=10, batch=512)


def train_without_pretrain(n_user, n_item, train_data, test_data, topk_data, gmf_dim, mlp_dim, layers, l2):
    neumf_model, _, _ = NeuMF_model(n_user, n_item, gmf_dim=gmf_dim, mlp_dim=mlp_dim, layers=layers, l2=l2)
    train(neumf_model, train_data, test_data, topk_data, epochs=10, batch=512)