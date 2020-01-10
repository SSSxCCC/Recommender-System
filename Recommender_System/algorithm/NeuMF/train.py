from Recommender_System.algorithm.NeuMF.model import NeuMF_model
from Recommender_System.algorithm.train import train, test
import tensorflow as tf


def train_with_pretrain(n_user, n_item, train_data, test_data, topk_data, gmf_dim, mlp_dim, layers, l2):
    neumf_model, gmf_model, mlp_model = NeuMF_model(n_user, n_item, gmf_dim=gmf_dim, mlp_dim=mlp_dim, layers=layers, l2=l2)
    print('预训练GMF部分')
    train(gmf_model, train_data, test_data, topk_data, epochs=10, batch=512)
    print('预训练MLP部分')
    train(mlp_model, train_data, test_data, topk_data, epochs=10, batch=512)

    out_kernel = tf.concat((gmf_model.get_layer('gmf_out').get_weights()[0], mlp_model.get_layer('mlp_out').get_weights()[0]), 0)
    out_bias = gmf_model.get_layer('gmf_out').get_weights()[1] + mlp_model.get_layer('mlp_out').get_weights()[1]
    neumf_model.get_layer('out').set_weights([out_kernel * 0.5, out_bias * 0.5])

    test(neumf_model, train_data, test_data, topk_data, batch=512)
    train(neumf_model, train_data, test_data, topk_data, optimizer=tf.keras.optimizers.SGD(0.0001), epochs=10, batch=512)


def train_without_pretrain(n_user, n_item, train_data, test_data, topk_data, gmf_dim, mlp_dim, layers, l2):
    neumf_model, _, _ = NeuMF_model(n_user, n_item, gmf_dim=gmf_dim, mlp_dim=mlp_dim, layers=layers, l2=l2)
    train(neumf_model, train_data, test_data, topk_data, epochs=10, batch=512)
