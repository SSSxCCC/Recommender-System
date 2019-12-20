import Recommender_System.utility.gpu_memory_growth
import tensorflow as tf
from Recommender_System.data import data_loader, data_process
from Recommender_System.algorithm.FM.model import FM_model
from Recommender_System.algorithm.GMF.model import GMF_model
from Recommender_System.algorithm.LFM.model import LFM_model
from Recommender_System.algorithm.MLP.model import MLP_model
from Recommender_System.algorithm.NeuMF.model import NeuMF_model
from Recommender_System.algorithm.DeepFM.model import DeepFM_model
from Recommender_System.algorithm.train import train


if __name__ == '__main__':
    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k)

    dim = 16

    model1 = FM_model(n_user, n_item, dim=dim, l2=0)
    train(model1, train_data, test_data, topk_data, epochs=10)

    model2 = GMF_model(n_user, n_item, dim=dim, l2=0)
    train(model2, train_data, test_data, topk_data, epochs=10)

    model3 = LFM_model(n_user, n_item, dim=dim, l2=0)
    train(model3, train_data, test_data, topk_data, loss_object=tf.losses.MeanSquaredError(), epochs=10)

    model4 = MLP_model(n_user, n_item, dim=dim, layers=[dim, dim // 2, dim // 4], l2=0)
    train(model4, train_data, test_data, topk_data, epochs=10)

    model5, _, _ = NeuMF_model(n_user, n_item, gmf_dim=dim // 4, mlp_dim=dim, layers=[dim, dim // 2, dim // 4], l2=0)
    train(model5, train_data, test_data, topk_data, epochs=10)

    model6 = DeepFM_model(n_user, n_item, dim // 2, layers=[dim, dim, dim], l2=0)
    train(model6, train_data, test_data, topk_data, epochs=10)
