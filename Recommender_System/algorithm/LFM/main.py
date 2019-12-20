import Recommender_System.utility.gpu_memory_growth
import tensorflow as tf
from Recommender_System.data import data_loader, data_process
from Recommender_System.algorithm.LFM.model import LFM, LFM_model
from Recommender_System.algorithm.train import train


if __name__ == '__main__':
    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k)

    model = LFM(n_user, n_item, dim=64, l2=0)

    train(model, train_data, test_data, topk_data, loss_object=tf.losses.MeanSquaredError(), epochs=10, batch=512, execution='graph')
