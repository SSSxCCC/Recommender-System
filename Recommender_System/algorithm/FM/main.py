import Recommender_System.utility.gpu_memory_growth
from Recommender_System.data import data_loader, data_process
from Recommender_System.algorithm.FM.model import FM_model
from Recommender_System.algorithm.train import train
import tensorflow as tf


if __name__ == '__main__':
    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml1m, negative_sample_threshold=4, split_test_ratio=0.4)

    model = FM_model(n_user, n_item, dim=16, l2=1e-6)

    train(model, train_data, test_data, topk_data, optimizer=tf.keras.optimizers.Adam(0.01), epochs=30, batch=512)
