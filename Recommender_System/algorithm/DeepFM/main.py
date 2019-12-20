import Recommender_System.utility.gpu_memory_growth
from Recommender_System.data import data_loader, data_process
from Recommender_System.algorithm.DeepFM.model import DeepFM_model
from Recommender_System.algorithm.train import train


if __name__ == '__main__':
    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k)

    model = DeepFM_model(n_user, n_item, dim=8, layers=[16, 16, 16], l2=1e-5)

    train(model, train_data, test_data, topk_data, epochs=30)
