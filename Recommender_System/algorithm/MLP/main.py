import Recommender_System.utility.gpu_memory_growth
from Recommender_System.data import data_loader, data_process
from Recommender_System.algorithm.MLP.model import MLP, MLP_model
from Recommender_System.algorithm.train import train


if __name__ == '__main__':
    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k)

    model = MLP_model(n_user, n_item, dim=32, layers=[64, 64, 64], l2=0)

    train(model, train_data, test_data, topk_data, epochs=100, batch=512)

    #model.save_weights('save/ml100k_40,[64,48,32],0.00001.h5')
