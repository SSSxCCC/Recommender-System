if __name__ == '__main__':
    import Recommender_System.utility.gpu_memory_growth
    from Recommender_System.data import data_loader, data_process
    from Recommender_System.algorithm.MLP.model import MLP_model
    from Recommender_System.algorithm.train import train

    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k)

    model = MLP_model(n_user, n_item, dim=32, layers=[64, 64, 64], l2=0, dropout=0.3)

    train(model, train_data, test_data, topk_data, epochs=30, batch=512)
