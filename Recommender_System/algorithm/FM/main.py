if __name__ == '__main__':
    import Recommender_System.utility.gpu_memory_growth
    from Recommender_System.data import data_loader, data_process
    from Recommender_System.algorithm.FM.model import FM_model
    from Recommender_System.algorithm.train import train

    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k)

    model = FM_model(n_user, n_item, dim=16, l2=1e-6)

    train(model, train_data, test_data, topk_data, epochs=10, batch=512)
