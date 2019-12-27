if __name__ == '__main__':
    import Recommender_System.utility.gpu_memory_growth
    from Recommender_System.data import data_loader, data_process
    from Recommender_System.algorithm.SLIM.tool import get_user_item_matrix
    from Recommender_System.algorithm.SLIM.model import SLIM
    from Recommender_System.algorithm.SLIM.train import train

    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k, negative_sample_ratio=0, split_test_ratio=0.125)

    A = get_user_item_matrix(n_user, n_item, train_data)

    model = SLIM(n_user, n_item, A)

    train(model, topk_data, epochs=1000)
