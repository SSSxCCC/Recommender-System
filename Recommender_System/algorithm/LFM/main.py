if __name__ == '__main__':
    import Recommender_System.utility.gpu_memory_growth
    import tensorflow as tf
    from Recommender_System.data import data_loader, data_process
    from Recommender_System.algorithm.LFM.model import LFM_model
    from Recommender_System.algorithm.train import train

    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k)

    model = LFM_model(n_user, n_item, dim=64, l2=1e-6)

    train(model, train_data, test_data, topk_data, loss_object=tf.losses.MeanSquaredError(), epochs=10, batch=512)
