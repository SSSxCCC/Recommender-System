if __name__ == '__main__':
    import Recommender_System.utility.gpu_memory_growth
    from Recommender_System.algorithm.RippleNet.tool import get_user_positive_item_list, construct_directed_kg, get_ripple_set
    from Recommender_System.algorithm.RippleNet.model import RippleNet_model
    from Recommender_System.algorithm.RippleNet.train import train
    from Recommender_System.data import kg_loader, data_process
    import tensorflow as tf

    n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.ml1m_kg1m, negative_sample_threshold=4, split_ensure_positive=True)
    hop_size, ripple_size = 2, 32
    ripple_set = get_ripple_set(n_user, hop_size, ripple_size, get_user_positive_item_list(train_data), construct_directed_kg(kg))
    model = RippleNet_model(n_entity, n_relation, ripple_set, hop_size, ripple_size, dim=16, kge_weight=0.01, l2=1e-7)
    train(model, train_data, test_data, topk_data, optimizer=tf.keras.optimizers.Adam(0.01), epochs=10, batch=512)
    '''
    n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.bx_kg150k, split_ensure_positive=True)
    hop_size, ripple_size = 2, 32
    ripple_set = get_ripple_set(n_user, hop_size, ripple_size, get_user_positive_item_list(train_data), construct_directed_kg(kg))
    model = RippleNet_model(n_entity, n_relation, ripple_set, hop_size, ripple_size, dim=8, kge_weight=0.01, l2=1e-6)
    train(model, train_data, test_data, topk_data, optimizer=tf.keras.optimizers.Adam(0.001), epochs=10, batch=1024)
    '''