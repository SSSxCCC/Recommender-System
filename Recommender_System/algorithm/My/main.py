if __name__ == '__main__':
    import Recommender_System.utility.gpu_memory_growth
    from Recommender_System.algorithm.KGCN.tool import construct_undirected_kg, get_adj_list
    from Recommender_System.algorithm.RippleNet.tool import get_user_positive_item_list, construct_directed_kg, get_ripple_set
    from Recommender_System.algorithm.My.model import My_model
    from Recommender_System.algorithm.My.train import train
    from Recommender_System.data import kg_loader, data_process
    import tensorflow as tf

    n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.ml1m_kg_RippleNet, split_ensure_positive=True)

    adj_entity, adj_relation = get_adj_list(construct_undirected_kg(kg), n_entity, neighbor_size=16)

    hop_size, ripple_size = 2, 32
    ripple_set = get_ripple_set(hop_size, ripple_size, get_user_positive_item_list(train_data), construct_directed_kg(kg))

    model = My_model(n_user, n_entity, n_relation, adj_entity, adj_relation, dim=16, iter_size=1, hop_size=hop_size, ripple_size=ripple_size, l2=1e-7)
    train(model, train_data, test_data, topk_data, ripple_set, optimizer=tf.keras.optimizers.Adam(0.01), epochs=7, batch=512)

    from Recommender_System.algorithm.RippleNet.model import RippleNet_model
    from Recommender_System.algorithm.RippleNet.train import train as train2

    model2 = RippleNet_model(n_entity, n_relation, dim=16, hop_size=hop_size, ripple_size=ripple_size, kge_weight=0, l2=1e-7)
    train2(model2, train_data, test_data, topk_data, ripple_set, optimizer=tf.keras.optimizers.Adam(0.01), epochs=7, batch=512)
