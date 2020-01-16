if __name__ == '__main__':
    import Recommender_System.utility.gpu_memory_growth
    from Recommender_System.algorithm.KGCN.tool import construct_undirected_kg, get_adj_list
    from Recommender_System.algorithm.KGCN.model import KGCN_model
    from Recommender_System.algorithm.KGCN.train import train
    from Recommender_System.algorithm.KGNNLS.tool import get_interaction_table
    from Recommender_System.algorithm.KGNNLS.model import KGNNLS_model
    from Recommender_System.data import kg_loader, data_process
    import tensorflow as tf

    n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.ml1m_kg1m, negative_sample_threshold=4)
    neighbor_size, iter_size, dim, l2, ls, aggregator, lr, epochs, batch = 16, 1, 16, 1e-7, 1., 'neighbor', 0.01, 10, 512
    
    #n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.ml20m_kg500k, negative_sample_threshold=4)
    #neighbor_size, iter_size, dim, l2, ls, aggregator, lr, epochs, batch = 16, 1, 32, 1e-7, 1., 'sum', 0.01, 10, 65536
    
    #n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.lastfm_kg15k)
    #neighbor_size, iter_size, dim, l2, ls, aggregator, lr, epochs, batch = 8, 1, 16, 4e-5, 0.1, 'sum', 0.001, 10, 128

    #n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.bx_kg150k)
    #neighbor_size, iter_size, dim, l2, ls, aggregator, lr, epochs, batch = 8, 2, 64, 1e-5, 0.5, 'sum', 1e-4, 10, 256

    interaction_table = get_interaction_table(train_data, n_entity)
    adj_entity, adj_relation = get_adj_list(construct_undirected_kg(kg), n_entity, neighbor_size)

    model = KGNNLS_model(n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, neighbor_size, iter_size, dim, l2, ls, aggregator)
    train(model, train_data, test_data, topk_data, tf.keras.optimizers.Adam(lr), epochs, batch)

    model = KGCN_model(n_user, n_entity, n_relation, adj_entity, adj_relation, neighbor_size, iter_size, dim, l2, aggregator)
    train(model, train_data, test_data, topk_data, tf.keras.optimizers.Adam(lr), epochs, batch)
