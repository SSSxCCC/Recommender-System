if __name__ == '__main__':
    import Recommender_System.utility.gpu_memory_growth
    from Recommender_System.algorithm.KGCN.tool import construct_undirected_kg, get_adj_list
    from Recommender_System.algorithm.KGCN.model import KGCN_model
    from Recommender_System.algorithm.KGCN.train import train as train2
    from Recommender_System.algorithm.KGNNLS.tool import get_interaction_table
    from Recommender_System.algorithm.KGNNLS.model import KGNNLS_model
    from Recommender_System.algorithm.KGNNLS.train import train
    from Recommender_System.data import kg_loader, data_process
    import tensorflow as tf

    n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.ml1m_kg_RippleNet)

    interaction_table = get_interaction_table(train_data, n_entity)
    neighbor_size = 16
    adj_entity, adj_relation = get_adj_list(construct_undirected_kg(kg), n_entity, neighbor_size)

    model = KGNNLS_model(n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, neighbor_size, iter_size=1, dim=32, l2=1e-7)
    train(model, train_data, test_data, topk_data, optimizer=tf.keras.optimizers.Adam(0.01), epochs=10, batch=512)

    model = KGCN_model(n_user, n_entity, n_relation, adj_entity, adj_relation, neighbor_size, iter_size=1, dim=32, l2=1e-7)
    train2(model, train_data, test_data, topk_data, optimizer=tf.keras.optimizers.Adam(0.01), epochs=10, batch=512)
