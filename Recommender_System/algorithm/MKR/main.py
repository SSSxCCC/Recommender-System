if __name__ == '__main__':
    import Recommender_System.utility.gpu_memory_growth
    from tensorflow.keras.optimizers import Adam
    from Recommender_System.data import kg_loader, data_process
    from Recommender_System.algorithm.MKR.model import MKR_model
    from Recommender_System.algorithm.MKR.train import train

    n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.ml1m_kg20k, keep_all_head=False, negative_sample_threshold=4)
    model_rs, model_kge = MKR_model(n_user, n_item, n_entity, n_relation, dim=8, L=1, H=1, l2=1e-6)
    train(model_rs, model_kge, train_data, test_data, kg, topk_data, kge_interval=3,
          optimizer_rs=Adam(0.02), optimizer_kge=Adam(0.01), epochs=20, batch=4096)

    '''
    n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.lastfm_kg15k, keep_all_head=False)
    model_rs, model_kge = MKR_model(n_user, n_item, n_entity, n_relation, dim=4, L=2, H=1, l2=1e-6)
    train(model_rs, model_kge, train_data, test_data, kg, topk_data, kge_interval=2,
          optimizer_rs=Adam(1e-3), optimizer_kge=Adam(2e-4), epochs=10, batch=256)
    '''
    '''
    n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.bx_kg20k, keep_all_head=False)
    model_rs, model_kge = MKR_model(n_user, n_item, n_entity, n_relation, dim=8, L=1, H=1, l2=1e-6)
    train(model_rs, model_kge, train_data, test_data, kg, topk_data, kge_interval=2,
          optimizer_rs=Adam(2e-4), optimizer_kge=Adam(2e-5), epochs=10, batch=32)
    '''
