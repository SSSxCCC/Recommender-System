import Recommender_System.utility.gpu_memory_growth
from Recommender_System.algorithm.KGCN.tool import construct_undirected_kg, get_adj_list
from Recommender_System.algorithm.KGCN.model import KGCN_model
from Recommender_System.algorithm.KGCN.train import train
from Recommender_System.data import kg_loader, data_process
import tensorflow as tf


if __name__ == '__main__':
    n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.ml1m_kg_RippleNet, split_test_ratio=0.4)

    adj_entity, adj_relation = get_adj_list(construct_undirected_kg(kg), n_entity, neighbor_size=6)

    model = KGCN_model(n_user, n_entity, n_relation, adj_entity, adj_relation, dim=16, iter_size=2, l2=1e-7)

    train(model, train_data, test_data, topk_data, optimizer=tf.keras.optimizers.Adam(0.01), epochs=10, batch=512)
