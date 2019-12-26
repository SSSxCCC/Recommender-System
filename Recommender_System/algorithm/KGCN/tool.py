from Recommender_System.utility.decorator import logger
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np


@logger('根据知识图谱结构构建无向图')
def construct_undirected_kg(kg: List[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    kg_dict = defaultdict(list)
    for head_id, relation_id, tail_id in kg:
        kg_dict[head_id].append((relation_id, tail_id))
        kg_dict[tail_id].append((relation_id, head_id))  # 将知识图谱视为无向图
    return kg_dict


@logger('根据知识图谱无向图构建邻接表，', ('n_entity', 'neighbor_size'))
def get_adj_list(kg_dict: Dict[int, List[Tuple[int, int]]], n_entity: int, neighbor_size: int) ->\
        Tuple[List[List[int]], List[List[int]]]:
    adj_entity, adj_relation = [None for _ in range(n_entity)], [None for _ in range(n_entity)]
    for entity_id in range(n_entity):
        neighbors = kg_dict[entity_id]
        n_neighbor = len(neighbors)
        sample_indices = np.random.choice(range(n_neighbor), size=neighbor_size, replace=n_neighbor < neighbor_size)
        adj_relation[entity_id] = [neighbors[i][0] for i in sample_indices]
        adj_entity[entity_id] = [neighbors[i][1] for i in sample_indices]
    return adj_entity, adj_relation
