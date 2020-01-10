from Recommender_System.utility.decorator import logger
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np


@logger('得到每个用户在训练集上的正反馈物品集合')
def get_user_positive_item_list(train_data: List[Tuple[int, int, int]]) -> Dict[int, List[int]]:
    user_positive_item_list = defaultdict(list)
    for user_id, item_id, label in train_data:
        if label == 1:
            user_positive_item_list[user_id].append(item_id)
    return user_positive_item_list


@logger('根据知识图谱结构构建有向图')
def construct_directed_kg(kg: List[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    kg_dict = defaultdict(list)
    for head_id, relation_id, tail_id in kg:
        kg_dict[head_id].append((relation_id, tail_id))
    return kg_dict


@logger('根据知识图谱有向图得到每个用户每跳的三元组，', ('n_user', 'hop_size', 'ripple_size'))
def get_ripple_set(n_user: int, hop_size: int, ripple_size: int, user_positive_item_list: Dict[int, List[int]],
                   kg_dict: Dict[int, List[Tuple[int, int]]]) -> List[List[Tuple[List[int], List[int], List[int]]]]:
    ripple_set = [[] for _ in range(n_user)]  # user_id -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]

    for user_id, positive_item_list in user_positive_item_list.items():
        for hop in range(hop_size):
            ripple_h, ripple_r, ripple_t = [], [], []
            tails_of_last_hop = positive_item_list if hop == 0 else ripple_set[user_id][-1][2]

            for entity_id in tails_of_last_hop:
                for relation_id, tail_id in kg_dict[entity_id]:
                    ripple_h.append(entity_id)
                    ripple_r.append(relation_id)
                    ripple_t.append(tail_id)

            if len(ripple_h) == 0:  # 如果当前用户当前跳的实体关系集合是空的
                ripple_set[user_id].append(ripple_set[user_id][-1])  # 仅复制上一跳的集合
            else:  # 对当前跳随机采样固定大小的实体关系集合
                replace = len(ripple_h) < ripple_size
                indices = np.random.choice(len(ripple_h), size=ripple_size, replace=replace)
                ripple_h = [ripple_h[i] for i in indices]
                ripple_r = [ripple_r[i] for i in indices]
                ripple_t = [ripple_t[i] for i in indices]
                ripple_set[user_id].append((ripple_h, ripple_r, ripple_t))

    return ripple_set
