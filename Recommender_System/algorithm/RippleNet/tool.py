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


@logger('根据知识图谱有向图得到每个用户每跳的三元组，', ('hop_size', 'ripple_size'))
def get_ripple_set(hop_size: int, ripple_size: int, user_positive_item_list: Dict[int, List[int]],
                   kg_dict: Dict[int, List[Tuple[int, int]]]) -> Dict[int, List[Tuple[List[int], List[int], List[int]]]]:
    ripple_set = defaultdict(list)  # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]

    for user, positive_item_list in user_positive_item_list.items():
        for hop in range(hop_size):
            ripple_h, ripple_r, ripple_t = [], [], []

            if hop == 0:
                tails_of_last_hop = positive_item_list
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity_id in tails_of_last_hop:
                for relation_id, tail_id in kg_dict[entity_id]:
                    ripple_h.append(entity_id)
                    ripple_r.append(relation_id)
                    ripple_t.append(tail_id)

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for hop = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(ripple_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(ripple_h) < ripple_size
                indices = np.random.choice(len(ripple_h), size=ripple_size, replace=replace)
                ripple_h = [ripple_h[i] for i in indices]
                ripple_r = [ripple_r[i] for i in indices]
                ripple_t = [ripple_t[i] for i in indices]
                ripple_set[user].append((ripple_h, ripple_r, ripple_t))

    return ripple_set
