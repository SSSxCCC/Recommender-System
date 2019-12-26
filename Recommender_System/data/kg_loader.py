import os
from typing import Dict, List, Tuple, Callable
from Recommender_System.data import data_loader, data_process
from Recommender_System.utility.decorator import logger

# 记下kg文件夹的路径，确保其它py文件调用时读文件路径正确
kg_path = os.path.join(os.path.dirname(__file__), 'kg')


@logger('开始读物品实体映射关系')
def _read_item_id2entity_id_file(relative_path: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    item_to_entity = {}
    entity_to_item = {}
    with open(os.path.join(kg_path, relative_path)) as f:
        for line in f.readlines():
            values = line.strip().split('\t')
            item_id = int(values[0])
            entity_id = int(values[1])
            item_to_entity[item_id] = entity_id
            entity_to_item[entity_id] = item_id
    return item_to_entity, entity_to_item


@logger('开始读知识图谱结构图：', ('keep_all_head',))
def _read_kg_file(relative_path: str, entity_id_old2new: Dict[int, int], keep_all_head=True) ->\
        Tuple[List[Tuple[int, int, int]], int, int]:
    n_entity = len(entity_id_old2new)
    relation_id_old2new = {}
    n_relation = 0
    kg = []
    with open(os.path.join(kg_path, relative_path)) as f:
        for line in f.readlines():
            values = line.strip().split('\t')
            head_old, relation_old, tail_old = int(values[0]), values[1], int(values[2])

            if head_old not in entity_id_old2new:
                if keep_all_head:
                    entity_id_old2new[head_old] = n_entity
                    n_entity += 1
                else:
                    continue
            head = entity_id_old2new[head_old]

            if tail_old not in entity_id_old2new:
                entity_id_old2new[tail_old] = n_entity
                n_entity += 1
            tail = entity_id_old2new[tail_old]

            if relation_old not in relation_id_old2new:
                relation_id_old2new[relation_old] = n_relation
                n_relation += 1
            relation = relation_id_old2new[relation_old]

            kg.append((head, relation, tail))

    return kg, n_entity, n_relation


@logger('----------开始载入带知识图谱的数据集：', ('kg_directory',), '----------带知识图谱的数据集载入完成', log_time=False)
def _read_data_with_kg(data_loader_fn: Callable[[], List[tuple]], kg_directory: str, negative_sample_threshold=0, keep_all_head=True) ->\
        Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], int, int, int, int]:
    old_item_to_old_entity, old_entity_to_old_item = _read_item_id2entity_id_file(os.path.join(kg_directory, 'item_id2entity_id.txt'))
    data = data_loader_fn()
    data = [d for d in data if d[1] in old_item_to_old_entity]  # 去掉知识图谱中不存在的物品
    data = data_process.negative_sample(data, threshold=negative_sample_threshold)
    data, n_user, n_item, _, item_id_old2new = data_process.neaten_id(data)
    entity_id_old2new = {old_entity: item_id_old2new[old_item] for old_entity, old_item in old_entity_to_old_item.items()}
    kg, n_entity, n_relation = _read_kg_file(os.path.join(kg_directory, 'kg.txt'), entity_id_old2new, keep_all_head)
    return data, kg, n_user, n_item, n_entity, n_relation


def ml1m_kg_MKR() -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], int, int, int, int]:
    return _read_data_with_kg(data_loader.ml1m, 'kg_ml1m&MKR', negative_sample_threshold=4, keep_all_head=False)


def lastfm_kg_MKR() -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], int, int, int, int]:
    return _read_data_with_kg(data_loader.lastfm, 'kg_lastfm&MKR', keep_all_head=False)


def ml1m_kg_RippleNet() -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], int, int, int, int]:
    return _read_data_with_kg(data_loader.ml1m, 'kg_ml1m&RippleNet', negative_sample_threshold=4)


def ml20_kg_KGCN() -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], int, int, int, int]:
    return _read_data_with_kg(data_loader.ml20m, 'kg_ml20m&KGCN', negative_sample_threshold=4)


if __name__ == '__main__':
    data, kg, n_user, n_item, n_entity, n_relation = ml1m_kg_RippleNet()


