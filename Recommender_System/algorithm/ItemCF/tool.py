import time
import math
from typing import List


def item_similarity(train_data: list, n_user: int, n_item: int) -> List[List[float]]:
    print('开始计算每两个物品之间的相似度。')
    start_time = time.time()

    train_user_items = [[] for _ in range(n_user)]  # train_user_items[i]是用户i有过正反馈的所有物品列表
    N = [0 for _ in range(n_item)]  # N[i]是物品i被有过正反馈的数量
    for user_id, item_id, _ in train_data:
        train_user_items[user_id].append(item_id)
        N[item_id] += 1

    W = [[0 for _ in range(n_item)] for _ in range(n_item)]  # W[i][j]是物品i和j的共同被正反馈的数量（j>i）
    for items in train_user_items:
        for i in items:
            for j in items:
                if j > i:
                    W[i][j] += 1

    for i in range(n_item - 1):
        for j in range(i + 1, n_item):
            if W[i][j] != 0:
                W[i][j] /= math.sqrt(N[i] * N[j])
                W[j][i] = W[i][j]

    print('（耗时', time.time() - start_time, '秒）', sep='')
    return W


def user_item_score(train_data: list, n_user: int, n_item: int, W: List[List[float]], N=10) -> List[List[float]]:
    print('开始计算用户物品评分矩阵，', 'N=', N, sep='')
    start_time = time.time()

    # 得到训练集中每个用户所有有过正反馈物品集合
    train_user_items = [set() for _ in range(n_user)]
    for user_id, item_id, _ in train_data:
        train_user_items[user_id].add(item_id)

    # 得到每个物品与其最相似的N个物品集合
    most_similar_items = []
    for i in range(n_item):
        Wi = dict()  # Wi[j]是物品i和j之间的相似度
        for j in range(n_item):
            if W[i][j] != 0:
                Wi[j] = W[i][j]
        most_similar_items.append(set(x[0] for x in sorted(Wi.items(), key=lambda x: x[1], reverse=True)[:N]))

    scores = [[0 for _ in range(n_item)] for _ in range(n_user)]  # scores[u][i]是用户u对物品i的评分
    for user_id in range(n_user):
        user_item_set = train_user_items[user_id]
        for i in user_item_set:
            for j in most_similar_items[i]:
                if j not in user_item_set:
                    scores[user_id][j] += W[i][j]
        #for i in set(range(n_item)) - user_item_set:
        #    for j in user_item_set & most_similar_items[i]:
        #        scores[user_id][i] += W[i][j]

    print('（耗时', time.time() - start_time, '秒）', sep='')
    return scores