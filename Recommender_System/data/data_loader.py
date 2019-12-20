import time
import os
from typing import List, Callable, Tuple

# 记下自己的路径，确保其它py文件调用时读数据路径正确
root_path = os.path.dirname(__file__)


def _read_ml(relative_path: str, separator: str) -> List[Tuple[int, int, int, int]]:
    data = []
    with open(os.path.join(root_path, relative_path), 'r') as f:
        for line in f.readlines():
            values = line.strip().split(separator)
            user_id, movie_id, rating, timestamp = int(values[0]), int(values[1]), int(values[2]), int(values[3])
            data.append((user_id, movie_id, rating, timestamp))
    return data


def _read_ml100k() -> List[Tuple[int, int, int, int]]:
    return _read_ml('ml-100k/u.data', '\t')


def _read_ml1m() -> List[Tuple[int, int, int, int]]:
    return _read_ml('ml-1m/ratings.dat', '::')


def _read_lastfm() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(root_path, 'lastfm-2k/user_artists.dat'), 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split('\t')
            user_id, artist_id, weight = int(values[0]), int(values[1]), int(values[2])
            data.append((user_id, artist_id, weight))
    return data


def _load_data(read_data_fn: Callable[[], List[tuple]], expect_length: int, expect_user: int, expect_item: int,
               data_name: str, user_name='用户', item_name='物品') -> List[tuple]:
    print('开始读数据', data_name, '。共', expect_length, '条数据，有',
          expect_user, '个', user_name, '，', expect_item, '个', item_name, '。', sep='')
    start_time = time.time()
    data = read_data_fn()
    n_user, n_item = len(set(d[0] for d in data)), len(set(d[1] for d in data))
    assert len(data) == expect_length and n_user == expect_user and n_item == expect_item
    print('（耗时', time.time() - start_time, '秒）', sep='')
    return data


def ml100k() -> List[Tuple[int, int, int, int]]:
    return _load_data(_read_ml100k, 100000, 943, 1682, 'ml100k', item_name='电影')


def ml1m() -> List[Tuple[int, int, int, int]]:
    return _load_data(_read_ml1m, 1000209, 6040, 3706, 'ml1m', item_name='电影')


def lastfm() -> List[Tuple[int, int, int]]:
    return _load_data(_read_lastfm, 92834, 1892, 17632, 'lastfm', item_name='艺术家')


# 测试数据读的是否正确
if __name__ == '__main__':
    data = ml1m()
