import os
from typing import List, Callable, Tuple
from Recommender_System.utility.decorator import logger

# 记下ds文件夹的路径，确保其它py文件调用时读文件路径正确
ds_path = os.path.join(os.path.dirname(__file__), 'ds')


def _read_ml(relative_path: str, separator: str) -> List[Tuple[int, int, int, int]]:
    data = []
    with open(os.path.join(ds_path, relative_path), 'r') as f:
        for line in f.readlines():
            values = line.strip().split(separator)
            user_id, movie_id, rating, timestamp = int(values[0]), int(values[1]), int(values[2]), int(values[3])
            data.append((user_id, movie_id, rating, timestamp))
    return data


def _read_ml100k() -> List[Tuple[int, int, int, int]]:
    return _read_ml('ml-100k/u.data', '\t')


def _read_ml1m() -> List[Tuple[int, int, int, int]]:
    return _read_ml('ml-1m/ratings.dat', '::')


def _read_ml20m() -> List[Tuple[int, int, float, int]]:
    data = []
    with open(os.path.join(ds_path, 'ml-20m/ratings.csv'), 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split(',')
            user_id, movie_id, rating, timestamp = int(values[0]), int(values[1]), float(values[2]), int(values[3])
            data.append((user_id, movie_id, rating, timestamp))
    return data


def _read_lastfm() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(ds_path, 'lastfm-2k/user_artists.dat'), 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split('\t')
            user_id, artist_id, weight = int(values[0]), int(values[1]), int(values[2])
            data.append((user_id, artist_id, weight))
    return data


def _read_book_crossing() -> List[Tuple[int, str, int]]:
    data = []
    with open(os.path.join(ds_path, 'Book-Crossing/BX-Book-Ratings.csv'), 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split(';')
            user_id, book_id, rating = int(values[0][1:-1]), values[1][1:-1], int(values[2][1:-1])
            data.append((user_id, book_id, rating))
    return data


@logger('开始读数据，', ('data_name', 'expect_length', 'expect_user', 'expect_item'))
def _load_data(read_data_fn: Callable[[], List[tuple]], expect_length: int, expect_user: int, expect_item: int,
               data_name: str) -> List[tuple]:
    data = read_data_fn()
    n_user, n_item = len(set(d[0] for d in data)), len(set(d[1] for d in data))
    assert len(data) == expect_length, data_name + ' length ' + str(len(data)) + ' != ' + str(expect_length)
    assert n_user == expect_user, data_name + ' user ' + str(n_user) + ' != ' + str(expect_user)
    assert n_item == expect_item, data_name + ' item ' + str(n_item) + ' != ' + str(expect_item)
    return data


def ml100k() -> List[Tuple[int, int, int, int]]:
    return _load_data(_read_ml100k, 100000, 943, 1682, 'ml100k')


def ml1m() -> List[Tuple[int, int, int, int]]:
    return _load_data(_read_ml1m, 1000209, 6040, 3706, 'ml1m')


def ml20m() -> List[Tuple[int, int, float, int]]:
    return _load_data(_read_ml20m, 20000263, 138493, 26744, 'ml20m')


def lastfm() -> List[Tuple[int, int, int]]:
    return _load_data(_read_lastfm, 92834, 1892, 17632, 'lastfm')


def book_crossing() -> List[Tuple[int, str, int]]:
    return _load_data(_read_book_crossing, 1149780, 105283, 340555, 'Book-Crossing')


# 测试数据读的是否正确
if __name__ == '__main__':
    data = book_crossing()
