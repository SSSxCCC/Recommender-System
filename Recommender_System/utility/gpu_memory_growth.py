"""
import此文件后将gpu设置为显存增量模式
"""

from tensorflow import config

gpus = physical_devices = config.list_physical_devices('GPU')
if len(gpus) == 0:
    print('当前没有检测到gpu，设置显存增量模式无效。')
for gpu in gpus:
    try:
        config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
