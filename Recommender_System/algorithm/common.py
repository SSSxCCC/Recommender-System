from typing import List, Callable, Dict
from Recommender_System.utility.evaluation import TopkData, topk_evaluate


def log(epoch, train_loss, train_auc, train_precision, train_recall, test_loss, test_auc, test_precision, test_recall):
    print('epoch=%d, train_loss=%.5f, train_auc=%.5f, train_f1=%.5f, test_loss=%.5f, test_auc=%.5f, test_f1=%.5f' %
          (epoch + 1, train_loss, train_auc, 2. * train_precision * train_recall / (train_precision + train_recall),
           test_loss, test_auc, 2. * test_precision * test_recall / (test_precision + test_recall)))


def topk(topk_data: TopkData, score_fn: Callable[[Dict[str, List[int]]], List[float]], ks=[10, 36, 100]):
    precisions, recalls = topk_evaluate(topk_data, score_fn, ks)
    for k, precision, recall in zip(ks, precisions, recalls):
        print('[k=%d, precision=%.3f%%, recall=%.3f%%, f1=%.3f%%]' %
              (k, 100 * precision, 100 * recall, 200 * precision * recall / (precision + recall)), end='')
    print()
