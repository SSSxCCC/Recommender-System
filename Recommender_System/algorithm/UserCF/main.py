from Recommender_System.algorithm.UserCF.tool import user_similarity, user_item_score
from Recommender_System.data import data_loader, data_process
from Recommender_System.utility.evaluation import topk_evaluate


if __name__ == '__main__':
    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k, negative_sample_ratio=0)

    W = user_similarity(train_data, n_user, n_item)
    scores = user_item_score(train_data, n_user, n_item, W, N=80)

    ks = [10, 36, 100]
    score_fn = lambda ui: [scores[u][i] for u, i in zip(ui['user_id'], ui['item_id'])]
    precisions, recalls = topk_evaluate(topk_data, score_fn, ks)
    for k, precision, recall in zip(ks, precisions, recalls):
        print('[k=%d, precision=%.3f%%, recall=%.3f%%, f1=%.3f%%]' %
              (k, 100 * precision, 100 * recall, 200 * precision * recall / (precision + recall)))
