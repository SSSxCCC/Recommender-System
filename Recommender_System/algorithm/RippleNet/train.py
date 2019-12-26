import time
from typing import List, Tuple, Dict
import tensorflow as tf
from Recommender_System.utility.decorator import logger
from Recommender_System.utility.evaluation import TopkData
from Recommender_System.algorithm.train import log, topk


def fill_x_dict(x_dict, ripple_set, user_ids):
    ripple_sample = next(iter(ripple_set.values()))
    hop_size = len(ripple_sample)
    for hop in range(hop_size):
        x_dict['ripple_h_' + str(hop)] = tf.constant([ripple_set[u][hop][0] for u in user_ids], dtype=tf.int32)
        x_dict['ripple_r_' + str(hop)] = tf.constant([ripple_set[u][hop][1] for u in user_ids], dtype=tf.int32)
        x_dict['ripple_t_' + str(hop)] = tf.constant([ripple_set[u][hop][2] for u in user_ids], dtype=tf.int32)


def _get_score_fn(model, ripple_set):
    @tf.function(experimental_relax_shapes=True)
    def _fast_model(ui):
        return model(ui)[0]

    def score_fn(ui):
        x_dict = {'item_id': tf.constant(ui['item_id'], dtype=tf.int32)}
        fill_x_dict(x_dict, ripple_set, ui['user_id'])
        return _fast_model(x_dict).numpy()

    return score_fn


@logger('开始训练，', ('epochs', 'batch'))
def train(model: tf.keras.Model, train_data: List[Tuple[int, int, int]], test_data: List[Tuple[int, int, int]],
          topk_data: TopkData, ripple_set: Dict[int, List[Tuple[List[int], List[int], List[int]]]],
          optimizer=None, epochs=100, batch=512):
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()

    # 根据ripple_set获取hop_size和ripple_size
    #ripple_sample = next(iter(ripple_set.values()))
    #hop_size = len(ripple_sample)
    #ripple_size = len(ripple_sample[0][0])

    def xy(data):
        x_dict = {'item_id': tf.constant([d[1] for d in data], dtype=tf.int32)}
        fill_x_dict(x_dict, ripple_set, [d[0] for d in data])
        label = tf.constant([d[2] for d in data], dtype=tf.keras.backend.floatx())
        return x_dict, label

    train_ds = tf.data.Dataset.from_tensor_slices(xy(train_data)).shuffle(len(train_data)).batch(batch)
    test_ds = tf.data.Dataset.from_tensor_slices(xy(test_data)).batch(batch)

    loss_mean_metric = tf.keras.metrics.Mean()
    auc_metric = tf.keras.metrics.AUC()
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    kge_loss_mean_metric = tf.keras.metrics.Mean()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    score_fn = _get_score_fn(model, ripple_set)

    def train_model():
        def reset_metrics():
            for metric in [loss_mean_metric, auc_metric, precision_metric, recall_metric, kge_loss_mean_metric]:
                tf.py_function(metric.reset_states, [], [])

        def update_metrics(loss, label, score, kge_loss):
            loss_mean_metric.update_state(loss)
            auc_metric.update_state(label, score)
            precision_metric.update_state(label, score)
            recall_metric.update_state(label, score)
            kge_loss_mean_metric.update_state(kge_loss)

        def get_metric_results():
            return loss_mean_metric.result(), auc_metric.result(), precision_metric.result(), recall_metric.result(),\
                   kge_loss_mean_metric.result()

        @tf.function
        def train_batch(x, label):
            with tf.GradientTape() as tape:
                score, kge_loss, l2_loss = model(x)
                loss = loss_object(label, score) + kge_loss + l2_loss + sum(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            update_metrics(loss, label, score, kge_loss)

        @tf.function
        def test_batch(x, label):
            score, kge_loss, l2_loss = model(x)
            loss = loss_object(label, score) + kge_loss + l2_loss + sum(model.losses)
            update_metrics(loss, label, score, kge_loss)

        for epoch in tf.range(epochs):
            epoch_start_time = time.time()

            reset_metrics()
            for x, label in train_ds:
                train_batch(x, label)
            train_loss, train_auc, train_precision, train_recall, train_kge_loss = get_metric_results()

            reset_metrics()
            for x, label in test_ds:
                test_batch(x, label)
            test_loss, test_auc, test_precision, test_recall, test_kge_loss = get_metric_results()

            tf.py_function(log, [epoch, train_loss, train_auc, train_precision, train_recall,
                                 test_loss, test_auc, test_precision, test_recall], [])
            tf.print('train_kge_loss=', train_kge_loss, ', test_kge_loss=', test_kge_loss, sep='')
            tf.py_function(lambda: topk(topk_data, score_fn), [], [])

            print('epoch_time=', time.time() - epoch_start_time, 's', sep='')

    train_model()
