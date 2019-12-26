import time
from typing import List, Tuple, Dict
import tensorflow as tf
from Recommender_System.utility.evaluation import TopkData
from Recommender_System.algorithm.train import log, topk
from Recommender_System.algorithm.RippleNet.train import fill_x_dict
from Recommender_System.utility.decorator import logger


def _get_score_fn(model, ripple_set):
    @tf.function
    def fast_model(x_dict):
        return model(x_dict)[0]

    def get_score(x_dict, start, end, batch):
        scores = []
        for i in range(start, end, batch):
            x_dict = {k: v[i:i+batch] for k, v in x_dict.items()}
            scores.extend(fast_model(x_dict).numpy())
        return scores

    def score_fn(ui):
        x_dict = {k: tf.constant(v, dtype=tf.int32) for k, v in ui.items()}
        fill_x_dict(x_dict, ripple_set, ui['user_id'])
        scores = []
        end = length = len(ui['user_id'])
        batch = 1
        while length > 0:
            step = length % 10
            start = end - step * batch
            scores = get_score(x_dict, start, end, batch) + scores
            length //= 10
            end = start
            batch *= 10
        return scores

    return score_fn


@logger('开始训练，', ('epochs', 'batch'))
def train(model: tf.keras.Model, train_data: List[Tuple[int, int, int]], test_data: List[Tuple[int, int, int]],
          topk_data: TopkData, ripple_set: Dict[int, List[Tuple[List[int], List[int], List[int]]]],
          optimizer=None, epochs=100, batch=512):
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()

    def xy(data):
        user_ids = [d[0] for d in data]
        x_dict = {'user_id': tf.constant(user_ids, dtype=tf.int32),
                  'item_id': tf.constant([d[1] for d in data], dtype=tf.int32)}
        fill_x_dict(x_dict, ripple_set, user_ids)
        label = tf.constant([d[2] for d in data], dtype=tf.keras.backend.floatx())
        return x_dict, label

    train_ds = tf.data.Dataset.from_tensor_slices(xy(train_data)).shuffle(len(train_data)).batch(batch)
    test_ds = tf.data.Dataset.from_tensor_slices(xy(test_data)).batch(batch)

    loss_mean_metric = tf.keras.metrics.Mean()
    auc_metric = tf.keras.metrics.AUC()
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    score_fn = _get_score_fn(model, ripple_set)

    def train_model():
        def reset_metrics():
            for metric in [loss_mean_metric, auc_metric, precision_metric, recall_metric]:
                tf.py_function(metric.reset_states, [], [])

        def update_metrics(loss, label, score):
            loss_mean_metric.update_state(loss)
            auc_metric.update_state(label, score)
            precision_metric.update_state(label, score)
            recall_metric.update_state(label, score)

        def get_metric_results():
            return loss_mean_metric.result(), auc_metric.result(), precision_metric.result(), recall_metric.result()

        @tf.function
        def train_batch(x, label):
            with tf.GradientTape() as tape:
                score, extra_loss = model(x, training=True)
                loss = loss_object(label, score) + extra_loss + sum(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            update_metrics(loss, label, score)

        @tf.function
        def test_batch(x, label):
            score, extra_loss = model(x)
            loss = loss_object(label, score) + extra_loss + sum(model.losses)
            update_metrics(loss, label, score)

        for epoch in tf.range(epochs):
            epoch_start_time = time.time()

            reset_metrics()
            for x, label in train_ds:
                train_batch(x, label)
            train_loss, train_auc, train_precision, train_recall = get_metric_results()

            reset_metrics()
            for x, label in test_ds:
                test_batch(x, label)
            test_loss, test_auc, test_precision, test_recall = get_metric_results()

            tf.py_function(log, [epoch, train_loss, train_auc, train_precision, train_recall,
                                 test_loss, test_auc, test_precision, test_recall], [])
            #print(model.get_layer('score').variables)
            tf.py_function(lambda: topk(topk_data, score_fn), [], [])
            print('epoch_time=', time.time() - epoch_start_time, 's', sep='')

    train_model()
