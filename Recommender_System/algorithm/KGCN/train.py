import time
from typing import List, Tuple
import tensorflow as tf
from Recommender_System.utility.decorator import logger
from Recommender_System.utility.evaluation import TopkData
from Recommender_System.algorithm.common import log, topk


def _get_score_fn(model):
    @tf.function
    def fast_model(ui):
        return model(ui)

    def get_score(ui, start, end, batch):
        scores = []
        for i in range(start, end, batch):
            ui = {k: tf.constant(v[i:i+batch], dtype=tf.int32) for k, v in ui.items()}
            scores.extend(fast_model(ui).numpy())
        return scores

    def score_fn(ui):
        scores = []
        end = length = len(ui['user_id'])
        batch = 1
        while length > 0:
            step = length % 10
            start = end - step * batch
            scores = get_score(ui, start, end, batch) + scores
            length //= 10
            end = start
            batch *= 10
        return scores

    return score_fn


@logger('开始训练，', ('epochs', 'batch'))
def train(model: tf.keras.Model, train_data: List[Tuple[int, int, int]], test_data: List[Tuple[int, int, int]],
          topk_data: TopkData, optimizer=None, epochs=100, batch=512):
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()

    def xy(data):
        user_id = tf.constant([d[0] for d in data], dtype=tf.int32)
        item_id = tf.constant([d[1] for d in data], dtype=tf.int32)
        label = tf.constant([d[2] for d in data], dtype=tf.keras.backend.floatx())
        return {'user_id': user_id, 'item_id': item_id}, label

    train_ds = tf.data.Dataset.from_tensor_slices(xy(train_data)).shuffle(len(train_data)).batch(batch)
    test_ds = tf.data.Dataset.from_tensor_slices(xy(test_data)).batch(batch)

    loss_mean_metric = tf.keras.metrics.Mean()
    auc_metric = tf.keras.metrics.AUC()
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    score_fn = _get_score_fn(model)

    #@tf.function
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
        def train_batch(ui, label):
            with tf.GradientTape() as tape:
                score = model(ui, training=True)
                loss = loss_object(label, score) + sum(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            update_metrics(loss, label, score)

        @tf.function
        def test_batch(ui, label):
            score = model(ui)
            loss = loss_object(label, score) + sum(model.losses)
            update_metrics(loss, label, score)

        for epoch in tf.range(epochs):
            epoch_start_time = time.time()

            reset_metrics()
            for ui, label in train_ds:
                train_batch(ui, label)
            train_loss, train_auc, train_precision, train_recall = get_metric_results()

            reset_metrics()
            for ui, label in test_ds:
                test_batch(ui, label)
            test_loss, test_auc, test_precision, test_recall = get_metric_results()

            tf.py_function(log, [epoch, train_loss, train_auc, train_precision, train_recall,
                                 test_loss, test_auc, test_precision, test_recall], [])
            tf.py_function(lambda: topk(topk_data, score_fn), [], [])

            print('epoch_time=', time.time() - epoch_start_time, 's', sep='')

    train_model()
