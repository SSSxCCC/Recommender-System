import time
from typing import List, Tuple
import tensorflow as tf
from Recommender_System.utility.decorator import logger
from Recommender_System.utility.evaluation import TopkData
from Recommender_System.algorithm.train import prepare_ds, get_score_fn
from Recommender_System.algorithm.common import log, topk


@logger('开始训练，', ('epochs', 'batch'))
def train(model: tf.keras.Model, train_data: List[Tuple[int, int, int]], test_data: List[Tuple[int, int, int]],
          topk_data: TopkData = None, optimizer=None, epochs=100, batch=512):
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()

    train_ds, test_ds = prepare_ds(train_data, test_data, batch)

    loss_mean_metric = tf.keras.metrics.Mean()
    auc_metric = tf.keras.metrics.AUC()
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    if topk_data:
        score_fn = get_score_fn(model)

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

    for epoch in range(epochs):
        epoch_start_time = time.time()

        reset_metrics()
        for ui, label in train_ds:
            train_batch(ui, label)
        train_loss, train_auc, train_precision, train_recall = get_metric_results()

        reset_metrics()
        for ui, label in test_ds:
            test_batch(ui, label)
        test_loss, test_auc, test_precision, test_recall = get_metric_results()

        log(epoch, train_loss, train_auc, train_precision, train_recall, test_loss, test_auc, test_precision, test_recall)
        if topk_data:
            topk(topk_data, score_fn)
        print('epoch_time=', time.time() - epoch_start_time, 's', sep='')
