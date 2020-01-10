from typing import List, Tuple
import tensorflow as tf
from Recommender_System.algorithm.train import RsCallback
from Recommender_System.utility.evaluation import TopkData
from Recommender_System.utility.decorator import logger


class _KgeCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.print('KGE: epoch=', epoch + 1, ', loss=', logs['loss'], sep='')


def _get_score_fn(model):
    @tf.function(experimental_relax_shapes=True)
    def _fast_model(inputs):
        return tf.squeeze(model(inputs))

    def _score_fn(inputs):
        inputs = {k: tf.constant(v, dtype=tf.int32) for k, v in inputs.items()}
        inputs['head_id'] = inputs['item_id']
        return _fast_model(inputs).numpy()
    return _score_fn


@logger('开始训练，', ('epochs', 'batch'))
def train(model_rs: tf.keras.Model, model_kge: tf.keras.Model, train_data: List[Tuple[int, int, int]],
          test_data: List[Tuple[int, int, int]], kg: List[Tuple[int, int, int]], topk_data: TopkData,
          optimizer_rs=None, optimizer_kge=None, kge_interval=3, epochs=100, batch=512):
    if optimizer_rs is None:
        optimizer_rs = tf.keras.optimizers.Adam()
    if optimizer_kge is None:
        optimizer_kge = tf.keras.optimizers.Adam()

    def xy(data):
        user_id = tf.constant([d[0] for d in data], dtype=tf.int32)
        item_id = tf.constant([d[1] for d in data], dtype=tf.int32)
        head_id = tf.constant([d[1] for d in data], dtype=tf.int32)
        label = tf.constant([d[2] for d in data], dtype=tf.float32)
        return {'user_id': user_id, 'item_id': item_id, 'head_id': head_id}, label

    def xy_kg(kg):
        item_id = tf.constant([d[0] for d in kg], dtype=tf.int32)
        head_id = tf.constant([d[0] for d in kg], dtype=tf.int32)
        relation_id = tf.constant([d[1] for d in kg], dtype=tf.int32)
        tail_id = tf.constant([d[2] for d in kg], dtype=tf.int32)
        label = tf.constant([0] * len(kg), dtype=tf.float32)
        return {'item_id': item_id, 'head_id': head_id, 'relation_id': relation_id, 'tail_id': tail_id}, label

    train_ds = tf.data.Dataset.from_tensor_slices(xy(train_data)).shuffle(len(train_data)).batch(batch)
    test_ds = tf.data.Dataset.from_tensor_slices(xy(test_data)).batch(batch)
    kg_ds = tf.data.Dataset.from_tensor_slices(xy_kg(kg)).shuffle(len(kg)).batch(batch)

    model_rs.compile(optimizer=optimizer_rs, loss='binary_crossentropy', metrics=['AUC', 'Precision', 'Recall'])
    model_kge.compile(optimizer=optimizer_kge, loss=lambda y_true, y_pre: y_pre)

    for epoch in range(epochs):
        model_rs.fit(train_ds, epochs=epoch + 1, verbose=0, validation_data=test_ds,
                     callbacks=[RsCallback(topk_data, _get_score_fn(model_rs))], initial_epoch=epoch)
        if epoch % kge_interval == 0:
            model_kge.fit(kg_ds, epochs=epoch + 1, verbose=0, callbacks=[_KgeCallback()], initial_epoch=epoch)
