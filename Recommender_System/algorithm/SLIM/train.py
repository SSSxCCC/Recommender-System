import tensorflow as tf
from Recommender_System.algorithm.train import topk
from Recommender_System.algorithm.SLIM.model import SLIM
from Recommender_System.utility.evaluation import TopkData
from Recommender_System.utility.decorator import logger


@logger('开始训练，', ('l12', 'epochs'))
def train(model: SLIM, topk_data: TopkData, l12=0.01, epochs=1000):
    optimizer = tf.keras.optimizers.Ftrl(l1_regularization_strength=l12, l2_regularization_strength=l12)
    score_fn = lambda ui: model({k: tf.constant(v, dtype=tf.int32) for k, v in ui.items()})

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = model.loss(training=True) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(1, epochs + 1):
        loss = train_step()
        if epoch == 1 or (epoch < 20 and epoch % 5 == 0) or (epoch < 100 and epoch % 20 == 0) or epoch % 100 == 0:
            print('epoch=', epoch, ', loss=', loss.numpy(), sep='')
            topk(topk_data, score_fn)
