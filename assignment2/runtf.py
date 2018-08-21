# context manager
# class: tf.Graph

from model import Model
import tensorflow as tf
from utils.general_utils import get_minibatches
import numpy as np


class Softmax(Model):

    def __init__(self, config):
        self.config = config
        self.build()


    def run_epoch(self, sess, inputs, labels):
        """
        run an epoch of training
        :param sess:
        :param inputs:
        :param labels:
        :return: average_loss
        """
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size):
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, input_batch, labels_batch)
        return total_loss / n_minibatches

    def fit(self, sess, inputs, labels):
        """
        fit model on data
        :param sess:
        :param inputs:
        :param labels:
        :return:
        """
        losses = []
        for epoch in range(self.config.n_epochs):
            average_loss = self.run_epoch(sess, inputs, labels)
            losses.append(average_loss)
        return losses



class Config:
    pass


config = Config()
inputs = np.randmom.rand(config.n_samples, config.n_features)
labels = np.zeros((config.n_samples, config.n_classes), dtype=np.int32)
labels[:, 0] = 1

with tf.Graph().as_default() as graph:
    model = Softmax(config)
    init_op = tf.global_variables_initializer()
graph.finalize()  # good practice

with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    losses = model.fit(sess, inputs, labels)





