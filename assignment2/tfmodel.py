# coding=utf-8

class Model:
    """
    1. build the tensorflow model graph
    2. run the graph: training, prediction
    """

# build the model
    def build(self):
        """
        build the dataflow graph
        :return:
        """
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        """
        添加占位函数
        :return:
        """
        pass

    def add_prediction_op(self):
        """
        mapping inputs into predictions
        add variables op
        :return: pred, predictions
        """
        pass

    def add_loss_op(self, pred):
        """
        loss op: compute loss
        :param pred:
        :return: loss
        """
        pass

    def add_training_op(self, loss):
        """
        train option
        create an optimizer and apply gradient to all trainable variables
        :param loss:
        :return: train_op
        """
        optimizer = None
        train_op = optimizer.minimize(loss)
        return train_op

# run the model

    # sess.run: run options or evaluate tensor
    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """
        输入数据
        :param inputs_batch: a batch of inputs
        :param labels_batch: a batch of labels
        :return: feed_dict: {placeholder: tensor}
        """
        pass
    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """
        run train
        :param sess:  to run options
        :param inputs_batch:
        :param labels_batch:
        :return:
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """
        run prediction
        :param sess: to run option or evaluate tensor
        :param inputs_batch: input dataset
        :return: predictions, tensor
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions