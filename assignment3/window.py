import tensorflow as tf

def make_windowed_data(data, start_vec, end_vec, window_size=1):
    features, labels = data
    _features = [start_vec]*window_size + features + [end_vec]*window_size
    L = len(labels)
    window_features = []
    for i in xrange(L):
        window_features.append(sum(_features[i:i + window_size * 2 + 1], []))
    _data = zip(window_features, labels)
    return _data


def test_make_windowed_data():
    data = ([[1, 9], [2, 9], [3, 8], [4, 8]], [1, 1, 4, 4])
    start_vec = [0, 0]
    end_vec = [0, 0]
    _data = make_windowed_data(data, start_vec, end_vec, window_size=1)
    print(_data)


class Model(object):
    def __init__(self):
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

    def add_placeholder(self):
        raise NotImplementedError

    def create_feed_dict(self):
        raise NotImplementedError

    def add_prediction_op(self):
        raise NotImplementedError

    def add_loss_op(self):
        raise NotImplementedError

    def add_trainning_op(self):
        raise NotImplementedError

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        raise NotImplementedError

    def predict_on_batch(self, sess, inputs_batch):
        raise NotImplementedError

    def build(self):
        self.add_placeholder()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op()
        self.train_op = self.add_trainning_op(self.loss)


class NERModel(Model):
    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(NERModel, self).__init__()
        self.helper = helper
        self.config = config
        self.report = report
        self.pretrained_embeddings = pretrained_embeddings
        self.build()


    def add_placeholder(self):
        n_features = self.config.n_window_features
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, n_features), name='input')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,), name='labels')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        feed_dict = {self.input_placeholder: inputs_batch,
                     self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        corpus_embeddings = tf.get_variable('embeddings', initializer=self.pretrained_embeddings)
        _embeddings = tf.nn.embedding_lookup(corpus_embeddings, self.input_placeholder)
        self.config.n_word_features = self.config.n_window_features*self.config.embed_size
        _embeddings = tf.reshape(_embeddings, (-1,  self.config.n_word_features))
        return _embeddings

    def add_prediction_op(self):
        x = self.add_embedding()
        _init_xavier = tf.contrib.layers.xavier_initializer
        _init_zero = tf.zeros_initializer
        dropout_rate = self.dropout_placeholder
        n_window_features = self.config.n_window_features
        ebd_size = self.config.embed_size
        n_classes = self.config.n_classes
        h_size = self.config.hidden_size
        n_word_features = self.config.n_word_features
        # parameters
        W = tf.get_variable('W', shape=(n_word_features, h_size), dtype=tf.float32, initializer=_init_xavier)
        b1 = tf.get_variable('b1', shape=(h_size,), dtype=tf.float32, initializer=_init_zero)
        U = tf.get_variable('U', shape=(h_size, n_classes), dtype=tf.float32, initializer=_init_xavier)
        b2 = tf.get_variable('b2', shape=(n_classes), dtype=tf.float32, initializer=_init_zero)

        h = tf.nn.relu(tf.matmul(x, W) + b1)
        h_drop = tf.nn.dropout(h, dropout_rate)
        pred = tf.matmul(h_drop, U) + b2
        return pred

    def add_loss_op(self, pred):
        _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder,
                                                               logits=pred)
        loss = tf.reduce_mean(_loss)
        return loss

    def add_trainning_op(self, loss):
        optimizer = tf.train.adam.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def preprocess_sequence_data(self, examples):
        start = self.helper.START
        end = self.helper.END
        window_size = self.config.window_size
        _data = make_windowed_data(examples, start, end, window_size=window_size)
        return _data

    def predict_on_batch(self, sess, inputs_batch):
        feed_dict = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed_dict)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed_dict = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                          dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss





if __name__ == "__main__":
    test_make_windowed_data()