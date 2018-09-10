import tensorflow as tf
import numpy as np
import time

vocab_size = 1
wordvec_size = 1


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBmodel:
    def __init__(self, is_training, config, input_):
        with tf.variable_scope('scope'):
            embedding = tf.get_variable('embedding', [vocab_size, wordvec_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

            # dropout
            inputs = tf.nn.dropout(inputs, config.keep_prob)
            output, state = self._build_rnn_graph(inputs, config, is_training)

            softmax_w = tf.get_variable('softmax_w', shape=[wordvec_size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())


            # output 是啥形状
            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
            logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

            loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                input_.targets,
                tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
                average_across_timesteps=False,
                average_across_batch=True
            )
            self._cost = tf.reduce_sum(loss)
            self._final_state = state
        if not is_training:
            return

        # trainning
        self._lr = tf.Variable(0., trainable=False)  # lr的更新
        tvars = tf.trainable_variables()

        #gradient clip
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_grpah(self, inputs, config, is_training):
        def make_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                reuse=not is_training)
            # dropout
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell()

        # loop1: num_layers

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True) # __init__
        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state

        # loop2: num_steps
        outputs = []
        with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE):
            for time_step in range(self.num_steps):                        # __call__
                cell_output, state = cell(inputs[:, time_step, :], state)  # cell_output: [batch_size, hidden_size];
                outputs.append(cell_output)
        # numpy order, [batch_size, hidden_size*num_steps] ==> [batch_size*num_steps, hidden_size]
        # len([[steps], [steps], ...]) = batch_size
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    @property
    def initial_state(self):
        return self._initial_state


class TestConfig:
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


def run_epoch(session, model, eval_op=None):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    fetches = {'cost': model.cost,
               'final_state': model.final_state}
    if eval_op is not None:
        fetches['eval_op'] = eval_op
    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

    return np.exp(costs / iters)


#todo: importing_data, metagraph, dropout, MultiRNNCell, BasicLSTMCell


