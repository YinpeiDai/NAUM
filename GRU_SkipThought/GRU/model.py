'''
GRU + one layer DNN   sigmoid output
'''

import tensorflow as tf


class GRU_SkipThought:
    def __init__(self,
                 name,
                 max_length=24,
                 max_grad_norm=5,
                 sent_emb_size=300,
                 GRU_hidden_size=150,
                 FNN_hidden_size=150,
                 learning_rate=0.001,
                 l2_normalisation=0.0001):
        self.name = name
        with tf.device('/gpu:0'), tf.variable_scope(name_or_scope=self.name,
                                                    initializer= tf.truncated_normal_initializer(0, 0.01)):

            # model placeholder
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, max_length, sent_emb_size])
            self.input_seqlen = tf.placeholder(dtype=tf.int32, shape=[None])
            self.output = tf.placeholder(dtype=tf.float32, shape=[None])
            self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])

            self.batch_size = tf.shape(self.input)[0]

            self.GRUCell = tf.nn.rnn_cell.GRUCell(
                num_units=GRU_hidden_size,
                kernel_initializer=tf.orthogonal_initializer())
            self.GRU_init_state = self.GRUCell.zero_state(batch_size=self.batch_size,
                                                          dtype=tf.float32)

            with tf.variable_scope("RNN"):
                _, self.final_states = tf.nn.dynamic_rnn(
                    cell=self.GRUCell,
                    inputs=self.input,
                    sequence_length=self.input_seqlen,
                    initial_state=self.GRU_init_state,
                    dtype=tf.float32,
                    swap_memory=True
                )

            with tf.variable_scope("projection"):
                self.hidden_layer = tf.layers.dense(inputs=self.final_states,
                                                    units=FNN_hidden_size,
                                                    activation=tf.nn.tanh)
                self.hidden_layer_dropout = tf.nn.dropout(self.hidden_layer, self.keep_prob)
                self.final_output_logits = tf.layers.dense(inputs=self.hidden_layer_dropout,
                                                           units=1)


            with tf.variable_scope("loss"):
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.final_output_logits,
                    labels=tf.expand_dims(self.output, 1))


            with tf.variable_scope("train"):
                # learning rate
                self._lr = tf.Variable(learning_rate, trainable=False, name='learning_rate')
                self.tvars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                self.optimizer = tf.train.AdamOptimizer(self._lr)
                self.l2_loss = [tf.nn.l2_loss(v) for v in self.tvars]
                self.final_loss = tf.reduce_mean(self.loss) + l2_normalisation * tf.add_n(self.l2_loss)
                self.grads = tf.gradients(self.final_loss, self.tvars)
                # clip the gradient based on the global norm
                self.grads, _ = tf.clip_by_global_norm(self.grads, max_grad_norm)
                # create training operation
                self.train_op = self.optimizer.apply_gradients(
                    zip(self.grads, self.tvars),
                    global_step=self.global_step
                )
                # update learining rate
                self.new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
                self.update_lr = tf.assign(self._lr, self.new_lr)

            with tf.variable_scope("predict"):
                # predict
                self.prob = tf.nn.sigmoid(self.final_output_logits)
                self.predict = tf.argmin(tf.concat(values=[self.prob, 1-self.prob], axis=1),1)
                # print(self.predict.get_shape())
                self.correct = tf.equal(tf.cast(self.predict, tf.int32),
                                        tf.cast(tf.expand_dims(self.output, 1), tf.int32))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))








