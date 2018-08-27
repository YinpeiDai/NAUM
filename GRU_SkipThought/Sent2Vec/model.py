import tensorflow as tf
import numpy as np


class SkipThought:
    """
    uni-skip/bi-skip, using GRU, learnt with Adam
    """
    def __init__(self,
                 name,
                 vocab_size,
                 word_dim=100,  # word vector dimensionality
                 GRU_dim=300, # the number of GRU units
                 max_grad_norm=5,
                 max_length=50,
                 learning_rate=0.001,
                 is_bidirectional=False,
                 ):
        self.name = name
        with tf.device('/gpu:0'), tf.variable_scope(self.name,
                                                    initializer=tf.truncated_normal_initializer(0, 0.1)):
            # model placeholder
            self.encoder_input = tf.placeholder(dtype=tf.int32, shape=[None, max_length])
            self.encoder_input_seqlen = tf.placeholder(dtype=tf.int32, shape=[None])

            self.decoder_input_last = tf.placeholder(dtype=tf.int32, shape=[None, max_length])
            self.decoder_output_last = tf.placeholder(dtype=tf.int32, shape=[None, max_length])
            self.decoder_input_seqlen_last = tf.placeholder(dtype=tf.int32, shape=[None])

            self.decoder_input_next = tf.placeholder(dtype=tf.int32, shape=[None, max_length])
            self.decoder_output_next = tf.placeholder(dtype=tf.int32, shape=[None, max_length])
            self.decoder_input_seqlen_next = tf.placeholder(dtype=tf.int32, shape=[None])

            self.batch_size = tf.shape(self.encoder_input)[0]
            self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])

            with tf.variable_scope("embedding"):
                self.emb = tf.get_variable(name="emb",
                                           shape=[vocab_size, word_dim],
                                           dtype=tf.float32)

                self.encoder_input_emb = tf.nn.embedding_lookup(self.emb, self.encoder_input)
                self.decoder_input_last_emb = tf.nn.embedding_lookup(self.emb, self.decoder_input_last)
                self.decoder_input_next_emb = tf.nn.embedding_lookup(self.emb, self.decoder_input_next)


            # create GRU cell
            def one_cell():
                cell = tf.nn.rnn_cell.GRUCell(
                    num_units=GRU_dim,
                    kernel_initializer=tf.orthogonal_initializer(),
                )
                return cell

            with tf.variable_scope("encoder"):
                if not is_bidirectional:
                    self.encoder_cell = one_cell()
                    self.encoder_init_state = self.encoder_cell.zero_state(
                        batch_size=self.batch_size,
                        dtype=tf.float32)
                    _, self.encoder_final_state = tf.nn.dynamic_rnn(
                        cell=self.encoder_cell,
                        inputs=self.encoder_input_emb,
                        sequence_length=self.encoder_input_seqlen,
                        initial_state=self.encoder_init_state,
                        dtype=tf.float32,
                        swap_memory=True
                    )
                else:
                    self.encoder_cell_fw = one_cell()
                    self.encoder_cell_bw = one_cell()
                    self.encoder_init_state_fw = self.encoder_cell_fw.zero_state(
                        batch_size=self.batch_size,
                        dtype=tf.float32)
                    self.encoder_init_state_bw = self.encoder_cell_bw.zero_state(
                        batch_size=self.batch_size,
                        dtype=tf.float32)
                    _, self.encoder_output_states = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=self.encoder_cell_fw,
                        cell_bw=self.encoder_cell_bw,
                        inputs=self.encoder_input_emb,
                        sequence_length=self.encoder_input_seqlen,
                        initial_state_fw=self.encoder_init_state_fw,
                        initial_state_bw=self.encoder_init_state_bw,
                        swap_memory=True
                    )
                    self.encoder_final_state = tf.concat(self.encoder_output_states, 1)

            with tf.variable_scope("decoder_last"):
                # decoder for the last sentence
                self.decoder_inputs_last = tf.concat(
                    [self.decoder_input_last_emb,
                     tf.tile(tf.expand_dims(self.encoder_final_state, 1), [1, max_length, 1])],
                    axis=2)
                self.decoder_cell_last = one_cell()
                self.decoder_init_state_last = self.decoder_cell_last.zero_state(
                    batch_size=self.batch_size,
                    dtype=tf.float32)
                self.outputs_last, _ = tf.nn.dynamic_rnn(
                    cell=self.decoder_cell_last,
                    inputs=self.decoder_inputs_last,
                    sequence_length=self.decoder_input_seqlen_last,
                    initial_state=self.decoder_init_state_last,
                    dtype=tf.float32,
                    swap_memory=True
                )

            with tf.variable_scope("decoder_next"):
                self.decoder_inputs_next = tf.concat(
                    [self.decoder_input_next_emb,
                     tf.tile(tf.expand_dims(self.encoder_final_state, 1), [1, max_length, 1])],
                    axis=2)
                self.decoder_cell_next = one_cell()
                self.decoder_init_state_next = self.decoder_cell_next.zero_state(
                    batch_size=self.batch_size,
                    dtype=tf.float32)
                self.outputs_next, _ = tf.nn.dynamic_rnn(
                    cell=self.decoder_cell_next,
                    inputs=self.decoder_inputs_next,
                    sequence_length=self.decoder_input_seqlen_next,
                    initial_state=self.decoder_init_state_next,
                    dtype=tf.float32,
                    swap_memory=True
                )

            with tf.variable_scope("projection"):
                # using get_variable to shard variables
                self.softmax_w = tf.get_variable('W', [GRU_dim, vocab_size], dtype=tf.float32)
                self.softmax_b = tf.get_variable('b', [vocab_size], dtype=tf.float32)
                # to shape [batch_size * step_num, hidden_size]

                self.logits_last = tf.reshape(
                    tf.matmul(tf.reshape(self.outputs_last, [-1, GRU_dim]), self.softmax_w) + self.softmax_b,
                    [-1, max_length, vocab_size])
                self.logits_next = tf.reshape(
                    tf.matmul(tf.reshape(self.outputs_next, [-1, GRU_dim]), self.softmax_w) + self.softmax_b,
                    [-1, max_length, vocab_size])

            with tf.variable_scope("loss"):
                self.loss_last = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_last,
                    labels=self.decoder_output_last)
                self.len_mask_last = tf.sequence_mask(
                    self.decoder_input_seqlen_last,
                    maxlen=max_length,
                    dtype=tf.float32
                )
                self.num_last = tf.count_nonzero(self.len_mask_last)
                self.loss_next = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_next,
                    labels=self.decoder_output_next)
                self.len_mask_next = tf.sequence_mask(
                    self.decoder_input_seqlen_next,
                    maxlen=max_length,
                    dtype=tf.float32
                )
                self.num_next = tf.count_nonzero(self.len_mask_next)
                self.loss = tf.divide(tf.reduce_sum(self.loss_last * self.len_mask_last),
                                      tf.cast(self.num_last, tf.float32)) + \
                            tf.divide(tf.reduce_sum(self.loss_next * self.len_mask_next),
                                      tf.cast(self.num_next, tf.float32))


            with tf.variable_scope("train"):
                # learning rate
                self._lr = tf.Variable(learning_rate, trainable=False, name='learning_rate')
                self.tvars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                self.optimizer = tf.train.AdamOptimizer(self._lr)
                self.grads = tf.gradients(self.loss, self.tvars)
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
                self.predict_last = tf.argmax(self.logits_last, 2)
                self.correct_pred_last = tf.equal(tf.cast(self.predict_last, tf.int32),
                                             self.decoder_output_last)
                self.predict_next = tf.argmax(self.logits_next, 2)
                self.correct_pred_next = tf.equal(tf.cast(self.predict_next, tf.int32),
                                             self.decoder_output_next)
                self.sum_accuracy = tf.divide(
                                    tf.reduce_sum(
                                        tf.cast(self.correct_pred_last, tf.float32) * self.len_mask_last
                                    ),
                                    tf.cast(self.num_last, tf.float32)) + tf.divide(
                                    tf.reduce_sum(
                                        tf.cast(self.correct_pred_next, tf.float32) * self.len_mask_next
                                    ),
                                    tf.cast(self.num_next, tf.float32)
                                )
                self.accuracy = self.sum_accuracy / 2













