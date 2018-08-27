import tensorflow as tf
class CNN_GloVe:
    def __init__(self,
                 name,
                 max_length_p=290,
                 max_length_n=120,
                 max_grad_norm=5,
                 word_emb_size=300,
                 feature_map=50,
                 filter_windows=(2,3,4),  # len(filter_windows) can only be 3
                 FNN_hidden_size=150,
                 learning_rate=0.001,
                 l2_normalisation=0.0001
                 ):
        self.name = name
        with tf.device('/gpu:0'), tf.variable_scope(name_or_scope=self.name,
                                                    initializer=tf.truncated_normal_initializer(0, 0.01)):
            # model placeholder
            self.input_p = tf.placeholder(dtype=tf.float32, shape=[None, max_length_p, word_emb_size])
            self.input_n = tf.placeholder(dtype=tf.float32, shape=[None, max_length_n, word_emb_size])
            self.output = tf.placeholder(dtype=tf.float32, shape=[None])
            self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])

            self.batch_size = tf.shape(self.input_p)[0]

            def conv_relu(inputs, filters, kernel, poolsize):
                conv = tf.layers.conv1d(
                    inputs=inputs,
                    filters=filters,
                    kernel_size=kernel,
                    strides=1,
                    padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0, 0.01)
                )
                pool = tf.layers.max_pooling1d(
                    inputs=conv,
                    pool_size=poolsize,
                    strides=1,
                )
                _pool = tf.squeeze(pool, [1])
                return _pool
            def cnn(inputs, maxlength):
                with tf.variable_scope("winsize_1"):
                    conv_1 = conv_relu(inputs, feature_map, filter_windows[0], maxlength)
                with tf.variable_scope("winsize_2"):
                    conv_2 = conv_relu(inputs, feature_map, filter_windows[1], maxlength)
                with tf.variable_scope("winsize_3"):
                    conv_3 = conv_relu(inputs, feature_map, filter_windows[2], maxlength)
                return tf.concat([conv_1, conv_2, conv_3], 1)

            with tf.variable_scope("CNN_output"):
                self.output_p = cnn(self.input_p, max_length_p)
            with tf.variable_scope("CNN_output", reuse=True):
                self.output_n = cnn(self.input_n, max_length_n)


            with tf.variable_scope("gating"):
                self.gate = tf.layers.dense(
                    inputs=tf.concat([self.output_p, self.output_n], 1),
                    units=3*feature_map,
                    activation=tf.nn.sigmoid,
                    kernel_initializer=tf.truncated_normal_initializer(0, 0.01)
                )
                self.feature = self.gate * self.output_p + (1-self.gate) * self.output_n


            with tf.variable_scope("projection"):
                self.hidden_layer = tf.layers.dense(inputs=self.feature,
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
                self.optimizer = tf.train.AdamOptimizer(self._lr)
                self.l2_loss = [tf.nn.l2_loss(v) for v in self.tvars]
                self.final_loss = tf.reduce_mean(self.loss) + l2_normalisation * tf.add_n(self.l2_loss)
                self.grads = tf.gradients(self.final_loss, self.tvars)
                # clip the gradient based on the global norm
                self.grads, _ = tf.clip_by_global_norm(self.grads, max_grad_norm)
                # create training operation
                self.train_op = self.optimizer.apply_gradients(
                    zip(self.grads, self.tvars)
                )
                # update learining rate
                self.new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
                self.update_lr = tf.assign(self._lr, self.new_lr)

            with tf.variable_scope("predict"):
                # predict
                self.prob = tf.nn.sigmoid(self.final_output_logits)
                self.predict = tf.argmin(tf.concat(values=[self.prob, 1-self.prob], axis=1),1)
                self.correct = tf.equal(tf.cast(self.predict, tf.int32),
                                        tf.cast(tf.expand_dims(self.output, 1), tf.int32))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
