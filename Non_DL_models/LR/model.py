import tensorflow as tf

class LogisticRgression:
    def __init__(self,
                 name,
                 input_size=10000,
                 max_grad_norm=5,
                 learning_rate=0.001,
                 l2_normalisation=0.1,
                 ):
        self.name = name
        with tf.device('/gpu:0'), tf.variable_scope(name_or_scope=self.name,
                                                    initializer=tf.truncated_normal_initializer(0, 0.01)):
            # model placeholder
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
            self.output = tf.placeholder(dtype=tf.float32, shape=[None])

            with tf.variable_scope("projection"):
                self.final_output_logits = tf.layers.dense(inputs=self.input,
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
                # print(self.predict.get_shape())
                self.correct = tf.equal(tf.cast(self.predict, tf.int32),
                                        tf.cast(tf.expand_dims(self.output, 1), tf.int32))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

