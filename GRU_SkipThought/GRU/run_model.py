import tensorflow as tf
import numpy as np
import pickle, os
from .input_data import DatasetManager
from .model import GRU_SkipThought
from utils import F1_score


class Run_GRU_SkipThought:
    def __init__(self, vector_size, **kwargs):

        self.save_dir = kwargs['save_dir']
        self.max_length = kwargs['max_length']
        self.gru_hidden_size = kwargs['gru_hidden_size']
        self.fnn_hidden_size = kwargs['fnn_hidden_size']
        self.plot_every_steps = kwargs['plot_every_steps']
        self.early_stopping_tolerance = kwargs['early_stopping_tolerance']
        self.max_grad_norm = kwargs['max_grad_norm']
        self.init_learning_rate = kwargs['init_learning_rate']
        self.min_learning_rate = kwargs['min_learning_rate']
        self.decay_rate = kwargs['decay_rate']
        self.total_steps = kwargs['total_steps']
        self.dropout_rate = kwargs['dropout_rate']
        self.l2_normalisation = kwargs['l2_normalisation']
        self.cross_validation = kwargs['cross_validation']
        self.batch_size = kwargs['batch_size']
        self.is_model_save = kwargs['is_model_save']

        # load traning data
        training_data_filepath = os.path.join(self.save_dir, 'data_sentence_embedded_%dd.pkl'%vector_size)
        with open(training_data_filepath, 'rb') as f:
            self.doc_data = pickle.load(f)

        self.sent_emb_size = vector_size

    def input_normalize(self, batch_input, batch_output):
        """
        feed data as proper input to model
        """
        batchsize = len(batch_output)
        input = np.zeros(shape=[batchsize, self.max_length, self.sent_emb_size], dtype=np.float32)
        input_seqlen = np.zeros(dtype=np.int32, shape=[batchsize])
        output = np.array(batch_output, dtype=np.float32)
        for batch in range(batchsize):
            length = np.shape(batch_input[batch][0])[0]
            input_seqlen[batch] = length
            input[batch, :length, :] = batch_input[batch][0]
        return input, input_seqlen, output


    def train(self, seed, super_category, sub_category, round_id, oversampling_ratio):
        """
        given a certain label, cross validation split, oversampling ratio
        train the GRU model, record the F1 score.
        :param super_category: 'emotions' 'thinking_errors' 'situations'
        :param sub_category: such as 'Blaming' ...refer to CBT_ontology
        :param round_id: cross validation split 1-10
        :param oversampling_ratio: 0,1,3,5,7
        """
        # generate dataset manager
        dataset = DatasetManager(self.doc_data,
                                 super_category,
                                 sub_category,
                                 round_id,
                                 oversampling_ratio)

        current_save_dir = os.path.join(
            self.save_dir,
            'GRU_Skipthought%dd_Results' % self.sent_emb_size,
            'seed%d' % seed,
            sub_category,
            'oversampling_ratio' + str(oversampling_ratio),
            'round' + str(round_id))

        if not os.path.exists(current_save_dir):
            os.makedirs(current_save_dir)
        best_valid_F1 = 0
        tolerance_count = 0
        average_loss = 0
        lr = self.init_learning_rate
        model = GRU_SkipThought(
            name="GRUSkipthought%dd_label_%s_round_%d_ratio_%d" %
                 (self.sent_emb_size, sub_category, round_id, oversampling_ratio),
            max_length=self.max_length,
            max_grad_norm=self.max_grad_norm,
            sent_emb_size=self.sent_emb_size,
            GRU_hidden_size=self.gru_hidden_size,
            FNN_hidden_size=self.fnn_hidden_size,
            learning_rate=lr,
            l2_normalisation = self.l2_normalisation
        )
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            saver = tf.train.Saver()
            file = open(os.path.join(current_save_dir, 'results.txt'), 'w')
            pred_test_F1 = 0
            for step in range(1, self.total_steps + 1):
                batch_input_list, batch_output_list = dataset.next_batch()
                train_batch_input, train_batch_input_seqlen, \
                train_batch_output = self.input_normalize(batch_input_list, batch_output_list)
                _, training_loss, train_prob = sess.run([model.train_op, model.final_loss, model.prob],
                                            feed_dict={
                                                model.input: train_batch_input,
                                                model.input_seqlen: train_batch_input_seqlen,
                                                model.output: train_batch_output,
                                                model.keep_prob : 0.8,
                                            })
                # print(train_prob)
                average_loss += training_loss / self.plot_every_steps
                if step % self.plot_every_steps == 0:
                    lr = max(self.min_learning_rate, lr * self.decay_rate)
                    sess.run(model.update_lr, feed_dict={model.new_lr:lr})
                    valid_batch_input, valid_batch_input_seqlen, \
                    valid_batch_output = self.input_normalize(dataset.validset_input, dataset.validset_output)

                    valid_prob = sess.run(model.prob,
                                        feed_dict={
                                            model.input: valid_batch_input,
                                            model.input_seqlen: valid_batch_input_seqlen,
                                            model.output: valid_batch_output,
                                            model.keep_prob: 1.0,
                                        })

                    valid_F1, _ = F1_score(np.squeeze(valid_prob), valid_batch_output)

                    test_input, test_input_seqlen, \
                    test_output = self.input_normalize(dataset.testset_input, dataset.testset_output)

                    test_prob = sess.run(model.prob,
                                    feed_dict={
                                        model.input: test_input,
                                        model.input_seqlen: test_input_seqlen,
                                        model.output: test_output,
                                        model.keep_prob: 1.0,
                                    })
                    test_F1, test_metrics = F1_score(np.squeeze(test_prob), test_output)
                    precision, recall, accu, TP, FP, TN, FN = test_metrics
                    print_result = "label %s round %2d step %5d, loss=%0.4f valid_F1=%0.4f test_F1=%0.4f\n" \
                                   "   other test_metrics: pre=%0.4f recall=%0.4f accu=%0.4f TP=%0.4f FP=%0.4f TN=%0.4f FN=%0.4f" % \
                                   (sub_category, round_id, step, average_loss, valid_F1, test_F1, precision, recall,
                                    accu, TP, FP, TN, FN)
                    print(print_result)
                    print()
                    file.writelines(print_result + '\n')
                    average_loss = 0
                    if valid_F1 >= best_valid_F1:
                        best_valid_F1 = valid_F1
                        best_test_F1 = test_F1
                        best_test_metric = test_metrics
                        best_test_prob = test_prob
                        tolerance_count = 0
                        if self.is_model_save == 'True':
                            saver.save(sess,
                                       os.path.join(current_save_dir,
                                                    os.path.join("model",
                                                                 'model.ckpt')))
                    else:
                        tolerance_count += 1
                    if tolerance_count > self.early_stopping_tolerance:
                        break
                    # stop trainig if too bad
                    if best_valid_F1 == 0 and step > 1200:
                        break
            precision, recall, accu, TP, FP, TN, FN = best_test_metric
            print_result = '=== best valid F1 score: %0.4f, test F1 score: %0.4f=== \n' \
                           '=== other test_metrics: pre=%0.4f recall=%0.4f accu=%0.4f TP=%0.4f FP=%0.4f TN=%0.4f FN=%0.4f===' \
                           % (best_valid_F1, best_test_F1, precision, recall, accu, TP, FP, TN, FN)
            print(print_result)
            file.writelines(print_result + '\n')
            file.writelines('predictions (1 means positive, 0 means negative):' + '\n')
            # print prediction for each test data (1/10 of the whole labelled data)
            for i, ID in enumerate(dataset.testset_ids):
                print_result = '%s %d' % (ID, best_test_prob[i] < 0.5)
                file.writelines(print_result + '\n')
            file.close()