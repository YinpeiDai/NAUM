import tensorflow as tf
import numpy as np
import json, pickle, os
from CNN_GloVe.CNN.input_data import DatasetManager
from CNN_GloVe.CNN.model import CNN_GloVe
from utils import F1_score

class Run_CNN_GloVe:
    '''
    run CNN-GloVe model
    '''
    def __init__(self, vector_size, **kwargs):

        self.save_dir = kwargs['save_dir']
        self.problem_max_length = kwargs['problem_max_length']
        self.negative_take_max_length = kwargs['negative_take_max_length']
        self.feature_map = kwargs['feature_map']
        self.filter_windows = kwargs['filter_windows']
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
        with open(kwargs['labelled_data_filepath'], 'r') as f:
            self.doc_data = json.load(f)

        # load GloVe word vectors
        glove_vectors_filepath = os.path.join(self.save_dir, 'GloVe_vectors_%dd.pkl'%vector_size)
        with open(glove_vectors_filepath, 'rb') as f:
            self.word_vectors = pickle.load(f)

        self.word_emb_size = vector_size

    def input_normalize(self, batch_input, batch_output):
        """
        feed data as proper input to model, such as the embedding matrix
        """
        batchsize = len(batch_output)
        input_p = np.zeros(shape=[batchsize, self.problem_max_length, self.word_emb_size], dtype=np.float32)
        input_n = np.zeros(shape=[batchsize, self.negative_take_max_length, self.word_emb_size], dtype=np.float32)
        output = np.array(batch_output, dtype=np.float32)
        for i, item in enumerate(batch_input):
            for j, word in enumerate(item[0].split()):
                if word in self.word_vectors:
                    input_p[i][j] = self.word_vectors[word]
            for j, word in enumerate(item[1].split()):
                if word in self. word_vectors:
                    input_n[i][j] = self.word_vectors[word]
        return input_p, input_n, output

    def train(self, seed, super_category, sub_category, round_id, oversampling_ratio):
        """
        given a certain label, cross validation split, oversampling ratio
        train the CNN model, record the F1 score.
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
            'CNN_GloVe_%dd_Results' % self.word_emb_size,
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
        model = CNN_GloVe(
            name="CNNGloVe%dd_label_%s_split_%d_ratio_%d" %
                 (self.word_emb_size, sub_category, round_id, oversampling_ratio),
            max_length_p=self.problem_max_length,
            max_length_n=self.negative_take_max_length,
            max_grad_norm=self.max_grad_norm,
            word_emb_size=self.word_emb_size,
            FNN_hidden_size=self.fnn_hidden_size,
            learning_rate=lr,
            feature_map=self.feature_map,
            filter_windows=self.filter_windows,
            l2_normalisation=self.l2_normalisation
        )
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            saver = tf.train.Saver()
            file = open(os.path.join(current_save_dir, 'results.txt'), 'w')
            for step in range(1, self.total_steps + 1):
                batch_input_list, batch_output_list = dataset.next_batch()
                input_p, input_n, output = self.input_normalize(batch_input_list, batch_output_list)
                _, training_loss, train_prob = sess.run([model.train_op, model.final_loss, model.prob],
                                            feed_dict={
                                                model.input_p: input_p,
                                                model.input_n: input_n,
                                                model.output: output,
                                                model.keep_prob : 0.8,
                                            })
                average_loss += training_loss / self.plot_every_steps
                if step % self.plot_every_steps == 0:
                    lr = max(self.min_learning_rate, lr * self.decay_rate)
                    sess.run(model.update_lr, feed_dict={model.new_lr:lr})
                    valid_input_p, valid_input_n, valid_output \
                        = self.input_normalize(dataset.validset_input, dataset.validset_output)

                    valid_prob = sess.run(model.prob,
                                        feed_dict={
                                            model.input_p: valid_input_p,
                                            model.input_n: valid_input_n,
                                            model.output: valid_output,
                                            model.keep_prob: 1.0,
                                        })
                    valid_F1, _ = F1_score(np.squeeze(valid_prob), valid_output)

                    test_input_p, test_input_n, \
                    test_output = self.input_normalize(dataset.testset_input, dataset.testset_output)

                    test_prob = sess.run(model.prob,
                                    feed_dict={
                                        model.input_p: test_input_p,
                                        model.input_n: test_input_n,
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
                print_result = '%s %d' %(ID, best_test_prob[i]<0.5)
                file.writelines(print_result + '\n')
            file.close()