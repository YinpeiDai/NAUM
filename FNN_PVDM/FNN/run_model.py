import tensorflow as tf
import numpy as np
import pickle, os
from .input_data import DatasetManager
from .model import FNN_DocVec
from utils import F1_score


class Run_FNN_PVDM:
    def __init__(self,vector_size, **kwargs):

        self.save_dir = kwargs['save_dir']
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
        training_data_filepath = os.path.join(self.save_dir, 'data_document_embedded_%dd.pkl'%vector_size)
        with open(training_data_filepath , 'rb') as f:
            self.doc_data = pickle.load(f)

        self.doc_emb_size = vector_size



    def input_normalize(self, batch_input, batch_output):
        """
        feed data as proper input to model
        """
        batchsize = len(batch_output)
        inputs = np.zeros(shape=[batchsize, self.doc_emb_size], dtype=np.float32)
        outputs = np.array(batch_output, dtype=np.float32)
        for i, item in enumerate(batch_input):
            inputs[i] = item[0]
            outputs[i] = batch_output[i]
        return inputs, outputs

    def train(self, seed, super_category, sub_category, round_id, oversampling_ratio):

        # generate dataset manager
        dataset = DatasetManager(self.doc_data,
                                 super_category,
                                 sub_category,
                                 round_id,
                                 oversampling_ratio)


        current_save_dir = os.path.join(
            self.save_dir,
            'FNN_PVDM_%dd_Results' % self.doc_emb_size,
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
        model = FNN_DocVec(
            name="FNN_PVDM%dd_label_%s_round_%d_ratio_%d" %
                 (self.doc_emb_size, sub_category, round_id, oversampling_ratio),
            doc_emb=self.doc_emb_size,
            max_grad_norm=self.max_grad_norm,
            FNN_hidden_size=self.fnn_hidden_size,
            learning_rate=lr,
            l2_normalisation=self.l2_normalisation
        )
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            saver = tf.train.Saver()
            file = open(os.path.join(current_save_dir, 'results.txt'), 'w')
            best_test_F1 = 0
            for step in range(1, self.total_steps + 1):
                batch_input_list, batch_output_list = dataset.next_batch()
                inputs, outputs = self.input_normalize(batch_input_list, batch_output_list)
                _, training_loss, train_prob = sess.run([model.train_op, model.final_loss, model.prob],
                                               feed_dict={
                                                model.input: inputs,
                                                model.output: outputs,
                                                model.keep_prob : 0.8,
                                            })
                # print(train_prob)
                average_loss += training_loss / self.plot_every_steps
                if step % self.plot_every_steps == 0:
                    lr = max(self.min_learning_rate, lr * self.decay_rate)
                    sess.run(model.update_lr, feed_dict={model.new_lr:lr})
                    valid_input, valid_output \
                        = self.input_normalize(dataset.validset_input, dataset.validset_output)

                    valid_prob = sess.run(model.prob,
                                        feed_dict={
                                            model.input: valid_input,
                                            model.output: valid_output,
                                            model.keep_prob: 1.0,
                                        })

                    valid_F1, _ = F1_score(np.squeeze(valid_prob), valid_output)

                    test_input, test_output = \
                        self.input_normalize(dataset.testset_input, dataset.testset_output)

                    test_prob = sess.run(model.prob,
                                    feed_dict={
                                        model.input: test_input,
                                        model.output: test_output,
                                        model.keep_prob: 1.0,
                                    })
                    test_F1, test_metrics  = F1_score(np.squeeze(test_prob), test_output)
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
