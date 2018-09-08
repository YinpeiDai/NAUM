'''
bog-of-words + svm
'''
import numpy as np
import json, os, pickle
from ..input_data import DatasetManager
from sklearn import svm
from utils import F1_score
from utils import meaningless_words

class Run_SVM_BOW:
    def __init__(self, **kwargs):
        self.save_dir = kwargs['save_dir']
        self.cross_validation = kwargs['cross_validation']
        self.batch_size = kwargs['batch_size']
        self.kernel = kwargs['kernel']
        self.gamma = kwargs['gamma']

        self.stopwords_list = meaningless_words()

        with open(kwargs['labelled_data_filepath'], 'r') as f:
            self.doc_data = json.load(f)
        with open(kwargs['vocabulary_filepath'], 'r') as f:
            word_vectors = json.load(f)

        self.word_list = list(word_vectors.keys())


    def input_normalize(self, batch_inputs, batch_outputs):
        """
        feed data as proper input to model
        """
        batchsize = len(batch_inputs)
        inputs = np.zeros(shape=[batchsize, len(self.word_list)], dtype=np.float32)
        for i, (sent_p, sent_n, ID) in enumerate(batch_inputs):
            words = sent_p.split() + sent_n.split()
            for w in words:
                if w in self.word_list and w not in self.stopwords_list:
                    inputs[i][self.word_list.index(w)] = 1
                else: # <UNK>
                    inputs[i][-1] = 1.0


        return inputs, batch_outputs

    def train(self, seed, super_category, sub_category, round_id, oversampling_ratio):
        # generate dataset manager
        dataset = DatasetManager(self.doc_data,
                                 super_category,
                                 sub_category,
                                 round_id,
                                 oversampling_ratio)
        current_save_dir = os.path.join(
            self.save_dir,
            'SVM_BOW_Results',
            'seed%d' % seed,
            sub_category,
            'oversampling_ratio' + str(oversampling_ratio),
            'round' + str(round_id))

        if not os.path.exists(current_save_dir):
            os.makedirs(current_save_dir)
        file = open(os.path.join(current_save_dir, 'results.txt'), 'w')
        if oversampling_ratio == 0: # no oversampling
            kernel_svm = svm.SVC(kernel=self.kernel, gamma=self.gamma)
        else: # oversampling ratio 1:1 1:3 1:5 1:7
            if dataset.ratio > oversampling_ratio:
                # print(dataset.ratio / oversampling_ratio)
                kernel_svm = svm.SVC(kernel=self.kernel, gamma=self.gamma,
                                     class_weight={0: dataset.ratio / oversampling_ratio, 1:1})
            else:
                kernel_svm = svm.SVC(kernel=self.kernel, gamma=self.gamma)


        # using the off-the-shelf toolkit, no early stopping
        batch_input, batch_output = dataset.train_valid_set_input, dataset.train_valid_set_output
        train_inputs, train_outputs = self.input_normalize(batch_input, batch_output)
        print('SVM is training, it may take a few minitues ... ')
        kernel_svm.fit(train_inputs, train_outputs)

        test_inputs, test_outputs = self.input_normalize(dataset.testset_input, dataset.testset_output)
        kernel_svm_prediction = kernel_svm.predict(test_inputs)
        f1_score, test_metric = F1_score(kernel_svm_prediction, test_outputs)
        precision, recall, accu, TP, FP, TN, FN = test_metric
        print_result = '=== test F1 score: %0.4f=== \n' \
                       '=== other metrics: pre=%0.4f recall=%0.4f accu=%0.4f TP=%0.4f FP=%0.4f TN=%0.4f FN=%0.4f===' \
                       % (f1_score, precision, recall, accu, TP, FP, TN, FN)
        print(print_result)
        file.writelines(print_result + '\n')
        file.writelines('predictions (1 means positive, 0 means negative):' + '\n')
        # print prediction for each test data (1/10 of the whole labelled data)
        for i, ID in enumerate(dataset.testset_ids):
            print_result = '%s %d' % (ID, 1-test_outputs[i])
            file.writelines(print_result + '\n')
        file.close()











