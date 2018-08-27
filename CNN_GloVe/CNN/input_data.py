"""
Generate mini-batch training data for CNN
"""
import copy,random

class DatasetManager:
    """
    given label, provide cross validation data and minibatch
    cross validation 8:1:1
    oversampling positive : negative = 1:7 or 1:5 or 1:3 or 1:1 or None
    """
    def __init__(self,
                 data,
                 super_category,
                 sub_category,
                 round_id,
                 oversampling_ratio,
                 cross_validation = 10):
        '''
        :param data: training data for CNN-GloVe
        :param super_category: 'emotions' 'thinking_errors' 'situations'
        :param sub_category: such as 'Blaming' ...refer to CBT_ontology
        :param round_id: the id th cross validation split, range [1,10]
        :param oversampling_ratio:  0 1 3 5 7
        :param cross_validation: default 10
        '''
        self.data = data
        self.super_category = super_category
        self.sub_category = sub_category
        self.round_id = round_id-1
        self.sampling_ratio = oversampling_ratio
        # extract positive negative set
        self.dataset = {'positive':[], 'negative':[]}
        for item in self.data:
            if self.sub_category in item['label'][self.super_category]:
                self.dataset['positive'].append((item['problem'], item['negative_take'], item['id']))
            else:
                self.dataset['negative'].append((item['problem'], item['negative_take'], item['id']))
        # print('P:', len(self.dataset['positive']), 'N:', len(self.dataset['negative']))

        # this is for cross validation
        self.round_size_positive = len(self.dataset['positive']) // cross_validation
        self.round_size_negative = len(self.dataset['negative']) // cross_validation

        self.split_data = []
        round_i = 0
        for round_i in range(cross_validation):
            one_split = {}
            one_split['positive'] = self.dataset['positive'][round_i*self.round_size_positive:
                                                             (round_i+1) * self.round_size_positive]
            one_split['negative'] = self.dataset['negative'][round_i * self.round_size_negative:
                                                             (round_i+1) * self.round_size_negative]

            self.split_data.append(copy.deepcopy(one_split))

        #extra data is feeded into the training set
        extra = {}
        extra['positive'] = self.dataset['positive'][(round_i+1) * self.round_size_positive:]
        extra['negative'] = self.dataset['negative'][(round_i+1) * self.round_size_negative:]

        self.batch_id = 0
        self.batch_pos_id = 0
        self.batch_neg_id = 0

        self.trainset_ = copy.deepcopy(self.split_data)

        self.validset = copy.deepcopy(self.split_data[self.round_id])
        self.validset_input = self.validset['positive'] + self.validset['negative']
        self.validset_output = [0] * len(self.validset['positive']) + [1] * len(self.validset['negative'])

        self.testset = copy.deepcopy(self.split_data[(self.round_id + 1) % cross_validation])
        self.testset_input = self.testset['positive'] + self.testset['negative']
        self.testset_output = [0] * len(self.testset['positive']) + [1] * len(self.testset['negative'])
        self.testset_ids = [item[2] for item in self.testset_input]


        del self.trainset_[self.round_id]
        if self.round_id == cross_validation-1:
            del self.trainset_[0]
        else:
            del self.trainset_[self.round_id]

        self.trainset = {'positive':[], 'negative':[]}
        for item in self.trainset_:
            self.trainset['positive'].extend(item['positive'])
            self.trainset['negative'].extend(item['negative'])
        self.trainset['positive'].extend(extra['positive'])
        self.trainset['negative'].extend(extra['negative'])
        self.trainset['positive'].extend(self.validset['positive'])
        self.trainset['negative'].extend(self.validset['negative'])
        del self.trainset_

        self.ratio = len(self.trainset['negative']) / len(self.trainset['positive'])

        random.shuffle(self.trainset['positive'])
        random.shuffle(self.trainset['negative'])

        # This is for no sampling ratio
        self.training_set = []
        for item in self.trainset['positive']:
            self.training_set.append([item, 0])
        for item in self.trainset['negative']:
            self.training_set.append([item, 1])

        random.shuffle(self.training_set)


    def next_batch(self, batchsize=24):
        """
        oversampling positive : negative = 1:7 or 1:5 or 1:3 or 1:1 or None
        :param batchsize = 24
        """
        self.ratio = len(self.dataset['positive']) / len(self.dataset['negative'])
        if self.sampling_ratio in [1, 3, 5, 7] and self.ratio < 1 / self.sampling_ratio:
            self.batch_pos_size = batchsize // (self.sampling_ratio + 1)
            if self.batch_pos_id + self.batch_pos_size >= len(self.trainset['positive']):
                self.batch_pos_id = 0
                random.shuffle(self.trainset['positive'])

            self.batch_neg_size = batchsize - self.batch_pos_size
            if self.batch_neg_id + self.batch_neg_size >= len(self.trainset['negative']):
                self.batch_neg_id = 0
                random.shuffle(self.trainset['negative'])
            #
            batch_pos_data = self.trainset['positive'][self.batch_pos_id:self.batch_pos_id + self.batch_pos_size]
            batch_neg_data = self.trainset['negative'][self.batch_neg_id:self.batch_neg_id + self.batch_neg_size]

            batch_input_list = batch_pos_data + batch_neg_data
            batch_output_list = [0] * self.batch_pos_size + [1] * self.batch_neg_size

            self.batch_pos_id = self.batch_pos_id + self.batch_pos_size
            self.batch_neg_id = self.batch_neg_id + self.batch_neg_size
            return batch_input_list, batch_output_list

        else:
            if self.batch_id + batchsize >= len(self.training_set):
                self.batch_id = 0
                random.shuffle(self.training_set)

            batch_data = self.training_set[self.batch_id:self.batch_id + batchsize]
            self.batch_id = self.batch_id + batchsize

            batch_input_list = []
            batch_output_list = []

            for i, item in enumerate(batch_data):
                batch_input_list.append(item[0])
                batch_output_list.append(item[1])
            self.batch_id = self.batch_id + batchsize
        # print(batch_output_list)

        return batch_input_list, batch_output_list

