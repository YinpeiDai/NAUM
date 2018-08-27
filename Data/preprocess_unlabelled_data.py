'''
1. transfer unlabelled  data into suitable format for embedding training
2. build vocab dict
'''
import collections
import csv,random
import re,json,copy,os
from utils import normalize, sub_UNK

class UnlabelledDataParsing:
    def __init__(self,
                 filename,
                 vocab_size):
        self.all_words_data = []
        # load data
        print('loading the unlablled data, it may take a while ... ')
        self.unlabelled_posts = []
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for each_data in reader:
                if each_data[7] != 't' and each_data[7] != 'quarantined':
                    post = {}
                    post['id'] = each_data[0]
                    post['problem'] = normalize(each_data[3])
                    post['negative_take'] = normalize(each_data[4])
                    self.unlabelled_posts.append(copy.deepcopy(post))
                    self.all_words_data += post['problem'].split()
                    self.all_words_data += post['negative_take'].split()

        print('build vocabulary ... ')
        self.word_dictionary = self.generate_dict(vocab_size, self.all_words_data)
        print('     vocab size:', len(self.word_dictionary))


        # change low frequency word into <UNK>, also get the max length sents
        problem_max_length, negative_take_max_length = 0, 0
        for i, post in enumerate(self.unlabelled_posts):
            self.unlabelled_posts[i]['problem'] = sub_UNK(post['problem'], self.word_dictionary)
            self.unlabelled_posts[i]['negative_take'] = sub_UNK(post['negative_take'], self.word_dictionary)
            problem_max_length = max(problem_max_length,
                                     len(self.unlabelled_posts[i]['problem'].split()))
            negative_take_max_length = max(negative_take_max_length,
                                           len(self.unlabelled_posts[i]['negative_take'].split()))

        print('     problem contains max %d words'% problem_max_length)
        print('     negative take contains max %d words'% negative_take_max_length)


        for i, w in enumerate(self.all_words_data):
            if w not in self.word_dictionary:
                self.all_words_data[i] = '<UNK>'



        print('build training data for GloVe word vectors ... ')
        self.generate_training_data_for_GloVe()

        print('build training data for SkipThought sentence vectors ... ')
        self.generate_training_data_for_SkipThought()

        print('build training data for PVDM doc vectors ... ')
        self.generate_training_data_for_PVDM()


    def generate_dict(self, vocab_size, words_data):
        word_count = [['<UNK>', -1]]
        word_dictionary = dict()
        word_counter_dict = collections.Counter(words_data)
        # four special symbols: <UNK>, <BOS>, <EOS>, <NUM>
        word_count.extend(word_counter_dict.most_common(vocab_size-3))
        for word, _ in word_count:
            word_dictionary[word] = len(word_dictionary)
        unk_count = 0
        for word in words_data:
            if word not in word_dictionary:
                unk_count += 1
        word_count[0][1] = unk_count
        # these two mainly for skipthought embedding traning
        word_dictionary['<BOS>'] = len(word_dictionary)
        word_dictionary['<EOS>'] = len(word_dictionary)
        reversed_word_dictionary = dict(zip(word_dictionary.values(), word_dictionary.keys()))
        with open('Data/vocab_dict.json', 'w') as f:
            json.dump(word_dictionary, f)
        with open('Data/reversed_vocab_dict.json', 'w') as f:
            json.dump(reversed_word_dictionary, f)
        with open('Data/word_count.json', 'w') as f:
            json.dump(word_count, f)

        return word_dictionary


    def generate_training_data_for_GloVe(self):
        with open('Data/training_data_for_GloVe.txt', 'w') as f:
            f.write(' '.join(self.all_words_data))
        print('     Total training word number for GloVe:', len(self.all_words_data))


    def generate_training_data_for_SkipThought(self):
        max_length_doc = 0 # max sents one doc has
        extracted_data = []
        for post in self.unlabelled_posts:
            split_sentences = []
            for sent in re.split(r'[?!.]',post['problem']):
                sent_words = sent.split()
                sent_len = len(sent_words)
                sent = ' '.join(sent_words)
                if sent_len == 1:
                    if not re.match('^\s*(<NUM>|<UNK>|\d|\w|,)\s*$', sent):
                        split_sentences.append(sent)
                elif sent_len <= 50 and sent_len > 0:
                    split_sentences.append(sent)
                elif sent_len > 50:
                    cutsize = random.choice(list(range(20, 40)))
                    for i in range(sent_len // cutsize):
                        split_sentences.append(' '.join(sent_words[i * cutsize:(i + 1) * cutsize]))
            for sent in re.split(r'[?!.]',post['negative_take']):
                sent_words = sent.split()
                sent_len = len(sent_words)
                sent = ' '.join(sent_words)
                if sent_len == 1:
                    if not re.match('^\s*(<NUM>|<UNK>|\d|\w|,)\s*$', sent):
                        split_sentences.append(sent)
                elif sent_len <= 50 and sent_len > 0:
                    split_sentences.append(sent)
                elif sent_len > 50:
                    cutsize = random.choice(list(range(20, 40)))
                    for i in range(sent_len // cutsize):
                        split_sentences.append(' '.join(sent_words[i * cutsize:(i + 1) * cutsize]))
            extracted_data.append(copy.deepcopy({'id':post['id'], 'split_sentences':split_sentences}))

        # transform into (s_t, s_{t-1}, s_{t+1}) tuple form for skipthought emb training
        training_data = []
        for post in extracted_data:
            split_sentences = post['split_sentences']
            length_of_doc = len(split_sentences)
            if length_of_doc > max_length_doc: max_length_doc = length_of_doc
            for i in range(length_of_doc):
                training_data.append([split_sentences[i], split_sentences[i - 1], split_sentences[(i + 1) % length_of_doc]])

        print('     one doc contains max %d sents'% max_length_doc)

        with open('Data/training_data_for_SkipThought.json', 'w') as f:
            json.dump(training_data, f, indent=2)
        print('     Total training sentence number for SkipThought: ', len(training_data))


    def generate_training_data_for_PVDM(self):
        with open('Data/training_data_for_PVDM.json', 'w') as f:
            json.dump(self.unlabelled_posts, f, indent=2)
        print('     Total training doc number for PVDM: ', len(self.unlabelled_posts))

def main(**kwargs):
    filepath = kwargs['raw_unlabelled_data_dir']
    paser = UnlabelledDataParsing(
        os.path.join(filepath,'unlabelled_posts'),
        vocab_size=18000) # min count around 5

if __name__ == '__main__':
    import time
    tstart = time.time()
    main()
    print('extracting unlabelled posts cost %.1f seconds:' %(time.time() - tstart))
