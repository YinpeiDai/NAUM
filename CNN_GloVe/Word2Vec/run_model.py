'''
use the glove toolkit, change and run demo.sh
'''

import os
import stat
import pickle, json
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import rcParams
import matplotlib.pyplot as plt


class Run_GloVe:
    def __init__(self,
                 vector_size,
                 **kwargs
                 ):
        with open(kwargs['vocabulary_filepath'], 'r') as f:
            self.vocab_dict = json.load(f)


        self.main_dir = os.path.abspath('.') # main dir of this whole project
        self.training_data_filepath = os.path.join(self.main_dir, kwargs['training_data_filepath'])
        self.save_dir = kwargs['save_dir']
        self.word_emb_size = vector_size
        self.output_filename = 'GloVe_vectors_%dd'%self.word_emb_size

        self.glove_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'glove') # glove dir

        with open(os.path.join(self.glove_dir, 'demo.sh'), 'r') as shell_file:
            self.lines = shell_file.read()

        self.lines = self.lines.replace('CORPUS=training_data_filepath', 'CORPUS=%s'%self.training_data_filepath)
        self.lines = self.lines.replace('SAVE_FILE=vectors', 'SAVE_FILE=%s'%self.output_filename)
        self.lines = self.lines.replace('MAX_ITER=total_epochs', 'MAX_ITER=%d' % kwargs['total_epochs'])
        self.lines = self.lines.replace('VECTOR_SIZE=vector_size', 'VECTOR_SIZE=%s'%self.word_emb_size)

        self.new_shell_filepath = os.path.join(self.glove_dir, 'run_%dd.sh' % self.word_emb_size)
        with open(self.new_shell_filepath, 'w') as new_shell_file:
            new_shell_file.write(self.lines)
        os.chmod(self.new_shell_filepath,stat.S_IRWXU)

    def train(self):
        os.chdir(self.glove_dir)
        os.system('make')
        os.system('./run_%dd.sh'%self.word_emb_size)

        # change vectors txt into dict pickle file, and save in save_dir
        emb_dict = {}

        with open(self.output_filename+'.txt') as f:
            for line in f:
                splits = line.split()
                word, emb = splits[0], splits[1:]
                if word in self.vocab_dict:
                    emb_dict[word] = np.array(emb)
        os.chdir(self.main_dir)
        with open(os.path.join(self.save_dir, self.output_filename+'.pkl'), 'wb') as f:
            pickle.dump(emb_dict, f)

    def generating(self):
        print(' sorry, only SkipThought model can generate sentences ...')

    def testing(self):
        '''
        using TSNE for to visual the distribution of word vectors
        :return: picture
        '''
        with open(os.path.join(self.save_dir, self.output_filename+'.pkl'), 'rb') as f:
            vocab = pickle.load(f)

        self.reverse_dictionary = {}
        self.embeddings = np.zeros(shape=[len(vocab), self.word_emb_size], dtype=np.float32)
        count = 0
        for k, v in vocab.items():
            self.reverse_dictionary[count] = k
            for i, num in enumerate(v):
                self.embeddings[count, i] = float(num)
            count += 1
        plot_only = 500
        print('tsne is working ... ')
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(self.embeddings[:plot_only, :])
        labels = [self.reverse_dictionary[i] for i in range(plot_only)]
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom',
                         size=8)
        plt.show()


