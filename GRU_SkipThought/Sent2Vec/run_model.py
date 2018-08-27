import tensorflow as tf
import numpy as np
import json,pickle,os,copy
from .input_data import DatasetManager
from .model import SkipThought
from utils import normalize



class Run_SkipThought:
    def __init__(self, vector_size,  **kwargs):

        self.save_dir = kwargs['save_dir']
        self.word_emb_size = kwargs['word_feature_size']
        self.total_epochs = kwargs['total_epochs']
        self.plot_every_steps = kwargs['plot_every_steps']
        self.update_lr_every_steps = kwargs['update_lr_every_steps']
        self.max_length = kwargs['max_length']  + 1  # take <EOS> <BOS> into account
        self.max_grad_norm = kwargs['max_grad_norm']
        self.init_learning_rate = kwargs['init_learning_rate']
        self.min_lr = kwargs['min_learning_rate']
        self.decay_rate = kwargs['decay_rate']
        self.batch_size = kwargs['batch_size']

        self.gru_hidden_size = self.sent_emb_size = vector_size
        self.output_filename = 'data_sentence_embedded_%dd'%self.sent_emb_size


        with open(kwargs['vocabulary_filepath'] , 'r') as f:
            self.vocab_dict = json.load(f)
            self.vocab_size = len(self.vocab_dict)

        # use this to translate word index into word to show the natural sentence
        with open(kwargs['reversed_vocabulary_filepath'] , 'r') as f:
            self.reverse_vocab_dict = json.load(f)

        # training_data are unlabelled posts
        # we want to train the skip-thought embedding by them
        with open(kwargs['training_data_filepath'] , 'r') as f:
            self.training_data = json.load(f)

        # testing_data are labelled posts
        # we want to evaluate the skip-thought embedding for them
        with open(kwargs['labelled_data_filepath'] , 'r') as f:
            self.testing_data = json.load(f)



    def input_normalize(self, data):
        """
        feed in batch size of tuple data which GRU can process
        """
        batch_encoder, batch_last, batch_next = data
        batchsize = len(batch_encoder)
        encoder_input = np.zeros(dtype=int, shape=(batchsize, self.max_length))
        encoder_input_seqlen = np.zeros(dtype=int, shape=(batchsize))

        decoder_input_last = np.zeros(dtype=int, shape=(batchsize, self.max_length))
        decoder_output_last = np.zeros(dtype=int, shape=(batchsize, self.max_length))
        decoder_input_seqlen_last = np.zeros(dtype=int, shape=(batchsize))

        decoder_input_next = np.zeros(dtype=int, shape=(batchsize, self.max_length))
        decoder_output_next = np.zeros(dtype=int, shape=(batchsize, self.max_length))
        decoder_input_seqlen_next = np.zeros(dtype=int, shape=(batchsize))

        for i, s in enumerate(batch_encoder):
            s = s.split()
            encoder_input_seqlen[i] = len(s)
            for j, word in enumerate(s):
                if word in self.vocab_dict:
                    encoder_input[i][j] = self.vocab_dict[word]
                else:
                    encoder_input[i][j] = self.vocab_dict["<UNK>"]

        for i, t in enumerate(batch_last):
            t = t.split()
            decoder_input_seqlen_last[i] = len(t)+1
            for j, word in enumerate(t):
                if word in self.vocab_dict:
                    decoder_input_last[i][j+1] = self.vocab_dict[word]
                    decoder_output_last[i][j] = self.vocab_dict[word]
                else:
                    decoder_input_last[i][j+1] = self.vocab_dict["<UNK>"]
                    decoder_output_last[i][j] = self.vocab_dict["<UNK>"]
            decoder_input_last[i][0] = self.vocab_dict["<BOS>"]
            decoder_output_last[i][len(t)] = self.vocab_dict["<EOS>"]

        for i, t in enumerate(batch_next):
            t = t.split()
            decoder_input_seqlen_next[i] = len(t)+1
            for j, word in enumerate(t):
                if word in self.vocab_dict:
                    decoder_input_next[i][j+1] = self.vocab_dict[word]
                    decoder_output_next[i][j] = self.vocab_dict[word]
                else:
                    decoder_input_next[i][j + 1] = self.vocab_dict["<UNK>"]
                    decoder_output_next[i][j] = self.vocab_dict["<UNK>"]
            decoder_input_next[i][0] = self.vocab_dict["<BOS>"]
            decoder_output_next[i][len(t)] = self.vocab_dict["<EOS>"]

        return encoder_input, encoder_input_seqlen, \
               decoder_input_last, decoder_output_last, decoder_input_seqlen_last, \
               decoder_input_next, decoder_output_next, decoder_input_seqlen_next

    def train(self):
        dataset = DatasetManager(self.training_data)
        model = SkipThought(name="SkipThought",
                            vocab_size=self.vocab_size,
                            word_dim=self.word_emb_size,
                            GRU_dim=self.gru_hidden_size,
                            max_grad_norm=self.max_grad_norm,
                            max_length=self.max_length,
                            learning_rate=self.init_learning_rate,
                            is_bidirectional=False)


        import time
        tstart = time.time()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            saver = tf.train.Saver()
            average_loss = 0
            all_losses = []
            total_step_one_epoch = dataset.size // self.batch_size
            print('%d steps in one epoch:'%total_step_one_epoch)
            print('SkipThought Model is training, it may take 1 hour for 1 epoch ... ')
            total_steps = total_step_one_epoch * self.total_epochs
            lr = self.init_learning_rate
            for step in range(1, total_steps+1):
                _encoder_input, _encoder_input_seqlen, \
                _decoder_input_last, _decoder_output_last, _decoder_input_seqlen_last, \
                _decoder_input_next, _decoder_output_next, _decoder_input_seqlen_next = \
                    self.input_normalize(dataset.next(self.batch_size))
                _,training_loss = sess.run([model.train_op, model.loss],
                                        feed_dict={
                                            model.encoder_input:_encoder_input,
                                            model.encoder_input_seqlen:_encoder_input_seqlen,
                                            model.decoder_input_last:_decoder_input_last,
                                            model.decoder_output_last:_decoder_output_last,
                                            model.decoder_input_seqlen_last:_decoder_input_seqlen_last,
                                            model.decoder_input_next: _decoder_input_next,
                                            model.decoder_output_next: _decoder_output_next,
                                            model.decoder_input_seqlen_next: _decoder_input_seqlen_next
                                        })
                average_loss += training_loss/self.plot_every_steps
                if step % self.plot_every_steps == 0:
                    _encoder_input, _encoder_input_seqlen, \
                    _decoder_input_last, _decoder_output_last, _decoder_input_seqlen_last, \
                    _decoder_input_next, _decoder_output_next, _decoder_input_seqlen_next = \
                        self.input_normalize(dataset.example)
                    pred, accu = sess.run([model.predict_next, model.accuracy],
                                                feed_dict={
                                                    model.encoder_input: _encoder_input,
                                                    model.encoder_input_seqlen: _encoder_input_seqlen,
                                                    model.decoder_input_last: _decoder_input_last,
                                                    model.decoder_output_last: _decoder_output_last,
                                                    model.decoder_input_seqlen_last: _decoder_input_seqlen_last,
                                                    model.decoder_input_next: _decoder_input_next,
                                                    model.decoder_output_next:_decoder_output_next,
                                                    model.decoder_input_seqlen_next: _decoder_input_seqlen_next
                                                })

                    print_result = "epoch %2d step %7d, loss=%0.4f accu=%0.4f" % \
                                   (step // total_step_one_epoch, step, average_loss, accu)
                    print(print_result)

                    if step % self.update_lr_every_steps == 0:
                        lr = max(self.min_lr, lr * self.decay_rate)
                        sess.run(model.update_lr, feed_dict={model.new_lr: lr})
                        print('learning rate is :', sess.run(model._lr))
                    if step % self.plot_every_steps == 0:
                        current_save_dir = os.path.join(self.save_dir,
                                                os.path.join("SkipThoughtModel_%dd"%self.sent_emb_size))
                        if not os.path.exists(current_save_dir):
                            os.makedirs(current_save_dir)
                        saver.save(sess,os.path.join(current_save_dir,'model.ckpt'))
                    all_losses.append(average_loss)
                    average_loss = 0
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(all_losses)
                plt.ylabel('loss')
                plt.xlabel('iteration steps')
                plt.title('traning loss for SkipThoughtModel_%dd'%self.sent_emb_size)
                plt.show()
            except:
                pass
        print('SkipThought Model training is over ... ')
        print('total training this_seed : %0.1f seconds' % (time.time()-tstart))




    def sample_search(self, input_text, sess, model):
        '''
        generate new sentence by sampling
        '''
        # probabilistic sample
        output_text_last = ""
        output_text_next = ""
        data = [[input_text], [output_text_last], [output_text_next]]

        #  predict last sentence
        last_token = "<BOS>"
        while len(output_text_last.split()) < self.max_length-1 and last_token != "<EOS>":
            _encoder_input, _encoder_input_seqlen, \
            _decoder_input_last, _decoder_output_last, _decoder_input_seqlen_last, \
            _decoder_input_next, _decoder_output_next, _decoder_input_seqlen_next = self.input_normalize(data)

            logits_last_ = sess.run(model.logits_last,
                                      feed_dict={
                                          model.encoder_input: _encoder_input,
                                          model.encoder_input_seqlen: _encoder_input_seqlen,
                                          model.decoder_input_last: _decoder_input_last,
                                          model.decoder_input_seqlen_last: _decoder_input_seqlen_last,
                                          model.decoder_input_next: _decoder_input_next,
                                          model.decoder_input_seqlen_next: _decoder_input_seqlen_next
                                      })

            last_logits = logits_last_[0][_decoder_input_seqlen_last[0] - 1]
            pro = np.exp(last_logits) / np.sum(np.exp(last_logits))
            word_id = np.random.choice(len(self.vocab_dict), 1, p=pro)[0]
            last_token = self.reverse_vocab_dict[str(word_id)]
            output_text_last += " %s" % last_token
            data = [[input_text], [output_text_last], [output_text_next]]

        #  predict next sentence
        last_token = "<BOS>"
        while len(output_text_next.split()) < self.max_length-1 and last_token != "<EOS>":
            _encoder_input, _encoder_input_seqlen, \
            _decoder_input_last, _decoder_output_last, _decoder_input_seqlen_last, \
            _decoder_input_next, _decoder_output_next, _decoder_input_seqlen_next = self.input_normalize(data)

            logits_next_ = sess.run(model.logits_next,
                                    feed_dict={
                                        model.encoder_input: _encoder_input,
                                        model.encoder_input_seqlen: _encoder_input_seqlen,
                                        model.decoder_input_last: _decoder_input_last,
                                        model.decoder_input_seqlen_last: _decoder_input_seqlen_last,
                                        model.decoder_input_next: _decoder_input_next,
                                        model.decoder_input_seqlen_next: _decoder_input_seqlen_next
                                    })

            last_logits = logits_next_[0][_decoder_input_seqlen_next[0] - 1]
            pro = np.exp(last_logits) / np.sum(np.exp(last_logits))
            word_id = np.random.choice(len(self.vocab_dict), 1, p=pro)[0]
            last_token = self.reverse_vocab_dict[str(word_id)]
            output_text_next += " %s" % last_token
            data = [[input_text], [output_text_last], [output_text_next]]

        return output_text_last, output_text_next

    def generating(self, story_mode=True):
        """
        if story_mode == False, generate the neighboring sentence,
        if story_mode == True, generate a story within 20 sentence
        """
        model = SkipThought(name="SkipThought",
                            vocab_size=self.vocab_size,
                            word_dim=self.word_emb_size,
                            GRU_dim=self.gru_hidden_size,
                            max_grad_norm=self.max_grad_norm,
                            max_length=self.max_length,
                            learning_rate=self.init_learning_rate,
                            is_bidirectional=False)

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            saver = tf.train.Saver()
            saver.restore(sess,
                       os.path.join(self.save_dir,
                                    os.path.join("SkipThoughtModel_%dd" % self.sent_emb_size),
                                    'model.ckpt'))
            print(' === ready to generate sentences === \n')
            if story_mode == True:
                while True:
                    input_text = input("Please input first sentence: ")
                    input_text = ' '.join(normalize(input_text).split()[:self.max_length-1])
                    print('turn 0 : %s\n'%input_text)
                    for turn in range(20):
                        output_text_last, output_text_next = self.sample_search(input_text, sess, model)
                        output_text_next = output_text_next.split()[:-1] # delete <EOS> tag
                        input_text = ' '.join(output_text_next[:self.max_length-1])
                        print('turn %d : %s\n'%(turn+1, input_text))
                    print('\n')
            else:
                while True:
                    input_text = input("Please input current sentence: ")

                    input_text = normalize(input_text)
                    # greedy_search(input_text, sess, model)
                    output_text_last, output_text_next =  self.sample_search(input_text, sess, model)
                    print("last sentence: ", output_text_last)
                    print("next sentence: ", output_text_next)
                    # beam_search(input_text, sess, model)

    def infer_vector(self, sents, model, sess):
        """
        generate sentence vectors using skip-thought embedding
        :param sents:a list of sentences of one doc
        """
        inputs = []
        for sent in sents:
            inputs.append([sent, "", ""]) # we dont use decoder sentence
        _encoder_input, _encoder_input_seqlen, \
        _decoder_input_last, _decoder_output_last, _decoder_input_seqlen_last, \
        _decoder_input_next, _decoder_output_next, _decoder_input_seqlen_next = \
            self.input_normalize(tuple(zip(*inputs)))

        sentvecs = sess.run(model.encoder_final_state,
                                    feed_dict={
                                        model.encoder_input: _encoder_input,
                                        model.encoder_input_seqlen: _encoder_input_seqlen,
                                        model.decoder_input_last: _decoder_input_last,
                                        model.decoder_output_last: _decoder_output_last,
                                        model.decoder_input_seqlen_last: _decoder_input_seqlen_last,
                                        model.decoder_input_next: _decoder_input_next,
                                        model.decoder_output_next: _decoder_output_next,
                                        model.decoder_input_seqlen_next: _decoder_input_seqlen_next
                                    })
        return sentvecs

    def testing(self):
        """
        generate the sentence vectors for labelled data,
        save it for GRU-Skipthought training
        """
        model = SkipThought(name="SkipThought",
                            vocab_size=self.vocab_size,
                            word_dim=self.word_emb_size,
                            GRU_dim=self.gru_hidden_size,
                            max_grad_norm=self.max_grad_norm,
                            max_length=self.max_length,
                            learning_rate=self.init_learning_rate,
                            is_bidirectional=False)


        sent_vectors_labelled_data = []
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            saver = tf.train.Saver()

            saver.restore(sess,
                          os.path.join(self.save_dir,
                                       os.path.join("SkipThoughtModel_%dd" % self.sent_emb_size),
                                       'model.ckpt'))
            for doc in self.testing_data:
                doc_vec = self.infer_vector(doc['split_sentences'], model, sess)
                doc['sentence_embeddings'] = doc_vec
                sent_vectors_labelled_data.append(copy.deepcopy(doc))
        with open(os.path.join(self.save_dir,
                               '%s.pkl'%self.output_filename),
                  'wb') as f:
            pickle.dump(sent_vectors_labelled_data, f)
