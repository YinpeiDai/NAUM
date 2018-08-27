'''
using the gensim toolkit
'''
import json,pickle,os, copy
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from gensim.test.utils import get_tmpfile


class Run_PVDM:
    def __init__(self,
                 vector_size,
                 **kwargs
                 ):
        # training_data are unlabelled posts
        # we want to train the PV-DM doc embedding by them
        with open(kwargs['training_data_filepath'], 'r') as f:
            self.training_data = json.load(f)

        # testing_data are labelled posts
        # we want to evaluate the PV-DM doc embedding for them
        with open(kwargs['labelled_data_filepath'], 'r') as f:
            self.testing_data = json.load(f)
        self.labelled_data_length = len(self.testing_data)


        # we merge them together to train doc vectors
        self.training_docs = []
        self.docID_set = set()
        # put the labelled data at first, and
        # we'll get their doc embeddings later
        for post in self.testing_data:
            if post['id'] not in self.docID_set:
                self.docID_set.add(post['id'])
                self.training_docs.append(post['problem']+' '+post['negative_take'])
        for post in self.training_data:
            if post['id'] not in self.docID_set:
                self.docID_set.add(post['id'])
                self.training_docs.append(post['problem']+' '+post['negative_take'])

        self.doc_emb_size = vector_size
        self.save_dir = kwargs['save_dir']
        self.output_filename = 'data_document_embedded_%dd'%self.doc_emb_size


    def train(self):
        print('%d doc to train in total'%len(self.docID_set))
        print('%d doc has labels'%self.labelled_data_length)
        docLabels = [i for i in range(len(self.training_docs))]
        docs = self.training_docs

        class LabeledLineSentence(object):
            def __init__(self, doc_list, labels_list):
               self.labels_list = labels_list
               self.doc_list = doc_list
            def __iter__(self):
                for idx, doc in enumerate(self.doc_list):
                    yield LabeledSentence(doc.split(),[self.labels_list[idx]])

        it = LabeledLineSentence(docs, docLabels)

        model = Doc2Vec(vector_size=self.doc_emb_size, window=10, min_count=1, workers=4)

        model.build_vocab(it)

        import time
        tstart = time.time()
        print('PVDM Model is training, it may take half an hour ... ')
        model.train(it, total_examples=model.corpus_count, epochs=100,
            start_alpha=0.03, end_alpha=0.0001)
        print('PVDM Model training is over ... ')
        print('total training this_seed : %0.1f seconds' % (time.time()-tstart))
        fname = get_tmpfile("PVDM_Model_%dd"%self.doc_emb_size)

        model.save(fname)

        # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    def testing(self):
        fname = get_tmpfile("PVDM_Model_%dd" % self.doc_emb_size)

        model = Doc2Vec.load(fname)
        doc_vectors_labelled_data = []
        for ID, doc in enumerate(self.testing_data):
            # take labelled data out, they are at the first
            doc['doc_embeddings'] = model[ID]

            # you can them too.
            # doc_vec = model.infer_vector(doc['problem'].split()+doc['negative_take'].split())
            doc_vectors_labelled_data.append(copy.deepcopy(doc))
        with open(os.path.join(self.save_dir,
                               '%s.pkl' % self.output_filename),
                  'wb') as f:
            pickle.dump(doc_vectors_labelled_data, f)


    def generating(self):
        print(' sorry, only SkipThought model can generate sentences ...')
