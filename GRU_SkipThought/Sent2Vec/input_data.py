import random

class DatasetManager:
    def __init__(self, text):
        """
        :param text: list, each item is a tuple of sentence (current, last, next)
        """
        self.batch_id = 0
        self.data = text
        self.size = len(self.data)
        random.shuffle(self.data)
        # this is only for evaluation, using a small part of data
        batch_encoder, batch_last, batch_next = zip(*self.data[:1000])
        self.example = (batch_encoder, batch_last, batch_next)

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id+batch_size>=len(self.data):
            self.batch_id = 0
            # random.shuffle(self.data)
        batch_data = self.data[self.batch_id:self.batch_id + batch_size]
        batch_encoder, batch_last, batch_next  = zip(*batch_data)
        self.batch_id = self.batch_id + batch_size
        return batch_encoder, batch_last, batch_next
