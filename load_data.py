import numpy as np
import utils

class DataSet(object):
    def __init__(self, fileslist, path):

        self._num_examples = len(fileslist)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._files = np.array(fileslist)
        self.animals = {'dog':0,'cat':1}
        self.path = path

    @property
    def files(self):
        return self._files
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def get_data(self, fbatch, path):
        images = np.empty([0,224,224,3])
        labels = np.empty([0])
        for f in fbatch:
            images = np.concatenate((images,utils.load_image(path+f).reshape((1,224,224,3))),axis =0)
            labels = np.concatenate((labels,[self.animals[f.split('.')[0]]]),axis=0)
        return images, labels#to_categorical(labels,2)
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            np.random.shuffle(self._files)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        #print 'start', start, 'end', end
        self._images, self._labels = self.get_data(self._files[start:end], self.path)
        return self._images, self._labels

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix.
    '''
    print len(y), nb_classes
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y
