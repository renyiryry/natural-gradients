"""Functions for downloading and reading MNIST data."""
import gzip
import os
# import urllib
import urllib.request
import numpy as np


import sys


def maybe_download(SOURCE_URL, filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    
    print('current path', os.getcwd())
    
    print('work_directory', work_directory)
    
    if not os.path.exists(work_directory):
#         os.mkdir(work_directory)
        os.makedirs(work_directory)
        
    filepath = os.path.join(work_directory, filename)
    
    print('filepath', filepath)
    
    if not os.path.exists(filepath):
        
        
        
        
        
        
        
#         filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        
        print('print(shape(data))', np.shape(data))
        
        data = data.reshape(num_images, rows, cols, 1)
        
        print('print(shape(data))', np.shape(data))
        
        return data


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels
    
    
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):

    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in range(784)]
            fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(name_dataset, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    
    train_dir = '../data/' + name_dataset + '_data'
    
    if name_dataset == 'MNIST':
        
        SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
        VALIDATION_SIZE = 5000
    
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
        train_images = extract_images(local_file)
        local_file = maybe_download(SOURCE_URL, TRAIN_LABELS, train_dir)
        train_labels = extract_labels(local_file, one_hot=one_hot)
        local_file = maybe_download(SOURCE_URL, TEST_IMAGES, train_dir)
        test_images = extract_images(local_file)
        local_file = maybe_download(SOURCE_URL, TEST_LABELS, train_dir)
        test_labels = extract_labels(local_file, one_hot=one_hot)
    
        
        
    elif name_dataset == 'CIFAR':
        import tarfile
        import pickle
        import numpy as np
        
        SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'
        file_name = 'cifar-10-python.tar.gz'
        
        local_file = maybe_download(SOURCE_URL, file_name, train_dir)
        
        print('local_file', local_file)
        
        
        tf = tarfile.open(train_dir + '/' + file_name)
        tf.extractall(train_dir)
        
        working_dir = train_dir + '/cifar-10-batches-py/'
        
#         train_images = []
#         train_labels = []
        for i in range(5):
            with open(working_dir + 'data_batch_' + str(i+1), 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                
#                 for key in dict:
#                     print('key')
#                     print(key)
#                     print(dict[key])

#                 print(train_images)
#                 print(dict['data'.encode('UTF-8')])
                
                if i == 0:
                    train_images = dict['data'.encode('UTF-8')]
                    train_labels = dict['labels'.encode('UTF-8')]
                else:
                    train_images = np.concatenate((train_images, dict['data'.encode('UTF-8')]))
                    train_labels = np.concatenate((train_labels, dict['labels'.encode('UTF-8')]))
                
        with open(working_dir + 'test_batch', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            test_images = dict['data'.encode('UTF-8')]
            test_labels = dict['labels'.encode('UTF-8')]
            test_labels = np.asarray(test_labels)

        print('train_images.shape')
        print(train_images.shape)
        print('train_labels.shape')
        print(train_labels.shape)
        print('test_images.shape')
        print(test_images.shape)
        print('test_labels.shape')
        print(test_labels.shape)
        
        train_images = train_images[:, np.newaxis, np.newaxis]
        test_images = test_images[:, np.newaxis, np.newaxis]
        
        print('train_images.shape')
        print(train_images.shape)
        print('train_labels.shape')
        print(train_labels.shape)
        print('test_images.shape')
        print(test_images.shape)
        print('test_labels.shape')
        print(test_labels.shape)
                
        
        VALIDATION_SIZE = 5000
        
    else:
        print('Dataset not supported.')
        sys.exit()
        
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)    
        
    
    return data_sets
