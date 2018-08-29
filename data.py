import collections
import os
import pickle
import random
import urllib
import zipfile
from io import open
from math import ceil

import numpy as np
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        print('start downloading...')
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

def read_data(filename='text8.zip'):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        print('Data size', len(data))  # 17 million
    return data


def read_own_data(filename):
    with open(filename, 'r') as f:
        data = tf.compat.as_str(f.read()).split()
        print('Data size', len(data))
    return data


def build_dataset(words, n_words):
    """
    build dataset
    :param words: corpus
    :param n_words: learn most common n_words
    :return:
        - data: []
        - count: [ [word_index, word_count], ]
        - dictionary: {word_str: word_index}
        - reversed_dictionary: {word_index: word_str}
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # UNK index is 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def dataset_tofile(data, count, dictionary, reversed_dictionary):
    pickle.dump(data, open("data/data.list", "wb"))
    pickle.dump(count, open("data/count.list", "wb"))
    pickle.dump(dictionary, open("data/word2index.dict", "wb"))
    pickle.dump(reversed_dictionary, open("data/index2word.dict", "wb"))

def read_fromfile():
    data = pickle.load(open("data/data.list", "rb"))
    count = pickle.load(open("data/count.list", "rb"))
    dictionary = pickle.load(open("data/word2index.dict", "rb"))
    reversed_dictionary = pickle.load(open("data/index2word.dict", "rb"))
    return data, count, dictionary, reversed_dictionary

def noise(vocabs, word_count):
    Z = 0.001
    unigram_table = []
    num_total_words = sum([c for w, c in word_count])
    for vo in vocabs:
        unigram_table.extend([vo] * int(((word_count[vo][1]/num_total_words)**0.75)/Z))

    print("vocabulary size", len(vocabs))
    print("unigram_table size:", len(unigram_table))
    return unigram_table


class DataPipeline:
    def __init__(self, data, word_count, data_index=0, use_noise_neg=True):
        self.data = data
        self.data_index = data_index
        vocabs = list(set(data))
        if use_noise_neg:
            self.unigram_table = noise(vocabs, word_count)
        else:
            self.unigram_table = vocabs

    def get_neg_data(self, batch_size, num, target_inputs):
        """
        sample the negative data. Don't use np.random.choice(), it is very slow.
        :param batch_size: int
        :param num: int
        :param target_inputs: []
        :return:
        """
        neg = np.zeros((num))
        for i in range(batch_size):
            delta = random.sample(self.unigram_table, num)
            while target_inputs[i] in delta:
                delta = random.sample(self.unigram_table, num)
            neg = np.vstack([neg, delta])
        return neg[1: batch_size + 1]

    def generate_batch(self, batch_size, num_skips, skip_window):
        """
        get the data batch
        :param batch_size:
        :param num_skips:
        :param skip_window:
        :return: target batch and context batch
        """
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window, target, skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

if __name__ == '__main__':
    vocabulary_size = 50000
    filename = maybe_download('text8.zip', 31344016)
    vocabulary = read_data(filename)
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    print('Most common words (+UNK)', count[:5])
    dataset_tofile(data, count, dictionary, reverse_dictionary)