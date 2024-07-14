import os

import numpy as np
from needle import backend_ndarray as nd
from needle import Tensor

class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.sz = 0

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            cur_idx = len(self.idx2word)
            self.word2idx[word] = cur_idx
            self.idx2word.append(word)
            self.sz += 1
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)
        
    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        # <eos>
        ids = []
        eos_id = self.dictionary.add_word('<eos>')
        # read the data and append <eos> to the end of each line

        def tokenize_line(line):
            words = line.split()
            for word in words:
                ids.append(self.dictionary.add_word(word))
            ids.append(eos_id)  # '<eos>' should be appended to the end of each line
        
        with open(path, 'r') as f:
            # lines = f.readlines() # 读完所有行，性能太差

            if max_lines is not None:
                for _ in range(max_lines):
                    tokenize_line(f.readline())
                
            else:
                for line in f.readlines():
                    tokenize_line(line)
                
                # 空格分隔
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    Inputs:
        data - sequence data of shape
        batch_size - int specifying the batch size
    """
    ### BEGIN YOUR SOLUTION
    n = len(data) // batch_size
    return np.array(data[:n*batch_size]).reshape((batch_size, n)).swapaxes(-1,-2)
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    # max_index = min(len(batches), (i + 1) * bptt)
    # data = batches[i * bptt:max_index]
    # target = batches[i * bptt + 1:max_index + 1]
    # return data, target.flatten()
    bptt = min(bptt, len(batches)-1-i)
    data = batches[i:i+bptt]
    target = batches[i+1:i+1+bptt]
    data = Tensor(data, device=device, dtype=dtype)
    target = Tensor(target.reshape(-1), device=device, dtype=dtype)
    return data, target
    ### END YOUR SOLUTION