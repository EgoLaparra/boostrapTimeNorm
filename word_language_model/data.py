import os
import torch
import re

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, path_del):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), path_del)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), path_del)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), path_del)

    def tokenize(self, path, path_del):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        assert path_del is None or os.path.exists(path_del)
        # Create regex of delimiters
        del_regex = " "
        if path_del is not None:
            with open(path_del, 'r', encoding="utf8") as f:
                delimiters = list()
                for line in f:
                    delimiter = line.rstrip("\n")
                    if delimiter != "":
                        delimiters.append(re.escape(delimiter))
            del_regex = "|".join(delimiters)
            
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = re.split(del_regex, line) + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = re.split(del_regex, line) + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
