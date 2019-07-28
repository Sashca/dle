import os
import torch
import torchtext.data
    
class Texts(object):
    def __init__(self, path):
        path_ = path + 'train.txt'
        with open(path_, 'r') as content:
            content_train = content.read()
        path_ = path + 'valid.txt'
        with open(path_, 'r') as content:
            content_valid = content.read()
        path_ = path + 'test.txt'
        with open(path_, 'r') as content:
            content_test = content.read()
            
        tokenize = lambda x: list(x)
        self.text = torchtext.data.Field(sequential=True, tokenize=tokenize, lower=True)
        cont = content_train + content_valid + content_test
        
        self.text.build_vocab(cont)
       
        self.train = self.text.process(content_train)
        self.valid = self.text.process(content_valid)
        self.test = self.text.process(content_test)

    

class TextLoader(object):
    def __init__(self, dataset, batch_size=128, sequence_length=30):
        self.data = dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self._batchify()
        
    def _batchify(self):
        # Work out how cleanly we can divide the dataset into batch_size parts.
        self.nbatch = self.data.size(0) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = self.data.narrow(0, 0, self.nbatch * self.batch_size)
        # Evenly divide the data across the batch_size batches.
        self.batch_data = data.view(self.batch_size, -1).t().contiguous()
    
    def _get_batch(self, i):
        seq_len = min(self.sequence_length, len(self.batch_data) - 1 - i)
        data = self.batch_data[i:i+seq_len]
        target = self.batch_data[i+1:i+1+seq_len].view(-1)
        return data, target
    
    def __iter__(self):
        for i in range(0, self.batch_data.size(0) - 1, self.sequence_length):
            data, targets = self._get_batch(i)
            yield data, targets
    
    def __len__(self):
        return self.batch_data.size(0)