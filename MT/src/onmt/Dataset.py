import onmt
import torch
from torch.autograd import Variable
import random

class Dataset(object):

    def __init__(self, srcData, tgtData, batchSize, cuda, align_right=False):
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda
        self.align_right = align_right
        self.batchSize = batchSize
        self.numBatches = len(self.src) // batchSize

    def _batchify(self, data, align_right=False):
        max_length = max(x.size(0) for x in data)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        out = out.t().contiguous()
        if self.cuda:
            out = out.cuda()

        v = Variable(out)
        return v

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch = self._batchify(
            self.src[index*self.batchSize:(index+1)*self.batchSize], align_right=self.align_right)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:(index+1)*self.batchSize])
        else:
            tgtBatch = None

        return srcBatch, tgtBatch

    def __len__(self):
        return self.numBatches

class BucketIterator(object):

    def __init__(self, srcData, tgtData, src_rb, tgt_rb, opt, threshold, align_right=False, bi_dir=False, dicts=None):
        print "reverse", opt.reverse_src
        if opt.reverse_src:
            self.src = []
            for src_sent in srcData:
                src_sent = src_sent.tolist()
                #print src_sent
                self.src.append(torch.LongTensor(src_sent[::-1]))
        else:
            self.src = srcData
        self.bi_dir = bi_dir
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.tgt_rb = self.get_readability_bin(threshold["tgt"], tgt_rb)
        self.src_rb = self.get_readability_bin(threshold["src"], src_rb)
        if opt.filter_src_rb is not None:
            new_src, new_tgt, new_sb, new_tb = [], [], [], []
            for s, t, sb, tb in zip(self.src, self.tgt, self.src_rb, self.tgt_rb):
                if opt.filter_src_rb == sb:
                    new_src += [s]
                    new_tgt += [t]
                    new_sb += [sb]
                    new_tb += [tb]
            self.src = new_src
            self.tgt = new_tgt
            self.src_rb = new_sb
            self.tgt_rb = new_tb
        self.cuda = opt.cuda
        self.align_right = align_right
        self.numData = len(self.src)
        print "num data", self.numData
        self.batchSize = opt.batch_size
        self.cacheSize = self.batchSize * 20
        self.numBatches = (self.numData - 1) // self.batchSize + 1
        print self.numBatches
        self.dicts = dicts
        self.reverse_src = opt.reverse_src
        self._reset()

    def get_readability_bin(self, threshold, list_rb):
        rb_bin = []
        len_threshold = len(threshold)
        for rb in list_rb:
            ans = len_threshold
            for j in range(len_threshold):
                if rb <= threshold[j]:
                    ans = j
                    break
            rb_bin.append(ans)
        return rb_bin

    def _batchify(self, data, align_right=False):
        max_length = max(x.size(0) for x in data)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        out = out.t().contiguous()
        if self.cuda:
            out = out.cuda()

        v = Variable(out)
        return v

    def _reset(self):
        self.currIdx = 0
        self.dataOrders = torch.randperm(self.numData)

    def __iter__(self):
        self._reset()
        while True:
            caches = []
            for i in range(self.cacheSize):
                if self.currIdx == self.numData:
                    break
                dataIdx = self.dataOrders[self.currIdx]
                caches.append((self.src[dataIdx], self.tgt[dataIdx], self.src_rb[dataIdx], self.tgt_rb[dataIdx]))
                self.currIdx += 1
            caches = sorted(caches, key = lambda (s, t, s_rb, t_rb): len(s), reverse=True)
            batches = []
            for i in range(0, len(caches), self.batchSize):
                batches.append(caches[i:i+self.batchSize])
            random.shuffle(batches)
            for batch in batches:
                sb, tb, s_rb, t_rb = zip(*batch)
                srcBatch = self._batchify(sb, align_right=self.align_right)
                tgtBatch = self._batchify(tb)
                s_rb = Variable(torch.LongTensor(s_rb))
                t_rb = Variable(torch.LongTensor(t_rb))
                if self.cuda:
                    s_rb = s_rb.cuda()
                    t_rb = t_rb.cuda()
                yield srcBatch, tgtBatch, s_rb, t_rb

            if self.currIdx == self.numData:
                raise StopIteration

    def __len__(self):
        return self.numBatches

    def __getitem__(self, index, sort=False):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        src_batch = self.src[index*self.batchSize:(index+1)*self.batchSize]
        tgt_batch = self.tgt[index*self.batchSize:(index+1)*self.batchSize]
        s_rb = self.src_rb[index*self.batchSize:(index+1)*self.batchSize]
        t_rb = self.tgt_rb[index*self.batchSize:(index+1)*self.batchSize]
        caches = zip(src_batch, tgt_batch, s_rb, t_rb)
        if sort:
            caches = sorted(caches, key = lambda (s, t, s_rb, t_rb): len(s), reverse=True)
        sb, tb, s_rb, t_rb = zip(*caches)
        srcBatch = self._batchify(sb, align_right=self.align_right)
        if self.tgt:
            tgtBatch = self._batchify(tb)
        else:
            tgtBatch = None
        s_rb = Variable(torch.LongTensor(s_rb))
        t_rb = Variable(torch.LongTensor(t_rb))
        if self.cuda:
            s_rb = s_rb.cuda()
            t_rb = t_rb.cuda()
        return srcBatch, tgtBatch, s_rb, t_rb

    def random_batch(self, sort=False):
        data_index = range(self.numData)
        random.shuffle(data_index)
        data_index = data_index[:self.batchSize]
        index = 0
        src_batch = []
        tgt_batch = []
        s_rb = []
        t_rb = []
        for i in data_index:
            src_batch += [self.src[i]]
            tgt_batch += [self.tgt[i]]
            s_rb += [self.src_rb[i]]
            t_rb += [self.tgt_rb[i]]
        caches = zip(src_batch, tgt_batch, s_rb, t_rb)
        caches = sorted(caches, key = lambda (s, t, s_rb, t_rb): len(s), reverse=True)
        sb, tb, s_rb, t_rb = zip(*caches)
        srcBatch = self._batchify(sb, align_right=self.align_right)
        if self.tgt:
            tgtBatch = self._batchify(tb)
        else:
            tgtBatch = None
        s_rb = Variable(torch.LongTensor(s_rb))
        t_rb = Variable(torch.LongTensor(t_rb))
        if self.cuda:
            s_rb = s_rb.cuda()
            t_rb = t_rb.cuda()
        return srcBatch, tgtBatch, s_rb, t_rb

class mixed_iterator(object):
    def __init__(self, datas, probability):
        self.datas = datas
        s = sum(probability)
        for i in range(len(probability)):
            probability[i] /= s * 1.0
        self.probability = probability
        self.iter_data = []

    def _reset(self):
        self.iter_data = []
        for d in self.datas:
            self.iter_data.append(d.__iter__())

    def random_batch(self, sort=False):
        return self.datas[0].random_batch(sort=sort)

    def __iter__(self):
        self._reset()
        while True:
            p = random.random()
            k = -1
            for i in range(len(self.probability)):
                if p <= self.probability[i]:
                    k = i
                    break
                else:
                    p -= self.probability[i]
            if k == -1:
                assert p < 1e-5
                k = len(self.probability) - 1
            try:
                batch = self.iter_data[k].next()
            except StopIteration:
                if k == 0:
                    raise StopIteration
                else:
                    self.iter_data[k] = self.datas[k].__iter__()
                    batch = self.iter_data[k].next()
            yield batch, k
