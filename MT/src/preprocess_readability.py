import copy
import collections
import onmt

import argparse
import torch
import tools
import random

parser = argparse.ArgumentParser(description='preprocess.lua')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")
parser.add_argument('-task', default="simp")
parser.add_argument('-dataset', default=None)
parser.add_argument('-train_src', required=False,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=False,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=False,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=False,
                     help="Path to the validation target data")
parser.add_argument('-AE_data', required=False, type=str, default=None,
                    help="Path to the AE data")
parser.add_argument('-AE_data_amount', type=int, default=None,
                    help="AE data amount")
parser.add_argument('-language_pairs', default=[], nargs='+', type=str)
parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")
parser.add_argument('-vocab_from_file', type=str, default=None)
parser.add_argument('-src_vocab_from_file', type=str, default=None)
parser.add_argument('-tgt_vocab_from_file', type=str, default=None)

parser.add_argument('-prune_by_freq', action="store_true",
                    help="prune the vocab by minimum symbol fequency")
parser.add_argument('-src_min_freq', type=int, default=5,
                    help="Minimum frequence of the source vocabulary")
parser.add_argument('-tgt_min_freq', type=int, default=5,
                    help="Minimum frequence of the target vocabulary")
parser.add_argument('-src_vocab_size', type=int, default=40000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=40000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()


def makeVocabulary(filename, size, min_freq=None):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD])
    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.strip().split():
                vocab.add(word)
    if opt.AE_data is not None:
        AE_vocab = onmt.Dict()
        with open(opt.AE_data) as f:
            for cnt, sent in enumerate(f.readlines()):
                if opt.AE_data_amount is None or cnt < opt.AE_data_amount:
                    for word in sent.strip().split():
                        AE_vocab.add(word)
        vocab.add_dict(AE_vocab, 10.)

    originalSize = vocab.size()
    if min_freq is not None:
        vocab = vocab.prune_by_freq(min_freq)
    else:
        vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + vocab.size() + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        if opt.prune_by_freq:
            if name == 'source':
                genWordVocab = makeVocabulary(dataFile, vocabSize, opt.src_min_freq)
            elif name == 'target':
                genWordVocab = makeVocabulary(dataFile, vocabSize, opt.tgt_min_freq)
        else:
            genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)

def get_unk_offset(srcFile, tgtFile, srcDicts, tgtDicts):
    srcF = open(srcFile)
    tgtF = open(tgtFile)
    num_unk = 0
    avg_unk_offset = 0.0
    while True:
        srcWords = srcF.readline().strip().split()
        tgtWords = tgtF.readline().strip().split()

        if not srcWords or not tgtWords:
            if srcWords and not tgtWords or not srcWords and tgtWords:
                print('WARNING: source and target do not have the same number of sentences')
            break

        if len(srcWords) <= opt.seq_length and len(tgtWords) <= opt.seq_length:
            src_index = srcDicts.convertToIdx(srcWords,
                                              onmt.Constants.UNK_WORD)
            tgt_index = tgtDicts.convertToIdx(tgtWords,
                                              onmt.Constants.UNK_WORD,
                                              onmt.Constants.BOS_WORD,
                                              onmt.Constants.EOS_WORD)
            src_r = tools.readability.FleschKincaid(" ".join(srcDicts.convertToLabels(src_index, onmt.Constants.EOS)), 0)
            tgt_r = tools.readability.FleschKincaid(" ".join(tgtDicts.convertToLabels(tgt_index, onmt.Constants.EOS)), 0)
            for i in src_index:
                if srcDicts.getLabel(i) == onmt.Constants.UNK_WORD:
                    num_unk += 1
            for i in tgt_index:
                if tgtDicts.getLabel(i) == onmt.Constants.UNK_WORD:
                    num_unk += 1
            ori_r, sent_len = tools.readability.FleschKincaid_len(" ".join(srcWords))
            avg_unk_offset += (ori_r - src_r) * sent_len
            ori_r, sent_len = tools.readability.FleschKincaid_len(" ".join(tgtWords))
            avg_unk_offset += (ori_r - tgt_r) * sent_len
    srcF.close()
    tgtF.close()
    if num_unk > 0:
        offset = avg_unk_offset * 1. / num_unk
    else:
        offset = 0
    return offset


def makeData(unk_offset, srcFile, tgtFile, srcDicts, tgtDicts, src = None, tgt = None, src_rb= None, tgt_rb = None, data_amount=None, target_rb=None, all_src_rb = None, all_tgt_rb = None):
    if src is None:
        src, tgt, src_rb, tgt_rb = [], [], [], []
    count, ignored = 0, 0
    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)
    if target_rb is not None:
        target_rb_dict = collections.defaultdict(int)
        ratio = data_amount * 1.0 / len(target_rb)
        for rb in target_rb:
            target_rb_dict[int(rb * 10)] += ratio
    else:
        target_rb_dict = None
    cont_empty = 0
    conse_ignore = 0
    while True:
        srcWords = srcF.readline().strip().split()
        tgtWords = tgtF.readline().strip().split()
        if not srcWords or not tgtWords:
            if srcWords and not tgtWords or not srcWords and tgtWords:
                print " ".join(srcWords)
                print " ".join(tgtWords)
                print('WARNING: source and target do not have the same number of sentences')
            cont_empty += 1
            if cont_empty > 5:
                break
            else:
                continue
        cont_empty = 0

        if len(srcWords) <= opt.seq_length and len(tgtWords) <= opt.seq_length:
            src_index = srcDicts.convertToIdx(srcWords,
                                              onmt.Constants.UNK_WORD)
            tgt_index = tgtDicts.convertToIdx(tgtWords,
                                              onmt.Constants.UNK_WORD,
                                              onmt.Constants.BOS_WORD,
                                              onmt.Constants.EOS_WORD)
            if all_src_rb is not None:
                src_r = all_src_rb
            else:
                src_r = tools.readability.FleschKincaid(" ".join(srcDicts.convertToLabels(src_index, onmt.Constants.EOS)), unk_offset)
            if all_tgt_rb is not None:
                tgt_r = all_tgt_rb
            else:
                tgt_r = tools.readability.FleschKincaid(" ".join(tgtDicts.convertToLabels(tgt_index, onmt.Constants.EOS)), unk_offset)
            if target_rb_dict is not None:
                if target_rb_dict[int(tgt_r * 10)] >= 1:
                    target_rb_dict[int(tgt_r * 10)] -= 1
                else:
                    ignored += 1
                    conse_ignore += 1
                    if conse_ignore >= 10000:
                        break
                    continue
            conse_ignore = 0
            src_rb += [src_r]
            tgt_rb += [tgt_r]
            src += [src_index]
            tgt += [tgt_index]

        else:
            ignored += 1

        count += 1
        if data_amount and len(src_rb) >= data_amount:
            break

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()
    sizes = [len(src_idx) for src_idx in src]
    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        src_rb = [src_rb[idx] for idx in perm]
        tgt_rb = [tgt_rb[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    src_rb = [src_rb[idx] for idx in perm]
    tgt_rb = [tgt_rb[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length))
    return src, tgt, src_rb, tgt_rb

def get_threshold(readability):
    threshold = {}
    all_readability = sorted(readability)
    for num_bin in range(3, 11):
        now_threshold = []
        l = len(all_readability)
        for j in range(1, num_bin):
            now_threshold.append(all_readability[int(l * 1. / num_bin * j)])
        threshold[num_bin] = now_threshold
    #print threshold
    return threshold

def balance(src, tgt, src_rb, tgt_rb):
    src_dict = collections.defaultdict(int)
    max_rb = 0
    for rb in src_rb:
        src_dict[rb] += 1
        max_rb = max(max_rb, src_dict[rb])
    new_src = []
    new_tgt = []
    new_src_rb = []
    new_tgt_rb = []
    for rb in src_dict:
        for k in range(max_rb - src_dict[rb]):
            while True:
                i = random.randrange(len(src))
                if src_rb[i] == rb:
                    new_src += [src[i]]
                    new_tgt += [tgt[i]]
                    new_src_rb += [src_rb[i]]
                    new_tgt_rb += [tgt_rb[i]]
                    break
    src += new_src
    tgt += new_tgt
    src_rb += new_src_rb
    tgt_rb += new_tgt_rb
    return src, tgt, src_rb, tgt_rb

def main():
    dicts = {}
    if opt.task == "Multi-MT":
        assert opt.src_vocab_from_file is not None
        dicts['src'] = initVocabulary('source', opt.src_vocab_from_file, opt.src_vocab,
                                      opt.src_vocab_size)
        dicts['tgt'] = initVocabulary('target', opt.tgt_vocab_from_file, opt.tgt_vocab,
                                      opt.tgt_vocab_size)
        unk_offset = 0
    elif opt.task == "MT":
        assert opt.vocab_from_file is not None
        dicts['src'] = initVocabulary('source', opt.vocab_from_file, opt.src_vocab,
                                      opt.src_vocab_size)
        dicts['tgt'] = initVocabulary('target', opt.vocab_from_file, opt.tgt_vocab,
                                      opt.tgt_vocab_size)
        unk_offset = 0
    else:
        dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                      opt.src_vocab_size)
        dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                      opt.tgt_vocab_size)
        unk_offset = get_unk_offset(opt.train_src, opt.train_tgt,
                       dicts['src'], dicts['tgt'])
        print "unk_offset", unk_offset
    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')
    if opt.task == "Multi-MT":
        language_pairs = opt.language_pairs
        src_language_mapping = {}
        tgt_language_mapping = {}
        data = {}
        data_direc = "../../data_%s/%s" % (opt.task, opt.dataset)
        for split_set in ["train", "valid"]:
            src = []
            tgt = []
            src_rb = []
            tgt_rb = []
            for lp in language_pairs:
                print (split_set + "_" + lp)
                line = lp.split("-")
                s_lang = line[0]
                t_lang = line[1]
                if s_lang not in src_language_mapping:
                    src_language_mapping[s_lang] = len(src_language_mapping)
                if t_lang not in tgt_language_mapping:
                    tgt_language_mapping[t_lang] = len(tgt_language_mapping)
                src, tgt, src_rb, tgt_rb = makeData(0, "%s/%s.%s.%s" % (data_direc, split_set, lp, s_lang), "%s/%s.%s.%s" % (data_direc, split_set, lp, t_lang), dicts['src'], dicts['tgt'], src=src, tgt=tgt, src_rb=src_rb, tgt_rb=tgt_rb,
                                                     all_src_rb=src_language_mapping[s_lang], all_tgt_rb=tgt_language_mapping[t_lang])
            print "src_len 0", len(src)
            src, tgt, src_rb, tgt_rb = balance(src, tgt, src_rb, tgt_rb)
            print "src len 1", len(src)
            src, tgt, src_rb, tgt_rb = balance(src, tgt, src_rb, tgt_rb)
            print "src len 2", len(src)
            data[split_set] = {"src": src, "tgt": tgt, "src_rb": src_rb, "tgt_rb": tgt_rb}
        num_bin = max(len(src_language_mapping), len(tgt_language_mapping))
        save_data = {'dicts': dicts,
                     'train': data['train'],
                     'valid': data['valid'],
                     "src_threshold": {num_bin: range(num_bin - 1)},
                     "tgt_threshold": {num_bin: range(num_bin - 1)},
                     "unk_offset": 0,
                     "language_pairs": language_pairs,
                     "src_language_mapping": src_language_mapping,
                     "tgt_language_mapping": tgt_language_mapping,
                     }
    else:
        print('Preparing training ...')

        train = {}
        if opt.task == "MT":
            all_src_rb = 0
            all_tgt_rb = 1
        else:
            all_src_rb = None
            all_tgt_rb = None
        train['src'], train['tgt'], train['src_rb'], train['tgt_rb'] = makeData(unk_offset, opt.train_src, opt.train_tgt,
                                              dicts['src'], dicts['tgt'], all_src_rb=all_src_rb, all_tgt_rb=all_tgt_rb)
        all_readability = train['src_rb'] + train['tgt_rb']
        if opt.task == "MT":
            threshold = {2 : [0]}
            src_threshold = threshold
            tgt_threshold = threshold
        else:
            threshold = get_threshold(all_readability)
            src_threshold = get_threshold(train['src_rb'])
            tgt_threshold = get_threshold(train['tgt_rb'])
        tgt_rb_offset = sum(train['src_rb']) - sum(train['tgt_rb'])
        tgt_rb_offset /= 1. * len(train['src_rb'])
        print "rb offset", tgt_rb_offset
        train_bi = {}
        train_bi['src'], train_bi['tgt'], train_bi['src_rb'], train_bi['tgt_rb'] = makeData(unk_offset, opt.train_tgt, opt.train_src, dicts['src'], dicts['tgt'], all_src_rb=all_tgt_rb, all_tgt_rb=all_src_rb)

        print('Preparing validation ...')
        valid = {}
        valid['src'], valid['tgt'], valid['src_rb'], valid['tgt_rb'] = makeData(unk_offset, opt.valid_src, opt.valid_tgt,
                                        dicts['src'], dicts['tgt'], all_src_rb=all_src_rb, all_tgt_rb=all_tgt_rb)

        if opt.AE_data:
            assert opt.task != "MT"
            print ('Preparing AE ...')
            AE_data = {}
            AE_data['src'], AE_data['tgt'], AE_data['src_rb'], AE_data['tgt_rb'] = makeData(unk_offset, opt.AE_data, opt.AE_data,
                                                                                    dicts['src'], dicts['tgt'], data_amount = opt.AE_data_amount, target_rb=copy.deepcopy(train['src_rb']))#tgt_rb!!!
        else:
            AE_data = None
        domain_AE = {}
        domain_AE['src'], domain_AE['tgt'], domain_AE['src_rb'], domain_AE['tgt_rb'] = makeData(unk_offset, opt.train_src, opt.train_src,
                                                                                                dicts['src'], dicts['tgt'], all_src_rb=all_src_rb, all_tgt_rb=all_src_rb)
        domain_AE['src'], domain_AE['tgt'], domain_AE['src_rb'], domain_AE['tgt_rb'] = makeData(unk_offset, opt.train_tgt, opt.train_tgt,
                                                                                                dicts['src'], dicts['tgt'],
                                                                                                domain_AE['src'], domain_AE['tgt'], domain_AE['src_rb'], domain_AE['tgt_rb'],
                                                                                                all_src_rb=all_tgt_rb, all_tgt_rb=all_tgt_rb)

        if opt.src_vocab is None:
            saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
        if opt.tgt_vocab is None:
            saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')


        print('Saving data to \'' + opt.save_data + '-train.pt\'...')
        save_data = {'dicts': dicts,
                     'train': train,
                     'valid': valid,
                     "AE_data": AE_data,
                     "domain_AE": domain_AE,
                     "all_threshold": threshold,
                     "src_threshold": src_threshold,
                     "tgt_threshold": tgt_threshold,
                     "unk_offset": unk_offset,
                     "tgt_rb_offset": tgt_rb_offset,
                     "train_bi": train_bi,
                     }
    torch.save(save_data, opt.save_data + '-train.pt')

if __name__ == "__main__":
    main()
