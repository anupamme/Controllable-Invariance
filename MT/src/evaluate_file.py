import os
import torch
#from pythonrouge.pythonrouge import Pythonrouge
import argparse
from tools.bleu_score import corpus_bleu, sentence_bleu
from tools.SARI import SARIsent, SARI_corpus
from tools.readability import FleschKincaid, get_FK_bin
import math
import copy
from nltk import sent_tokenize
import codecs
import string
import subprocess

def sigmoid(x):
    return 1. / (1 + math.exp(-x))

printable = set(string.printable)

def clean(data):
    global printable
    if type(data) == list:
        return [clean(d) for d in data]
    elif type(data) == str:
        new_data = ''.join(filter(lambda x: x in printable, data))
        return new_data
    else:
        assert False

def evaluate_bleu(ref, output):
    cmd = ['perl', 'tools/multi-bleu.perl', ref]
    #cmd = ['perl', 'tools/mteval-v13a.pl', ref]
    bleu_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, stdin = open(output, "r"))
    st = "BLEU = "
    pos = bleu_output.find(",")
    bleu = float(bleu_output[len(st): pos])
    return bleu

def main(arg_lists):
    parser = argparse.ArgumentParser()
    parser.add_argument('-outputs', type=str, default=[], nargs='+', metavar='PATH',
                        help="Output file (default: None)")
    parser.add_argument('-data', type=str, default="newsela")
    parser.add_argument('-task', type=str, default="simp")
    parser.add_argument('-verbose', action="store_true")
    parser.add_argument('-single_rb', type=int, default=None)
    parser.add_argument('-language_pair', default=None)
    parser.add_argument('-test_set', default=None)
    if arg_lists is not None:
        args = parser.parse_args(arg_lists)
    else:
        args = parser.parse_args()
    options = []
    suffix = ""
    dataset = args.data
    if args.task == "simp":
        dataset_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../data_%s/%s/test/" % (args.task, dataset)
        source_file = dataset_dir + "test.normal%s" % suffix
        test_target = dataset_dir + "test.simple."
    elif args.task == "MT":
        #test_src = "test.de-en.de%s"
        test_src = "test.en-zh.en%s"
        dataset_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../data_%s/%s/" % (args.task, dataset)
        source_file = dataset_dir + test_src % suffix
        test_target = "test.en-zh.zh"
    elif args.task == "Multi-MT":
        line = args.language_pair.split("-")
        S_lang = line[0]
        T_lang = line[1]
        dataset_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../data_%s/%s/" % (args.task, dataset)
        source_file = dataset_dir + "%s.%s.%s" % (args.test_set, args.language_pair, S_lang)
        test_target = "orig/%s.%s.%s" % (args.test_set, args.language_pair, T_lang)
    else:
        assert False
    source_file = open(source_file, "r")
    target_files = []
    for inf in os.listdir(dataset_dir):
        if inf.find("test.simple") == 0:
            target_files += [open(dataset_dir + inf, "r")]
    ROUGE_path = "../../pythonrouge/rouge/ROUGE-1.5.5.pl"
    ROUGE_data_path = "../../pythonrouge/rouge/data"
    #rouge = Pythonrouge(n_gram=2, ROUGE_SU4=True, ROUGE_L=True, stemming=True, stopwords=True, word_level=True, length_limit=True, length=50, use_cf=False, cf=95, scoring_formula="average", resampling=True, samples=1000, favor=True, p=0.5)
    data_path = "../../data/%s/%s-train.pt" % (args.data, args.data)
    if args.single_rb is not None:
        output = args.outputs[0]
        model_path = os.path.dirname(output) + "/model.pt"
        model = torch.load(model_path)
        threshold = model["threshold"]["tgt"]
    #unk_offset = torch.load(data_path)["unk_offset"]
    aligned_sentences = {}
    sources = []
    for s in source_file:
        s = s.strip()
        if args.task == "simp":
            assert s not in aligned_sentences
        aligned_sentences[s] = []
        for tar_file in target_files:
            t = tar_file.readline().strip()
            if t.strip() != "":
                aligned_sentences[s].append(t)
        sources.append(s)
    output_files = []
    systems = []
    scores = []
    for ouf in args.outputs:
        output_files.append([st.strip() for st in open(ouf, "r")])
        systems.append(os.path.basename(os.path.dirname(os.path.normpath(ouf))))
        assert len(output_files[-1]) == len(sources)
    n_samples = len(sources)
    source_to_evaluate = sources
    output_to_evaluate = output_files
    reference_to_evaluate = []
    unk_offset = 10000000
    for i in range(len(sources)):
        reference_to_evaluate += [aligned_sentences[sources[i]]]
    for k in range(len(systems)):
        bleu = evaluate_bleu(dataset_dir + test_target, args.outputs[k])
        #print systems[k]
        #rouge_output = [sent_tokenize(doc) for doc in output_to_evaluate[k]]
        #rouge_ref = [[sent_tokenize(ref) for ref in refs] for refs in reference_to_evaluate]
        #rouge_output = [[doc] for doc in clean(output_to_evaluate[k])]
        #rouge_ref = [[[ref] for ref in refs] for refs in clean(reference_to_evaluate)]
        #setting_file = rouge.setting(files=False, summary=rouge_output, reference=rouge_ref)
        '''result = rouge.eval_rouge(setting_file, recall_only=True, ROUGE_path=ROUGE_path, data_path=ROUGE_data_path)
        R1 = result["ROUGE-1"] * 100
        R2 = result['ROUGE-2'] * 100
        RL = result['ROUGE-L'] * 100'''
        R1 = 0
        R2 = 0
        RL = 0
        FK_O = FleschKincaid("\n".join(output_to_evaluate[k]), unk_offset)
        FK_acc = 0
        if args.single_rb is not None:
            for st in output_to_evaluate[k]:
                real_FK = get_FK_bin(st, threshold, unk_offset)
                if real_FK == args.single_rb:
                    FK_acc += 1
            FK_acc /= 1. * len(output_to_evaluate[k])
        '''input_bleu = evaluate_bleu(dataset_dir + "test.normal", args.outputs[k])
        alpha = 0.9
        ibleu = alpha * bleu - (1 - alpha) * input_bleu
        FK_I = FleschKincaid("\n".join(source_to_evaluate), unk_offset)
        FKdiff_before = FK_I - FK_O
        FKdiff = sigmoid(FKdiff_before)
        #FKdiff = FKdiff_before
        FKBLEU = FKdiff * ibleu'''
        if args.task == "simp":
            SARI = SARI_corpus(source_to_evaluate, output_to_evaluate[k], reference_to_evaluate) * 100
        else:
            SARI = 0
        print "BLEU %6.2f, SARI %6.2f, R1 %6.2f, R2 %6.2f, RL %6.2f, FK_O %6.2f, acc %6.2f" % (bleu, SARI,  R1, R2, RL, FK_O, FK_acc)
        '''bleu = sentence_bleu(aligned_sentences[sources[i]], o, emulate_multibleu=True) * 100
        scores[k][0] += bleu
        FK_BLEU = compute_FK_BLEU(sources[i], o, aligned_sentences[sources[i]], unk_offset)
        scores[k][3] += FK_BLEU
        SARI = SARIsent(sources[i], o, aligned_sentences[sources[i]]) * 100
        scores[k][2] += SARI
        FK = FleschKincaid(o, unk_offset)
        scores[k][1] += FK
        min_FK = max(min_FK, FK)
        #print bleu, FK_BLEU
        irint vars(self.checkpoint_opt)
        FK_ori = FleschKincaid(sources[i].strip(), unk_offset)
        prod = bleu * min((FK_ori + 16.) / (FK + 16.), 2)
        scores[k][4] += prod
        summary = [[o]]
        reference = [[aligned_sentences[sources[i]]]]
        scores[k][5] += R1 * 100
        scores[k][6] += R2 * 100
        scores[k][7] += RL * 100
        if args.verbose:
            print "BLEU %6.2f, R1 %6.2f, R2 %6.2f, RL %6.2f, SARI %6.2f, FKBLEU %6.2f, FK %6.2f, PROD %6.2f" % (bleu, R1, R2, RL, SARI, FK_BLEU, FK, prod), "\n"
            print'''

if __name__ == "__main__":
    main(None)
