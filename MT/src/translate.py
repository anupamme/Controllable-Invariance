import onmt
import torch
import argparse
import math
import os

def parse_arg(arg_list = None):
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model', required=True, help='Path to model .pt file')
    parser.add_argument('-task', default="simp", help="simp or MT")
    parser.add_argument('-data', default="newsela",
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-language_pair', default=None)
    parser.add_argument('-test_set', default=None)
    parser.add_argument('-output', default=None,
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-output_representation', default=None, help="path to the outputed representation file")
    parser.add_argument('-beam_size',  type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-max_sent_length', default=100,
                        help='Maximum sentence length.')
    parser.add_argument('-replace_unk', action="store_true",
                        help="""Replace the generated UNK tokens with the source
                        token that had the highest attention weight. If phrase_table
                        is provided, it will lookup the identified source token and
                        give the corresponding target token. If it is not provided
                        (or the identified source token does not exist in the
                        table) then it will copy the source token""")
    # parser.add_argument('-phrase_table',
    #                     help="""Path to source-target dictionary to replace UNK
    #                     tokens. See README.md for the format of this file.""")
    parser.add_argument('-verbose', action="store_true",
                        help='Print scores and predictions for each sentence')
    parser.add_argument('-bpe', action="store_true")
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-gpus', type=int, default=-1,
                        help="Device to run on")
    parser.add_argument('-tgt_rb_offset', type=float, default=None, help="tgt_rb_offset")
    parser.add_argument('-tgt_rb_all', type=int, default=None, help="tgt_rb_all")
    if arg_list is None:
        opt = parser.parse_args()
    else:
        opt = parser.parse_args(arg_list)
    return opt

def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f, %d" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal), wordsTotal))

def main(arg_list = None):
    opt = parse_arg(arg_list)
    if opt.task == "simp":
        opt.src = "../../data_%s/%s/test/test.normal" % (opt.task, opt.data)
        opt.tgt = "../../data_%s/%s/test/test.simple.0" % (opt.task, opt.data)
    elif opt.task == "MT":
        #opt.src = "../../data_%s/%s/test.de-en.de" % (opt.task, opt.data)
        #opt.tgt = "../../data_%s/%s/test.de-en.en" % (opt.task, opt.data)
        opt.src = "../../data_%s/%s/test.en-zh.en" % (opt.task, opt.data)
        opt.tgt = "../../data_%s/%s/test.en-zh.zh" % (opt.task, opt.data)
    elif opt.task == "Multi-MT":
        line = opt.language_pair.split("-")
        S_lang = line[0]
        T_lang = line[1]
        opt.src = "../../data_%s/%s/%s.%s.%s" % (opt.task, opt.data, opt.test_set, opt.language_pair, S_lang)
        opt.tgt = "../../data_%s/%s/%s.%s.%s" % (opt.task, opt.data, opt.test_set, opt.language_pair, T_lang)
    else:
        assert False
    if opt.output is None:
        opt.output = os.path.dirname(opt.model) + "/" + "test.txt"
    opt.gpu = opt.gpus
    opt.cuda = opt.gpu > -1
    torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator(opt)

    outF = open(opt.output, 'w')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None
    src_contents = open(opt.src).readlines()
    representations = []
    src_rb_list = []
    for line_num, line in enumerate(src_contents):

        srcTokens = line.split()
        srcBatch += [srcTokens]
        if tgtF:
            tgtTokens = tgtF.readline().split() if tgtF else None
            tgtBatch += [tgtTokens]

        if line_num < len(src_contents) - 1 and len(srcBatch) < opt.batch_size:
            continue

        predBatch, predScore, goldScore, rep, src_rb = translator.translate(srcBatch, tgtBatch)
        representations.append(rep)
        src_rb_list.append(src_rb)
        '''predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        if tgtF is not None:
            goldScoreTotal += sum(goldScore)
            goldWordsTotal += sum(len(x) for x in tgtBatch)'''

        for b in range(len(predBatch)):
            count += 1
            pred_sent = " ".join(predBatch[b][0])
            if opt.bpe:
                pred_sent = pred_sent.replace("@@ ", "")
            outF.write(pred_sent + '\n')

            if opt.verbose:
                print('SENT %d: %s' % (count, " ".join(srcBatch[b])))
                print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                if opt.bpe:
                    print('PRED CON %d: %s' % (count, pred_sent))
                print("PRED SCORE: %.4f" % predScore[b][0])

                if tgtF is not None:
                    print('GOLD %d: %s ' % (count, " ".join(tgtBatch[b])))
                    print("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        print("[%.4f] %s" % (predScore[b][n], " ".join(predBatch[b][0])))
                print('')
            predScoreTotal += predScore[b][0]
            predWordsTotal += len(predBatch[b][0])
            if tgtF is not None:
                goldScoreTotal += goldScore[b]
                goldWordsTotal += len(tgtBatch[b])
            '''reportScore('PRED', predScoreTotal, predWordsTotal)
            if tgtF:
                reportScore('GOLD', goldScoreTotal, goldWordsTotal)'''

        srcBatch, tgtBatch = [], []

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if tgtF:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()
    outF.close()
    if opt.output_representation:
        save_data = {
            "representations": torch.cat(representations),
            "src_rb": torch.cat(src_rb_list)
        }
        torch.save(save_data, opt.output_representation)

if __name__ == "__main__":
    main(None)
