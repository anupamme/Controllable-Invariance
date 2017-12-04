import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import translate
import tools
import os
from tools.Util import Logger
import sys
import random
from collections import deque
from onmt.Translator import translate_batch_external, idxToToken
import copy
import evaluate_file
import numpy

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-task', default="simp", help="simp or MT")
parser.add_argument('-data', default="newsela",
                    help='dataset')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from', help="""If training from a checkpoint then this is the path to the pretrained model.""")
parser.add_argument('-debug', action='store_true', help='debug')

## Model options
parser.add_argument('-random_seed', type=int, default=22)
parser.add_argument('-num_rb_bin', type=int, default=3, help="number of readability bin")
parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=256,
                    help='Size of LSTM hidden states')
parser.add_argument('-disc_size', type=int, default=128,
                    help='Size of disc hidden states')
parser.add_argument('-word_vec_size', type=int, default=128,
                    help='Word embedding sizes')
parser.add_argument('-rb_vec_size', type=int, default=64, help='readability embedding size')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-disc_obj_reverse', type=int, default=0,
                    help="reverting the objective of discriminator")
parser.add_argument('-label_smooth', type=int, default=0,
                    help="label smoothing")
parser.add_argument('-disc_type', type=str, default="DNN")
parser.add_argument('-filter_src_rb', type=int, default=None)
parser.add_argument('-no_brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")
parser.add_argument('-attn_type', type=str, default='global',
                    help='The type of attention to use')
parser.add_argument('-no_adv', action="store_true", help="no adversarial")
parser.add_argument('-use_rb_emb', type=int, default=1)
parser.add_argument('-adv_att', type=int, help="no attention model for adversarial", default=0)
parser.add_argument('-adv_param_init', type=float, default=0.1, help="uniform distribution range")
parser.add_argument('-adv_lambda', type=float, default=2., help="")
parser.add_argument('-adv_update_freq', type=int, default=2, help="")
parser.add_argument('-separate_update', type=int, default=0, help="separate updating instead of register hook")
parser.add_argument('-rb_init_token', type=int, help="use rb as the init token", default=0)
parser.add_argument('-rb_init_tgt', type=int, help="use rb as the init token", default=0)
parser.add_argument('-no_tgt_to_src', type=int, help="no double direction", default=1)
parser.add_argument('-AE_data_probability', type=float, default=0.0, help="AE data probability")
parser.add_argument('-AE_data_weight', type=float, default=0.3, help="AE data weight")
parser.add_argument('-extra_AE_decoder_weight', type=float, default=0.0, help="")
parser.add_argument('-reverse_src', type=int, default=0)

parser.add_argument('-only_AE', type=int, default=0, help="only use AE data")
parser.add_argument('-limit_disc_acc', type=float, default=0)
parser.add_argument('-min_disc_acc', type=float, default=0)
parser.add_argument('-separate_threshold', type=int, default=1)
parser.add_argument('-clear_buffer', type=int, default=0)
parser.add_argument('-non_linear', type=str, default="relu")
parser.add_argument('-parallel_ratio', type=float, default=None)
parser.add_argument('-adaptive_lambda', type=int, default=0)
parser.add_argument('-lambda_decay', type=float, default=1.)
parser.add_argument('-use_tgt_rb_emb', type=int, default=1)
parser.add_argument('-use_src_rb_emb', type=int, default=1)
parser.add_argument('-separate_encoder', type=int, default=0)

## Optimization options

parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=64,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='adam',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-learning_rate', type=float, default=0.001,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1""")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-decoder_context_dropout', type=float, default=0.0,
                    help="context dropout")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""Decay learning rate by this much if (i) perplexity
                    does not decrease on the validation set or (ii) epoch has
                    gone past the start_decay_at_limit""")
parser.add_argument('-start_decay_at', default=6,
                    help="Start decay after this epoch")
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-batch_norm', type=int, default=1)
parser.add_argument('-disc_bi_dir', type=int, default=1)
parser.add_argument('-adv_dropout_prob', type=float, default=0.5)
parser.add_argument('-disc_layer', type=int, default=3)
parser.add_argument('-adam_momentum', type=float, default=0.9)
parser.add_argument('-buffer_length', type=int, default=20)
parser.add_argument('-verbose', type=int, default=0)


# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA")
parser.add_argument('-log_interval', type=int, default=500,
                    help="Print stats at this interval.")

opt = parser.parse_args()
random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
numpy.random.seed(opt.random_seed)

opt_values = vars(opt)
opt.cuda = len(opt.gpus)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.random_seed)

opt.brnn = not opt.no_brnn
param_print = ""
for item in vars(parser.parse_args()):
    if item != "gpus" and item != "random_seed" and opt_values[item] != parser.get_default(item):
        param_print += item + "_" + str(opt_values[item]) + "-"
exp_path = os.path.dirname(os.path.realpath(__file__)) + "/../obj/exp-%stime-%s/" % (param_print, time.strftime("%y.%m.%d_%H.%M.%S"))
if opt.debug:
    #opt.data = "debug"
    pass
os.mkdir(exp_path)
opt.save_model = saveto = exp_path + "model"
logger = Logger(exp_path + "log.txt")
sys.stdout = logger
print "exp path", exp_path
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus [id1 id2]")

if opt.cuda:
    cuda.set_device(opt.gpus[0])

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.cuda:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, targets, disc_out, disc_target, generator, crit, discriminator_criterion, eval=False, print_prob=False, dicts=None):
    # compute generations one piece at a time
    dec_loss = 0
    disc_acc = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval).contiguous()
    #print outputs.size()
    batch_size = outputs.size(0)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets.contiguous(), opt.max_generator_batches)
    for out_t, targ_t in zip(outputs_split, targets_split):
        out_t = out_t.view(-1, out_t.size(2))
        pred_t = generator(out_t)
        if print_prob:
            print pred_t
        dec_loss_t = crit(pred_t, targ_t.view(-1))
        if print_prob:
            print dec_loss_t
        if print_prob:
            for j, seq in enumerate(targ_t.data.tolist()):
                for k, word in enumerate(seq):
                    print word, dicts['tgt'].getLabel(word), pred_t[j * len(seq) + k][word].data[0],
                print
        dec_loss += dec_loss_t.data[0]
        if not eval:
            dec_loss_t.div(batch_size).backward()
    grad_output = None if outputs.grad is None else outputs.grad.data

    if discriminator_criterion:
        disc_out = Variable(disc_out.data, requires_grad=(not eval), volatile=eval).contiguous()
        disc_out_split = torch.split(disc_out, opt.max_generator_batches)
        disc_target_split = torch.split(disc_target, opt.max_generator_batches)
        for disc_out_t, disc_target_t in zip(disc_out_split, disc_target_split):
            if opt.label_smooth:
                #print disc_target_t
                smooth_disc_target_t = disc_target_t.float().add(0.2)
                #ones = torch.ones(*disc_target_t.size())
                #if opt.cuda:
                #    ones = ones.cuda()
                #disc_target_t = torch.min(disc_target_t, ones)
                smooth_disc_target_t = smooth_disc_target_t.clamp(max=1)
                smooth_disc_target_t = smooth_disc_target_t.add(-0.1)
                disc_loss_t = discriminator_criterion(disc_out_t, smooth_disc_target_t)
                arg_max_index = torch.ge(disc_out_t, 0.5)
                disc_acc += torch.eq(arg_max_index.long(), disc_target_t).sum().data[0]
                if opt.verbose:
                    print "target", disc_target_t
                    print "prediction", arg_max_index
                    print "prob", disc_out_t
            else:
                if opt.disc_obj_reverse:
                    prob = -disc_out_t
                    prob = prob.add(1.000001)
                    prob = torch.log(prob)
                    disc_loss_t = -discriminator_criterion(prob, disc_target_t)
                    prob, arg_max_index = torch.kthvalue(disc_out_t.cpu(), opt.num_rb_bin)
                else:
                    disc_loss_t = discriminator_criterion(disc_out_t, disc_target_t)
                    prob, arg_max_index = torch.kthvalue(disc_out_t.cpu(), opt.num_rb_bin)
                    prob = torch.exp(prob)
                if opt.verbose:
                    #print "target", disc_target_t
                    #print "prediction", arg_max_index
                    print "prob", prob

                disc_acc += torch.eq(arg_max_index.squeeze(1), disc_target_t.cpu()).sum().data[0]
            if not eval:
                disc_loss_t.div(batch_size).backward()
            if not eval and random.random() < 0.02:
                #print arg_max_index
                #print disc_target_t
                #print prob
                pass
        grad_disc_out = None if disc_out.grad is None else disc_out.grad.data
    else:
        grad_disc_out = None
    #print disc_acc
    return dec_loss, disc_acc, grad_output, grad_disc_out


def eval(models, criterion, disc_criterion, data, dicts, weights):
    total_loss = 0
    total_words = 0
    total_disc_acc = 0
    total_sent = 0
    for i, single_data in enumerate(data.datas):
        models[i].eval()
        for k, batch in enumerate(single_data):
            #batch = [x.transpose(0, 1) for x in batch] # must be batch first for gather/scatter in DataParallel
            batch = [batch[0].transpose(0, 1), batch[1].transpose(0, 1), batch[2], batch[3]]
            outputs, disc_outputs, dec_norm, adv_norm = models[i](batch)  # FIXME volatile
            targets = batch[1][:, 1:]  # exclude <s> from targets
            dec_loss, disc_acc, _, _ = memoryEfficientLoss(
                    outputs, targets, disc_outputs, batch[2], models[i].generator, criterion, disc_criterion, eval=True,
                    print_prob=(k == 0) and opt.verbose, dicts=dicts)
            total_loss += dec_loss * weights[i]
            total_words += targets.data.ne(onmt.Constants.PAD).sum()
            total_disc_acc += disc_acc
            total_sent += targets.size(0)
            if opt.debug:
                break
        models[i].train()
    return total_loss / total_words, total_disc_acc * 1. / total_sent

def generate_sample(st, valid_sample_batch, dataset, model):
    print "sampling", st
    sample_size = valid_sample_batch[0].size(0)
    rb_samples = []
    n_best = 1
    for target_rb in range(opt.num_rb_bin):
        new_sample_batch = [valid_sample_batch[0], valid_sample_batch[1], valid_sample_batch[2], valid_sample_batch[3]]
        target_rb_batch = torch.ones((sample_size,)).long() * target_rb
        if opt.cuda:
            target_rb_batch = target_rb_batch.cuda()
        target_rb_batch = Variable(target_rb_batch)
        new_sample_batch[3] = target_rb_batch

        pred, predScore, attn, goldScore, representation = translate_batch_external(new_sample_batch, 5, model, opt.cuda, opt.rb_init_token, opt.rb_init_tgt, 100, n_best)

        #  (3) convert indexes to words
        predBatch = []
        for b in range(new_sample_batch[0].size(0)):
            predBatch.append(
                [idxToToken(pred[b][n], new_sample_batch[0][b], attn[b][n], dataset['dicts']['tgt'], False, opt.rb_init_token)
                 for n in range(n_best)]
            )
        rb_samples += [predBatch]
    for j in range(sample_size):
        print "source: ", " ".join(dataset['dicts']['src'].convertToLabels(new_sample_batch[0].data[j].tolist(), onmt.Constants.PAD))
        print "source class:", new_sample_batch[2].data[j]
        print "target: ", " ".join(dataset['dicts']['tgt'].convertToLabels(new_sample_batch[1].data[j].tolist(), onmt.Constants.PAD))
        for k in range(opt.num_rb_bin):
            print "\tsample", k, " ".join(rb_samples[k][j][0])


def trainModel(models, trainData, validData, dataset, optims, dicts, weights, valid_weight, threshold):
    print(models[0])
    for single_model in models:
        single_model.train()
    #if optim.last_ppl is None:
    #    for p in model.parameters():
    #        p.data.uniform_(-opt.param_init, opt.param_init)

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())
    if opt.no_adv:
        discriminator_criterion = None
    else:
        if opt.label_smooth:
            discriminator_criterion = nn.BCELoss(size_average=False)
        else:
            discriminator_criterion = nn.NLLLoss(size_average=False)
            if opt.cuda:
                discriminator_criterion.cuda()
    start_time = time.time()
    def get_sample_data(valid_sample_batch):
        sample_lim = 10
        valid_sample_batch = [valid_sample_batch[0].transpose(0, 1), valid_sample_batch[1].transpose(0, 1), valid_sample_batch[2], valid_sample_batch[3]]
        valid_sample_batch = [valid_sample_batch[0][:sample_lim], valid_sample_batch[1][:sample_lim], valid_sample_batch[2][:sample_lim], valid_sample_batch[3][:sample_lim]]
        return valid_sample_batch
    valid_sample_batchs = []
    for vd in validData.datas:
        valid_sample_batchs += [get_sample_data(vd.random_batch(sort=True))]
    train_sample_batch = get_sample_data(trainData.random_batch(sort=True))
    #caches = sorted(caches, key = lambda (s, t, s_rb, t_rb): len(s), reverse=True)
    def trainEpoch(epoch):
        report_ce_loss, report_attn_loss = 0, 0
        total_loss, report_loss = 0, 0
        total_words, report_words = 0, 0
        total_disc_acc, report_disc_acc = 0, 0
        total_sent, report_sent = 0, 0
        report_seq2seq_gn, report_seq2seq_enc_gn = 0, 0
        report_disc_gn, report_disc_enc_gn = 0, 0
        start = time.time()
        disc_acc_buffer_q = deque()
        batch_size_buffer_q = deque()
        disc_acc_buffer = 0
        batch_size_buffer = 0
        buffer_length = opt.buffer_length
        for i, (batch, data_source_index) in enumerate(trainData):
            #batch = [x.transpose(0, 1) for x in batch] # must be batch first for gather/scatter in DataParallel
            if opt.separate_encoder:
                model_index = data_source_index
            else:
                model_index = 0
            batch = [batch[0].transpose(0, 1), batch[1].transpose(0, 1), batch[2], batch[3]]
            if i % opt.adv_update_freq == 0 and i != 0 and disc_acc_buffer * 1. / batch_size_buffer >  1. / opt.num_rb_bin * opt.min_disc_acc:
                ll = opt.adv_lambda
                if opt.verbose:
                    print "lambda neq", 0
                if opt.adaptive_lambda:
                    assert not opt.clear_buffer
                    ll /= (1.00001 - disc_acc_buffer * 1. / batch_size_buffer)
            else:
                ll = 0
            if opt.verbose:
                print "adv lambda", ll
            '''for j in range(1):#range(batch[0].size(0)):
                print weights[data_source_index]
                print batch[2].data[j], " ".join(dataset['dicts']['src'].convertToLabels(batch[0].data[j].tolist(), onmt.Constants.EOS))
                print batch[3].data[j], " ".join(dataset['dicts']['tgt'].convertToLabels(batch[1].data[j].tolist(), onmt.Constants.EOS))'''
            models[model_index].zero_grad()
            outputs, attns, disc_out, dec_norm, adv_norm = models[model_index](batch, return_attn=True, grad_scale=ll)
            targets = batch[1][:, 1:]  # exclude <s> from targets
            dec_loss, disc_acc, gradOutput, grad_disc_output = memoryEfficientLoss(
                    outputs, targets, disc_out, batch[2], models[model_index].generator, criterion, discriminator_criterion, print_prob=False, dicts=dataset['dicts'])
        
            dec_loss *= weights[data_source_index]
            gradOutput = gradOutput * weights[data_source_index]
            if len(disc_acc_buffer_q) == buffer_length:
                disc_acc_buffer -= disc_acc_buffer_q.popleft()
                batch_size_buffer -= batch_size_buffer_q.popleft()
            disc_acc_buffer_q.append(disc_acc)
            batch_size_buffer_q.append(targets.size(0))
            disc_acc_buffer += disc_acc
            batch_size_buffer += targets.size(0)
            if grad_disc_output is not None:
                #grad_disc_output = grad_disc_output * weights[data_source_index]
                grad_disc_output = grad_disc_output * 1.
                #TBC does not put low weight on auto encoding data
            if opt.separate_update:
                outputs.backward(gradOutput, retain_variables=True)
                enc_grad_norm, grad_norm = optims[model_index].step_seq2seq()
                report_seq2seq_gn += grad_norm
                report_seq2seq_enc_gn += enc_grad_norm
                models[model_index].zero_grad()
                disc_out.backward(grad_disc_output)
                disc_enc_grad_norm, disc_grad_norm = optims[model_index].step_disc_enc(ll)
                report_disc_gn += disc_grad_norm
                report_disc_enc_gn += disc_enc_grad_norm
            else:
                if opt.no_adv:
                    outputs.backward(gradOutput)
                else:
                    torch.autograd.backward([outputs, disc_out], [gradOutput, grad_disc_output])
                #disc_out.backward(grad_disc_output)
                #print "disc_acc_buffer", disc_acc_buffer * 1. / batch_size_buffer,len(disc_acc_buffer_q), batch[0].size(1)

                #if opt.verbose:
                if opt.verbose:
                    print i, disc_acc_buffer * 1. / batch_size_buffer
                if opt.limit_disc_acc < 1e-7 or len(disc_acc_buffer_q) >= buffer_length and disc_acc_buffer * 1. / batch_size_buffer < 1. / opt.num_rb_bin * opt.limit_disc_acc:
                    if opt.verbose:
                        print "clean", i
                    _, grad_norm = optims[model_index].step_all()
                    if opt.clear_buffer:
                        disc_acc_buffer_q.clear()
                        batch_size_buffer_q.clear()
                        disc_acc_buffer = 0
                        batch_size_buffer = 0
                else:
                    _, grad_norm = optims[model_index].step_seq2seq()
                report_seq2seq_gn += grad_norm
                report_seq2seq_enc_gn += dec_norm[0]
                report_disc_enc_gn += adv_norm[0] * opt.adv_lambda
                if opt.verbose:
                    print "dec norm", dec_norm[0], "adv norm", adv_norm[0]
            # prediction = model.generator(outputs.contiguous().view(-1, outputs.size(2)))
            # loss = criterion(prediction, targets.contiguous().view(-1))
            # ce_loss = loss / opt.batch_size

            # attn_src = (attns.sum(1) - 1.).squeeze(1) ** 2
            # mask_src = batch[0].ne(onmt.Constants.PAD).float()
            # attn_loss = torch.mean(torch.sum(attn_src * mask_src, 1))

            # backward_loss = ce_loss + 0.0 * attn_loss
            # backward_loss.backward()

            # loss = loss.data[0]

            # update the parameters
            # report_ce_loss += ce_loss.data[0]
            # report_attn_loss += attn_loss.data[0]

            report_loss += dec_loss
            total_loss += dec_loss
            report_disc_acc += disc_acc
            total_disc_acc += disc_acc
            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            total_words += num_words
            report_words += num_words
            #print outputs.size()
            report_sent += targets.size(0)
            total_sent += targets.size(0)
            #print report_sent
            if i % opt.log_interval == 0 and i > 0:
                #print disc_out, batch[2]
                for k, vb in enumerate(valid_sample_batchs):
                    generate_sample("valid %d" % k, vb, dataset, models[k])
                generate_sample("train", train_sample_batch, dataset, models[0])
                print("Epoch %2d, %5d batches; perplexity: %6.2f; disc prob: %f; grad_norm(s2s, adv): %6.2f, %6.2f;  enc_grad_norm(s2s, adv): %f, %f; %3.0f tokens/s; %6.0f s elapsed" %
                      (epoch, i,
                      math.exp(report_loss / report_words),
                      report_disc_acc * 1. / report_sent,
                      report_seq2seq_gn * 1. / opt.log_interval, report_disc_gn * 1. / opt.log_interval,
                      report_seq2seq_enc_gn * 1. / opt.log_interval, report_disc_enc_gn * 1. / opt.log_interval,
                      report_words/(time.time()-start),
                      time.time()-start_time))
                report_loss = report_words = 0
                report_disc_acc = report_sent = 0
                report_seq2seq_gn, report_seq2seq_enc_gn = 0, 0
                report_disc_gn, report_disc_enc_gn = 0, 0
                start = time.time()
            if opt.debug:
                break
        return total_loss / total_words, total_disc_acc * 1. / total_sent
    history_valid = []
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')
        #handle = discriminator_fist_layer.register_backward_hook(hook_func)
        #  (1) train for one epoch on the training set
        if epoch != 0 and epoch >= opt.start_decay_at:
            opt.adv_lambda *= opt.lambda_decay
            print "decay lambda to", opt.adv_lambda
        train_loss, disc_acc = trainEpoch(epoch)
        print('Train perplexity: %g, disc probability %g' % (math.exp(min(train_loss, 100)), disc_acc))

        #  (2) evaluate on the validation set
        valid_loss, valid_disc_acc = eval(models, criterion, discriminator_criterion, validData, dicts, valid_weight)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g, disc probability %g' % (valid_ppl, valid_disc_acc))

        #  (3) maybe update the learning rate
        if opt.optim == 'sgd':
            for single_optim in optims:
                single_optim.updateLearningRate(valid_loss, epoch)

        #  (4) drop a checkpoint
        #handle.remove()
        checkpoint = {
            'models': models,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optims': optims,
            'threshold': threshold,
            'unk_offset': dataset['unk_offset'],
        }
        if opt.task == "Multi-MT":
            checkpoint["src_language_mapping"] = dataset["src_language_mapping"]
            checkpoint["tgt_language_mapping"] = dataset["tgt_language_mapping"]
        else:
            checkpoint['tgt_rb_offset'] = dataset['tgt_rb_offset']
        #torch.save(checkpoint,
        #           '%s_e%d_%.2f.pt' % (opt.save_model, epoch, valid_ppl))
        if len(history_valid) == 0 or valid_ppl < min(history_valid):
            print "best validation performance"
            torch.save(checkpoint,
                   '%s.pt' % (opt.save_model))
        history_valid.append(valid_ppl)

def main():
    data = "../../data_%s/%s/%s-train.pt" % (opt.task, opt.data, opt.data)
    print("Loading data from '%s'" % data)
    if opt.label_smooth:
        assert opt.num_rb_bin == 2
    dataset = torch.load(data)
    if opt.separate_threshold:
        print dataset["src_threshold"]
        print dataset["tgt_threshold"]
        threshold = {"src": dataset["src_threshold"][opt.num_rb_bin], "tgt": dataset["tgt_threshold"][opt.num_rb_bin]}
    else:
        if opt.num_rb_bin > 0:
            single_threshold = dataset['all_threshold'][opt.num_rb_bin]
        else:
            single_threshold = [0]
        threshold = {"src": single_threshold, "tgt": single_threshold}
    print threshold
    dicts = dataset['dicts']
    ori_datasets = copy.deepcopy(dataset)
    if opt.parallel_ratio is not None:
        parallel_len = l = int(len(dataset['train']['src']) * opt.parallel_ratio)
        dataset['train']['src'] = dataset['train']['src'][:l]
        print dataset['train']['src'][-1]
        dataset['train']['tgt'] = dataset['train']['tgt'][:l]
        dataset['train']['src_rb'] = dataset['train']['src_rb'][:l]
        dataset['train']['tgt_rb'] = dataset['train']['tgt_rb'][:l]
    else:
        parallel_len = None
    if opt.separate_encoder == 0:
        forward_data = onmt.BucketIterator(dataset['train']['src'],
                                        dataset['train']['tgt'], dataset['train']['src_rb'], dataset['train']['tgt_rb'], opt, threshold)
        valid_data = onmt.BucketIterator(dataset['valid']['src'],
                                         dataset['valid']['tgt'], dataset['valid']['src_rb'], dataset['valid']['tgt_rb'], opt, threshold)
        valid_datas = [valid_data]
        valid_weight = [1.]
        valid_probability = [1.]
        train_datas = [forward_data]
        probability = [1.]
        weights = [1.]
        print len(forward_data)
    else:
        opt.filter_src_rb = 0
        forward_data = onmt.BucketIterator(dataset['train']['src'],
                                           dataset['train']['tgt'], dataset['train']['src_rb'], dataset['train']['tgt_rb'], opt, threshold)
        #print len(forward_data)
        valid_data = onmt.BucketIterator(dataset['valid']['src'],
                                         dataset['valid']['tgt'], dataset['valid']['src_rb'], dataset['valid']['tgt_rb'], opt, threshold)
        valid_datas = [valid_data]
        valid_weight = [1.]
        valid_probability = [1.]
        train_datas = [forward_data]
        probability = [1.]
        weights = [1.]

        opt.filter_src_rb = 1
        forward_data = onmt.BucketIterator(dataset['train']['src'],
                                           dataset['train']['tgt'], dataset['train']['src_rb'], dataset['train']['tgt_rb'], opt, threshold)
        valid_data = onmt.BucketIterator(dataset['valid']['src'],
                                         dataset['valid']['tgt'], dataset['valid']['src_rb'], dataset['valid']['tgt_rb'], opt, threshold)
        valid_datas += [valid_data]
        valid_weight += [1.]
        valid_probability += [1.]
        train_datas += [forward_data]
        probability += [1.]
        weights += [1.]
        opt.filter_src_rb = None

    if not opt.no_tgt_to_src:
        backwardData = onmt.BucketIterator(dataset['train_bi']['src'], dataset['train_bi']['tgt'], dataset['train_bi']['src_rb'], dataset['train_bi']['tgt_rb'],
                                           opt, threshold)
        train_datas.append(backwardData)
        weights.append(1.)
        probability = [0.5, 0.5]
    trainData = onmt.mixed_iterator(train_datas, probability)
    validData = onmt.mixed_iterator(valid_datas, valid_probability)

    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.train_from is None:
        decoder = onmt.Models.Decoder(opt, dicts['tgt'], attn_type=opt.attn_type)
        generator = nn.Sequential(
            nn.Linear(opt.rnn_size, dicts['tgt'].size()),
            nn.LogSoftmax())
        if opt.cuda > 1:
            generator = nn.DataParallel(generator, device_ids=opt.gpus)
        discriminator = onmt.Models.Discriminator(opt)
        if not opt.separate_encoder:
            encoder = onmt.Models.Encoder(opt, dicts['src'])
            models = [onmt.Models.NMTModel(encoder, decoder, generator, discriminator, opt)]
        else:
            models = []
            for i in range(opt.num_rb_bin):
                encoder = onmt.Models.Encoder(opt, dicts['src'])
                models += [onmt.Models.NMTModel(encoder, decoder, generator, discriminator, opt)]
        optims = []
        for model_single in models:
            if opt.cuda > 1:
                model_single = nn.DataParallel(model_single, device_ids=opt.gpus)
            if opt.cuda:
                model_single.cuda()
            else:
                model_single.cpu()
            model_single.generator = generator
            for p in model_single.get_seq2seq_parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)
            for p in model_single.get_disc_parameters():
                if opt.non_linear == "relu":
                    opt.adv_para_init = 2. / opt.disc_size
                p.data.uniform_(-opt.adv_param_init, opt.adv_param_init)
            optim_single = onmt.Optim(
                model_single.parameters(), model_single.get_seq2seq_parameters(), model_single.get_disc_parameters(), model_single.get_encoder_parameters(),
                opt.optim, opt.learning_rate, opt.max_grad_norm,
                lr_decay=opt.learning_rate_decay,
                start_decay_at=opt.start_decay_at,
                adam_momentum=opt.adam_momentum,
            )
            optims += [optim_single]
    else:
        print('Loading from checkpoint at %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from)
        model_single = checkpoint['model']
        if opt.cuda:
            model_single.cuda()
        else:
            model_single.cpu()
        optim_single = checkpoint['optim']
        opt.start_epoch = checkpoint['epoch'] + 1

    nParams = sum([p.nelement() for model_single in models for p in model_single.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(models, trainData, validData, dataset, optims, dicts, weights, valid_weight, threshold)
    if opt.task == "MT":
        translate.main(["-task", opt.task, "-data", opt.data, "-model", "%s/model.pt" % exp_path, "-replace_unk", "-gpus", str(opt.gpus[0]), "-output", "%s/test_no_unk.txt" % exp_path, "-verbose"])
        evaluate_file.main(["-task", opt.task, "-data", opt.data, "-outputs", "%s/test_no_unk.txt" % exp_path])
    elif opt.task == "Multi-MT":
        for test_set in ["test"]:
            for language_pair in dataset["language_pairs"]:
                line = language_pair.split("-")
                S_lang = line[0]
                T_lang = line[1]
                print "test_set", test_set + "_" + language_pair
                if opt.filter_src_rb is None or opt.filter_src_rb == dataset["src_language_mapping"][S_lang]:
                    translate.main(["-task", opt.task, "-data", opt.data, "-model", "%s/model.pt" % exp_path, "-replace_unk", "-gpus", str(opt.gpus[0]), "-output",
                                    "%s/%s_%s_no_unk.txt" % (exp_path, test_set, language_pair), "-verbose", "-language_pair", language_pair, "-test_set", test_set, "-bpe"])

                    evaluate_file.main(["-task", opt.task, "-data", opt.data, "-outputs", "%s/%s_%s_no_unk.txt" % (exp_path, test_set, language_pair), "-language_pair", language_pair, "-test_set", test_set])
                else:
                    print "BLEU  0.0, SARI   0.00, R1   0.00, R2   0.00, RL   0.00, FK_O   0.0, acc   0.00"
    else:
        for i in range(opt.num_rb_bin):
            translate.main(["-task", opt.task, "-data", opt.data, "-model", "%s/model.pt" % exp_path, "-replace_unk", "-gpus", str(opt.gpus[0]), "-output", "%s/test_no_unk.txt" % exp_path, "-verbose", "-tgt_rb_all", str(i)])
            evaluate_file.main(["-task", opt.task, "-data", opt.data, "-outputs", "%s/test_no_unk.txt" % exp_path, "-single_rb", str(i)])
            print "all rb", i
        translate.main(["-task", opt.task, "-data", opt.data, "-model", "%s/model.pt" % exp_path, "-replace_unk", "-gpus", str(opt.gpus[0]), "-output", "%s/test_no_unk.txt" % exp_path, "-verbose"])
        evaluate_file.main(["-task", opt.task, "-data", opt.data, "-outputs", "%s/test_no_unk.txt" % exp_path])

if __name__ == "__main__":
    main()
