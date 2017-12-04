import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
import torch.nn.utils.rnn as rnn_utils
import math

def check_decreasing(lengths):
    lens, order = torch.sort(lengths, 0, True) 
    if torch.ne(lens, lengths).sum() == 0:
        return None
    else:
        _, rev_order = torch.sort(order)

        return lens, Variable(order), Variable(rev_order)

class Encoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        #self.hidden_size = opt.rnn_size
        inputSize = opt.word_vec_size
        self.opt = opt
        super(Encoder, self).__init__()
        if opt.rb_init_token:
            self.rb_lut = None
            self.word_lut = nn.Embedding(dicts.size() + opt.num_rb_bin,
                                         opt.word_vec_size,
                                         padding_idx=onmt.Constants.PAD)
        else:
            self.word_lut = nn.Embedding(dicts.size(),
                                          opt.word_vec_size,
                                          padding_idx=onmt.Constants.PAD)
            if opt.num_rb_bin > 0 and opt.use_rb_emb and opt.use_src_rb_emb:
                self.rb_lut = nn.Embedding(opt.num_rb_bin, opt.rb_vec_size)
                inputSize += opt.rb_vec_size
            else:
                self.rb_lut = None
        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                           num_layers=opt.layers,
                           dropout=opt.dropout,
                           bidirectional=opt.brnn)
        # self.rnn.bias_ih_l0.data.div_(2)
        # self.rnn.bias_hh_l0.data.copy_(self.rnn.bias_ih_l0.data)
        self.dict_size = dicts.size()
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.copy_(pretrained)

    def forward(self, input, input_rb, hidden=None):
        batch_size = input.size(0)                  # [batch x sourceL] batch first for multi-gpu compatibility
        if self.opt.rb_init_token:
            input = torch.cat([input_rb.unsqueeze(1) + self.dict_size, input], 1)
        emb = self.word_lut(input).transpose(0, 1)  # [sourceL x batch x emb_size]
        if self.rb_lut is not None:
            rb_emb = self.rb_lut(input_rb) #[batch x emb_size]
            seq_len = emb.size(0)
            emb = torch.cat([emb, rb_emb.unsqueeze(0).expand(seq_len, *rb_emb.size())], 2)

        # if hidden is None:
        #     h_size = (self.layers * self.num_directions, batch_size, self.hidden_size)
        #     h_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
        #     c_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
        #     hidden = (h_0, c_0)

        # outputs, hidden_t = self.rnn(emb, hidden)

        lengths = input.data.ne(onmt.Constants.PAD).sum(1).squeeze(1)
        check_res = check_decreasing(lengths)
        if check_res is None:
            packed_emb = rnn_utils.pack_padded_sequence(emb, lengths.tolist())
            packed_out, hidden_t = self.rnn(packed_emb)
            outputs, srclens = rnn_utils.pad_packed_sequence(packed_out)
        else:
            lens, order, rev_order = check_res
            packed_emb = rnn_utils.pack_padded_sequence(emb.index_select(1, order), lens.tolist())
            packed_out, hidden_t = self.rnn(packed_emb)
            outputs, srclens = rnn_utils.pad_packed_sequence(packed_out)
            outputs = outputs.index_select(1, rev_order)
            hidden_t = (hidden_t[0].index_select(1, rev_order), 
                        hidden_t[1].index_select(1, rev_order))

        return hidden_t, outputs

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.adv_att = None
        self.opt = opt
        self.disc_type = opt.disc_type
        if opt.no_adv:
            self.discriminator = None
        else:
            if opt.disc_type == "DNN":
                init_in_size = opt.rnn_size
            elif opt.disc_type == "RNN":
                init_in_size = opt.disc_size
                self.num_directions = 2 if opt.disc_bi_dir else 1
                self.rnn = nn.LSTM(opt.rnn_size, opt.disc_size // self.num_directions,
                                        num_layers=1,
                                        dropout=opt.dropout,
                                        bidirectional=opt.disc_bi_dir)
            elif opt.disc_type == "CNN":
                assert False
            else:
                assert False
            if opt.adv_att:
                self.adv_att = onmt.modules.SelfAttention(init_in_size)
            modules = []
            for i in range(opt.disc_layer):
                if i == 0:
                    in_size = init_in_size
                else:
                    in_size = opt.disc_size
                modules += [nn.Linear(in_size, opt.disc_size)]
                if opt.batch_norm:
                    modules += [nn.BatchNorm1d(opt.disc_size)]
                if opt.non_linear == "tanh":
                    modules += [nn.Tanh()]
                elif opt.non_linear == "relu":
                    modules += [nn.ReLU()]
                else:
                    assert False
                modules += [nn.Dropout(opt.adv_dropout_prob)]
            if opt.label_smooth:
                modules += [nn.Linear(opt.disc_size, 1)]
                modules += [nn.Sigmoid()]
            else:
                modules += [nn.Linear(opt.disc_size, opt.num_rb_bin)]
                if opt.disc_obj_reverse:
                    modules += [nn.Softmax()]
                else:
                    modules += [nn.LogSoftmax()]
            self.dnn = nn.Sequential(*modules)

    def forward(self, input, context, grad_scale):
        adv_norm = []
        if self.opt.no_adv:
            disc_out = None
            adv_norm.append(0)
        else:
            adv_context_variable = torch.mul(context, 1)
            if not self.opt.separate_update:
                adv_context_variable.register_hook(adv_wrapper(adv_norm, grad_scale))
            else:
                adv_norm.append(0)
            if self.disc_type == "DNN":
                adv_context_variable = adv_context_variable.t().contiguous()
                padMask = input.eq(onmt.Constants.PAD)
                if self.opt.rb_init_token:
                    rb_token_mask = Variable(torch.zeros(padMask.size(0), 1).byte())
                    if self.opt.cuda:
                        rb_token_mask = rb_token_mask.cuda()
                    padMask = torch.cat([rb_token_mask, padMask], 1)
                if self.adv_att:
                    self.adv_att.applyMask(padMask.data) #let it figure out itself. Backprop may have problem if not
                    averaged_context = self.adv_att(adv_context_variable)
                else:
                    padMask = 1. - padMask.float() #batch * sourceL
                    masked_context = adv_context_variable * padMask.unsqueeze(2).expand(padMask.size(0), padMask.size(1), context.size(2))
                    sent_len = torch.sum(padMask, 1).squeeze(1)
                    averaged_context = torch.div(torch.sum(masked_context, 1).squeeze(1), sent_len.unsqueeze(1).expand(sent_len.size(0), context.size(2)))
                disc_out = self.dnn(averaged_context)
            elif self.disc_type == "RNN":
                lengths = input.data.ne(onmt.Constants.PAD).sum(1).squeeze(1)
                check_res = check_decreasing(lengths)
                if check_res is None:
                    packed_emb = rnn_utils.pack_padded_sequence(adv_context_variable, lengths.tolist())
                    packed_out, hidden_t = self.rnn(packed_emb)
                    if self.adv_att:
                        assert False
                        outputs, srclens = rnn_utils.pad_packed_sequence(packed_out)
                    else:
                        hidden_t = (_fix_enc_hidden(hidden_t[0], self.num_directions)[-1],
                                    _fix_enc_hidden(hidden_t[1], self.num_directions)[-1]) #The first one is h, the other one is c
                        #print hidden_t[0].size(), hidden_t[1].size()
                        #hidden_t = torch.cat(hidden_t, 1)
                        #print hidden_t.size()
                        disc_out = self.dnn(hidden_t[0])
                else:
                    assert False
            else:
                assert False
        return disc_out, adv_norm

class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        for i in range(num_layers):
            layer = nn.LSTMCell(input_size, rnn_size)
            self.add_module('layer_%d' % i, layer)
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i in range(self.num_layers):
            layer = getattr(self, 'layer_%d' % i)
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)

class Decoder(nn.Module):

    def __init__(self, opt, dicts, attn_type='global'):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size
        self.opt = opt
        self.dict_size = dicts.size()
        super(Decoder, self).__init__()
        if opt.rb_init_tgt:
            self.word_lut = nn.Embedding(dicts.size() + opt.num_rb_bin,
                                         opt.word_vec_size,
                                         padding_idx=onmt.Constants.PAD)
            self.rb_lut = None
        else:
            self.word_lut = nn.Embedding(dicts.size(),
                                         opt.word_vec_size,
                                         padding_idx=onmt.Constants.PAD)
            if opt.num_rb_bin > 0 and opt.use_rb_emb and opt.use_tgt_rb_emb:
                self.rb_lut = nn.Embedding(opt.num_rb_bin, opt.rb_vec_size)
                input_size += opt.rb_vec_size
            else:
                self.rb_lut = None
        if self.input_feed:
            self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        else:
            self.rnn = nn.LSTM(input_size, opt.rnn_size, num_layers=opt.layers, dropout=opt.dropout)
        if attn_type.lower() == 'global':
            self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        elif attn_type.lower() == 'cosine':
            self.attn = onmt.modules.CosineAttention(opt.rnn_size)
        elif attn_type.lower() == 'mlp':
            self.attn = onmt.modules.MLPAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.context_dropout = nn.Dropout(opt.decoder_context_dropout)
        # self.rnn.bias_ih.data.div_(2)
        # self.rnn.bias_hh.data.copy_(self.rnn.bias_ih.data)

        self.hidden_size = opt.rnn_size

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.copy_(pretrained)


    def forward(self, input, input_rb, hidden, context, init_output):
        emb = self.word_lut(input).transpose(0, 1)
        context = self.context_dropout(context)
        if self.rb_lut is not None:
            #print input_rb
            rb_emb = self.rb_lut(input_rb) #[batch x emb_size]
            #print rb_emb
            seq_len = emb.size(0)
            emb = torch.cat([emb, rb_emb.unsqueeze(0).expand(seq_len, *rb_emb.size())], 2)
        batch_size = input.size(0)

        h_size = (batch_size, self.hidden_size)
        output = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        attns = []
        output = init_output
        if self.input_feed:
            for i, emb_t in enumerate(emb.chunk(emb.size(0), dim=0)):
                emb_t = emb_t.squeeze(0)
                if self.input_feed:
                    emb_t = torch.cat([emb_t, output], 1)
                output, h = self.rnn(emb_t, hidden)
                output, attn = self.attn(output, context.t())
                output = self.dropout(output)
                outputs += [output]
                attns.append(attn)
                hidden = h
        else:
            rnn_out, h = self.rnn(emb, hidden)
            for i, rnn_out_t in enumerate(rnn_out.split(split_size=1, dim=0)):
                output, attn = self.attn(rnn_out_t.squeeze(0), context.t())
                output = self.dropout(output)
                outputs += [output]
                attns.append(attn)
        outputs = torch.stack(outputs)
        attns = torch.stack(attns)
        return outputs.transpose(0, 1), h, attns.transpose(0, 1) #it becomes batch * targetL * embedding

def _fix_enc_hidden(h, num_directions):
    #  the encoder hidden is  (layers*directions) x batch x dim
    #  we need to convert it to layers x batch x (directions*dim)
    if num_directions == 2:
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
            .transpose(1, 2).contiguous() \
            .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
    else:
        return h

class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator, discriminator, opt):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.generate = False
        self.discriminator = discriminator
        self.opt = opt
        self.adv_grad_norm = 0
        self.dec_grad_norm = 0

    def get_seq2seq_parameters(self):
        for comp in [self.encoder, self.decoder, self.generator]:
            for p in comp.parameters():
                yield p

    def get_disc_parameters(self):
        for comp in [self.discriminator]:
            if comp is None:
                continue
            for p in comp.parameters():
                yield p

    def get_encoder_parameters(self):
        for comp in [self.encoder]:
            for p in comp.parameters():
                yield p

    def set_generate(self, enabled):
        self.generate = enabled

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def forward(self, input, return_attn=False, grad_scale=None):
        src = input[0]
        tgt = input[1][:, :-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, input[2])
        init_output = self.make_init_decoder_output(context)
        #how does it works
        enc_hidden = (_fix_enc_hidden(enc_hidden[0], self.encoder.num_directions),
                      _fix_enc_hidden(enc_hidden[1], self.encoder.num_directions))
        dec_context_variable = torch.mul(context, 1)
        dec_norm = []
        dec_context_variable.register_hook(dec_wrapper(dec_norm))
        if self.opt.no_adv:
            disc_out = None
            adv_norm = [0]
        else:
            disc_out, adv_norm = self.discriminator(input[0], context, grad_scale)
        if self.opt.rb_init_tgt:
            tgt = torch.cat([input[3].unsqueeze(1) + self.decoder.dict_size, tgt[:, 1:]], 1)
        out, dec_hidden, attn = self.decoder(tgt, input[3], enc_hidden, dec_context_variable, init_output)
        if self.generate:
            out = self.generator(out)
        if return_attn:
            return out, attn, disc_out, dec_norm, adv_norm
        else:
            return out, disc_out, dec_norm, adv_norm

def dec_wrapper(norm):
    def hook_func(grad):
        norm.append(math.pow(grad.norm().data[0], 2))
        pass
    return hook_func

def adv_wrapper(norm, grad_scale):
    def hook_func(grad):
        new_grad = -grad * grad_scale
        #print new_grad
        norm.append(math.pow(new_grad.norm().data[0], 2))
        return new_grad
        pass
    return hook_func

torch.backends.cudnn.enabled = False
