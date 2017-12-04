import onmt
import torch
from torch.autograd import Variable
import tools.readability
from Models import _fix_enc_hidden

def translate_batch_external(batch, beamSize, model, cuda, rb_init_token, rb_init_tgt, max_sent_length, n_best):
    srcBatch, tgtBatch, src_rb, tgt_rb = batch
    batchSize = srcBatch.size(0)

    #  (1) run the encoder on the src

    # padding is dealt with by variable-length cudnn.RNN
    encStates, context = model.encoder(srcBatch, src_rb)
    # # have to execute the encoder manually to deal with padding
    # encStates = None
    # context = []
    # for srcBatch_t in srcBatch.chunk(srcBatch.size(1), dim=1):
    #     encStates, context_t = self.model.encoder(srcBatch_t, hidden=encStates)
    #     batchPadIdx = srcBatch_t.data.squeeze(1).eq(onmt.Constants.PAD).nonzero()
    #     if batchPadIdx.nelement() > 0:
    #         batchPadIdx = batchPadIdx.squeeze(1)
    #         encStates[0].data.index_fill_(1, batchPadIdx, 0)
    #         encStates[1].data.index_fill_(1, batchPadIdx, 0)
    #     context += [context_t]

    # context = torch.cat(context)

    rnnSize = context.size(2)

    encStates = (_fix_enc_hidden(encStates[0], model.encoder.num_directions),
                 _fix_enc_hidden(encStates[1], model.encoder.num_directions))

    #  This mask is applied to the attention model inside the decoder
    #  so that the attention ignores source padding
    padMask = srcBatch.data.eq(onmt.Constants.PAD)
    rb_token_mask = torch.zeros(padMask.size(0), 1).byte()
    if cuda:
        rb_token_mask = rb_token_mask.cuda()
    if rb_init_token:
        padMask = torch.cat([rb_token_mask, padMask], 1)
    def applyContextMask(m):
        if isinstance(m, onmt.modules.GlobalAttention):
            m.applyMask(padMask)

    #  (2) if a target is specified, compute the 'goldScore'
    #  (i.e. log likelihood) of the target under the model
    goldScores = context.data.new(batchSize).zero_()
    re_padMask = 1 - padMask
    re_padMask = re_padMask.float()
    context_t = context.transpose(0, 1).data
    masked_context = context_t * re_padMask.unsqueeze(2).expand(re_padMask.size(0), re_padMask.size(1), context_t.size(2))
    sent_len = torch.sum(re_padMask, 1).squeeze(1)
    representation = torch.div(torch.sum(masked_context, 1).squeeze(1), sent_len.unsqueeze(1).expand(sent_len.size(0), context.size(2)))
    if tgtBatch is not None:
        if rb_init_tgt:
            new_tgt_batch = tgtBatch[:, 1:]
            tgt_rb_token = tgt_rb.unsqueeze(1) + model.decoder.dict_size
            tgtBatch = torch.cat([tgt_rb_token, new_tgt_batch], 1)
        decStates = encStates
        decOut = model.make_init_decoder_output(context)
        model.decoder.apply(applyContextMask)
        initOutput = model.make_init_decoder_output(context)
        decOut, decStates, attn = model.decoder(
            tgtBatch[:, :-1], tgt_rb, decStates, context, initOutput)
        for dec_t, tgt_t in zip(decOut.transpose(0, 1), tgtBatch.transpose(0, 1)[1:].data):
            gen_t = model.generator.forward(dec_t)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.data.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
            goldScores += scores

    #  (3) run the decoder to generate sentences, using beam search

    # Expand tensors for each beam.
    context = Variable(context.data.repeat(1, beamSize, 1))
    decStates = (Variable(encStates[0].data.repeat(1, beamSize, 1)),
                 Variable(encStates[1].data.repeat(1, beamSize, 1)))
    if rb_init_tgt:
        beam = [onmt.Beam(beamSize, cuda, tgt_rb[k].data[0]) for k in range(batchSize)]
    else:
        beam = [onmt.Beam(beamSize, cuda) for k in range(batchSize)]

    decOut = model.make_init_decoder_output(context)

    padMask = srcBatch.data.eq(onmt.Constants.PAD).unsqueeze(0).repeat(beamSize, 1, 1)
    rb_token_mask = torch.zeros(padMask.size(0), padMask.size(1), 1).byte()
    if cuda:
        rb_token_mask = rb_token_mask.cuda()
    if rb_init_token:
        padMask = torch.cat([rb_token_mask, padMask], 2)
    batchIdx = list(range(batchSize))
    remainingSents = batchSize
    for i in range(max_sent_length):
        model.decoder.apply(applyContextMask)

        # Prepare decoder input.
        input = torch.stack([b.getCurrentState() for b in beam
                             if not b.done]).t().contiguous().view(1, -1)
        new_tgt_rb = torch.stack([tgt_rb[i].expand(beamSize) for i, b in enumerate(beam) if not b.done]).contiguous().view(-1)
        '''some_done = False
        data = []
        for i, b in enumerate(beam):
            if b.done:
                some_done = True
            else:
                data.append(tgt_rb.data[i])
        if some_done:
            print data
            print new_tgt_rb'''
        decOut, decStates, attn = model.decoder(
            Variable(input).transpose(0, 1), new_tgt_rb, decStates, context, decOut)
        # decOut: 1 x (beam*batch) x numWords
        decOut = decOut.transpose(0, 1).squeeze(0)
        out = model.generator.forward(decOut)

        # batch x beam x numWords
        wordLk = out.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
        attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

        active = []
        for b in range(batchSize):
            if beam[b].done:
                continue

            idx = batchIdx[b]
            if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                active += [b]

            for decState in decStates:  # iterate over h, c
                # layers x beam*sent x dim
                sentStates = decState.view(
                    -1, beamSize, remainingSents, decState.size(2))[:, :, idx]
                sentStates.data.copy_(
                    sentStates.data.index_select(1, beam[b].getCurrentOrigin()))

        if not active:
            break

        # in this section, the sentences that are still active are
        # compacted so that the decoder is not run on completed sentences
        tt = torch.cuda if cuda else torch
        activeIdx = tt.LongTensor([batchIdx[k] for k in active])
        batchIdx = {beam: idx for idx, beam in enumerate(active)}
        def updateActive(t):
            # select only the remaining active sentences
            view = t.data.view(-1, remainingSents, rnnSize)
            newSize = list(t.size())
            newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
            return Variable(view.index_select(1, activeIdx) \
                            .view(*newSize))

        decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
        decOut = updateActive(decOut)
        context = updateActive(context)
        padMask = padMask.index_select(1, activeIdx)

        remainingSents = len(active)
    #  (4) package everything up

    allHyp, allScores, allAttn = [], [], []

    for b in range(batchSize):
        scores, ks = beam[b].sortBest()

        allScores += [scores[:n_best]]
        valid_attn = srcBatch.transpose(0, 1).data[:, b].ne(onmt.Constants.PAD).nonzero().squeeze(1)
        hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
        attn = [a.index_select(1, valid_attn) for a in attn]
        allHyp += [hyps]
        allAttn += [attn]
    padMask = None
    model.decoder.apply(applyContextMask)
    return allHyp, allScores, allAttn, goldScores, representation

def idxToToken(pred, src, attn, tgt_dict, replace_unk, rb_init_token):
    tokens = tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
    tokens = tokens[:-1]  # EOS
    if replace_unk:
        for i in range(len(tokens)):
            if tokens[i] == onmt.Constants.UNK_WORD:
                _, maxIndex = attn[i].max(0)
                maxIndex = maxIndex[0]
                if rb_init_token:
                    maxIndex = max(0, maxIndex - 1)
                # FIXME phrase table
                tokens[i] = src[maxIndex]
    return tokens

class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        #print opt.cuda
        checkpoint = torch.load(opt.model)
        self.models = checkpoint['models']
        self.checkpoint_opt = checkpoint['opt']
        #if "rb_init_tgt" not in vars(self.checkpoint_opt):
        #    self.checkpoint_opt.rb_init_tgt = False
        if opt.task == "Multi-MT":
            line = opt.language_pair.split("-")
            self.src_language_rb = checkpoint['src_language_mapping'][line[0]]
            self.tgt_language_rb = checkpoint['tgt_language_mapping'][line[1]]
            if self.checkpoint_opt.separate_encoder:
                self.model = self.models[self.src_language_rb]
            else:
                self.model = self.models[0]
        else:
            self.model = self.models[0]
        self.model.eval()
        self.tgt_rb_offset = opt.tgt_rb_offset
        if self.tgt_rb_offset is None and opt.task != "Multi-MT":
            self.tgt_rb_offset = checkpoint["tgt_rb_offset"]
        if opt.cuda:
            self.model.cuda()
        else:
            self.model.cpu()

        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']
        self.threshold = checkpoint['threshold']
        self.unk_offset = checkpoint["unk_offset"]

    def buildTargetTokens(self, pred, src, attn):
        return idxToToken(pred, src, attn, self.tgt_dict, self.opt.replace_unk, self.checkpoint_opt.rb_init_token)

    def buildData(self, srcBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD) for b in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD) for b in goldBatch]
            gold_tgt_rb = [tools.readability.FleschKincaid(" ".join(self.tgt_dict.convertToLabels(src_sent, onmt.Constants.EOS)), self.unk_offset) for src_sent in tgtData]
        if self.checkpoint_opt.task == "Multi-MT":
            src_rb = [self.src_language_rb for src_sent in srcData]
        elif self.checkpoint_opt.task == "MT":#previous serious bugg!!!!
            src_rb = [0 for src_sent in srcData]
        else:
            src_rb = [tools.readability.FleschKincaid(" ".join(self.src_dict.convertToLabels(src_sent, onmt.Constants.EOS)), self.unk_offset) for src_sent in srcData]
        if self.checkpoint_opt.task == "Multi-MT":
            tgt_rb = [self.tgt_language_rb for src_sent in srcData]
        elif self.checkpoint_opt.task == "MT":
            tgt_rb = [1 for rb in src_rb]
        else:
            if self.opt.tgt_rb_all is None:
                tgt_rb = [rb - self.tgt_rb_offset for rb in src_rb]
            else:
                threshold = self.threshold['tgt']
                if self.opt.tgt_rb_all == 0:
                    tgt_rb = [threshold[0] - 1 for rb in src_rb]
                elif self.opt.tgt_rb_all == self.checkpoint_opt.num_rb_bin - 1:
                    tgt_rb = [threshold[self.checkpoint_opt.num_rb_bin - 2] + 1 for rb in src_rb]
                else:
                    tgt_rb = [(threshold[self.opt.tgt_rb_all] + threshold[self.opt.tgt_rb_all - 1]) / 2. for rb in src_rb]
            #return onmt.Dataset(srcData, tgtData, self.opt.batch_size, self.opt.cuda)
        return onmt.BucketIterator(srcData, tgtData, src_rb, tgt_rb, self.checkpoint_opt, self.threshold)

    def translateBatch(self, batch):
        return translate_batch_external(batch, self.opt.beam_size, self.model, self.opt.cuda, self.checkpoint_opt.rb_init_token, self.checkpoint_opt.rb_init_tgt, self.opt.max_sent_length, self.opt.n_best)

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        batch = dataset[0]
        #batch = [x.transpose(0, 1) for x in batch]
        batch = [batch[0].transpose(0, 1), batch[1].transpose(0, 1), batch[2], batch[3]]
        #  (2) translate
        pred, predScore, attn, goldScore, representation = self.translateBatch(batch)

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batch[0].size(0)):
            if self.checkpoint_opt.reverse_src:
                src_sent = srcBatch[b][::-1]
            else:
                src_sent = srcBatch[b]
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], src_sent, attn[b][n])
                        for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, goldScore, representation, batch[2]
