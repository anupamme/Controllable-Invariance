import math
import torch.optim as optim

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(self.adam_momentum, 0.999))
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, seq2seq_params, disc_params, enc_params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None, adam_momentum=0.9):
        self.params = list(params)  # careful: params may be a generator
        self.seq2seq_params = list(seq2seq_params)
        self.disc_params = list(disc_params)
        self.encoder_params = list(enc_params)
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.adam_momentum = adam_momentum
        self._makeOptimizer()

    def step_all(self):
        # Compute gradients norm.
        enc_grad_norm = 0
        for param in self.encoder_params:
            enc_grad_norm += math.pow(param.grad.data.norm(), 2)
        enc_grad_norm = math.sqrt(enc_grad_norm)
        grad_norm = 0
        for param in self.params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
        shrinkage = self.max_grad_norm / grad_norm

        for param in self.params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return enc_grad_norm, grad_norm

    def step_seq2seq(self):
        # Compute gradients norm.
        enc_grad_norm = 0
        for param in self.encoder_params:
            enc_grad_norm += math.pow(param.grad.data.norm(), 2)
        enc_grad_norm = math.sqrt(enc_grad_norm)
        grad_norm = 0
        for param in self.seq2seq_params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
        shrinkage = self.max_grad_norm / grad_norm
        for param in self.seq2seq_params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return enc_grad_norm, grad_norm

    def step_disc_enc(self, disc_lambda):
        # Compute gradients norm.
        enc_grad_norm = 0
        grad_norm = 0
        for param in self.encoder_params:
            n = math.pow(param.grad.data.norm(), 2)
            if math.fabs(disc_lambda) > 1e-5:
                n *= math.pow(disc_lambda, 2)
            param.grad.data.mul_(-disc_lambda)
            enc_grad_norm += n
            grad_norm += n
        enc_grad_norm = math.sqrt(enc_grad_norm)
        for param in self.disc_params:
            grad_norm += math.pow(param.grad.data.norm(), 2)
        grad_norm = math.sqrt(grad_norm)
        if grad_norm > 0:
            shrinkage = self.max_grad_norm / grad_norm
        else:
            shrinkage = 1.

        for param in self.encoder_params + self.disc_params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return enc_grad_norm, grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.method != "sgd":
            return
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl

        self._makeOptimizer()
