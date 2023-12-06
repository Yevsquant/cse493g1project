import numpy as np
from . import optim

import torch

def create_minibatch(data, batch_size=100, split="train"):
    split_size = data["%s_captions" % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data["%s_captions" % split][mask]
    image_idxs = data["%s_image_idxs" % split][mask]
    image_features = data["%s_features" % split][image_idxs]
    urls = data["%s_urls" % split][image_idxs]
    return captions, image_features, urls

class Trainer(object):
    def __init__(self, model, data, idx_to_word, **kwargs):
        self.model = model
        self.data = data
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)
        self._reset()
        self.idx_to_word = idx_to_word

    def _reset(self):
        self.epoch = 0
        self.loss_history = []


    def _step(self):
        minibatch = create_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        mask = captions_out != self.model._null
        t_features = torch.Tensor(features)
        t_captions_in = torch.LongTensor(captions_in)
        t_captions_out = torch.LongTensor(captions_out)
        t_mask = torch.LongTensor(mask)
        logits = self.model(t_features, t_captions_in)
        loss = self.transformer_temporal_softmax_loss(logits, t_captions_out, t_mask)
        self.loss_history.append(loss.detach().numpy())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        num_train = self.data["train_captions"].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        for t in range(num_iterations):
            self._step()
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )
            epoch_end = (t + 1) % iterations_per_epoch == 0

    def transformer_temporal_softmax_loss(self, x, y, mask):
        N, T, V = x.shape
        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)
        loss = torch.nn.functional.cross_entropy(x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)
        return loss
