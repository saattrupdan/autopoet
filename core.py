import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import init
from torch.nn import functional as F

class BaseModel(nn.Module):
    ''' A PyTorch module with logging and training built-in.

    INPUT
        char2idx
            Dictionary containing the character -> index pairs
        vocab_size
            Size of the vocubulary
        verbose = 0
            Verbosity level, can be set to 0, 1 or 2
    '''
    def __init__(self, **params):
        super().__init__()
        import logging
        import pandas as pd

        self.params = params
        self.char2idx = params['char2idx']
        self.vocab_size = params['vocab_size']
        self.verbose = params.get('verbose', 0)
        self.history = None
        self.abbrevs = dict(
            pd.read_csv('data/abbrevs.tsv', sep = '\t', header = None).values
            )

        logging.basicConfig()
        logging.root.setLevel(logging.NOTSET)
        self.logger = logging.getLogger()
        if not self.verbose:
            self.logger.setLevel(logging.WARNING)
        elif self.verbose == 1:
            self.logger.setLevel(logging.INFO)
        elif self.verbose == 2:
            self.logger.setLevel(logging.DEBUG)

    def trainable_params(self):
        return sum(param.numel() for param in self.parameters() 
                if param.requires_grad)

    def word2bits(self, word: str):
        bits = torch.zeros((len(word), 1), dtype = torch.long)
        for j, char in enumerate(word):
            bits[j, 0] = self.char2idx.get(char, 0)
        return bits

    def plot(self, metrics = {'val_acc', 'val_f1'}, save_to = None, 
        show_plot = True, title = 'Model performance by epoch',
        xlabel = 'epochs', ylabel = 'score'):
        import matplotlib.pyplot as plt

        plt.style.use('ggplot')
        fig, ax = plt.subplots()

        for metric in metrics:
            ax.plot(self.history[metric], label = metric)

        ax.legend(loc = 'best')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if save_to is not None and isinstance(save_to, str):
            plt.savefig(save_to)

        if show_plot:
            plt.show()

        return self

    def save(self, file_name, optimizer, scheduler = None):
        params = {
            'history': self.history,
            'params': self.params,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }

        if scheduler is not None:
            params['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(params, file_name)
        return self

    @staticmethod
    def get_scores(TP, TN, FP, FN, beta = 1, epsilon = 1e-12):
        ''' Computes precision, recall and F-score from a confusion matrix.

        INPUT
            TP
                The number of true positives
            TN
                The number of true negatives
            FP
                The number of false positives
            FN
                The number of false negatives
            beta = 1
                The F-score parameter, by default computing the F1-score
            epsilon = 1e-12
                A small number to avoid dividing by zero

        OUTPUT
            scores
                A dictionary with keys 'precision', 'recall' and 'F_score'
        '''
        scores = {}
        scores['precision'] = torch.mean(TP / (TP + FP + epsilon))
        scores['recall'] = torch.mean(TP / (TP + FN + epsilon))
        scores['F_score'] = 1 + beta ** 2 
        scores['F_score'] /= 1 / scores['precision'] + 1 / scores['recall']
        return scores

    def fit(self, train_loader, val_loader, criterion, optimizer,
        scheduler = None, epochs = np.inf, monitor = 'val_loss', 
        target_value = None, patience = 10, ema = 0.99, ema_bias = 25,
        pred_threshold = 0.5, announce = True, save_model = True):
        from tqdm import tqdm
        from itertools import count
        from pathlib import Path
        from functools import cmp_to_key
        import os

        if self.history is None:
            self.history = {
                'loss': [], 
                'val_loss': [],
                'acc': [],
                'val_acc': [],
                'val_prec': [],
                'val_rec': [],
                'val_f1': []
                }

        bad_epochs = 0
        avg_loss = 0
        avg_acc = 0
        val_loss = 0
        start_epoch = len(self.history['loss'])
        acc_batch = start_epoch * len(train_loader)

        # Working with all scores rather than the current score since
        # lists are mutable and floats are not, allowing us to update
        # the score on the fly
        scores = self.history[monitor]

        if monitor == 'loss' or monitor == 'val_loss':
            score_cmp = lambda x, y: x < y
            best_score = np.inf
        else:
            score_cmp = lambda x, y: x > y
            best_score = 0

        if start_epoch:
            avg_loss = self.history['loss'][-1]
            val_loss = self.history['val_loss'][-1]
            avg_acc = self.history['acc'][-1]
            best_score = max(scores, key = cmp_to_key(score_cmp))

        if announce:
            num_train = len(train_loader) * train_loader.batch_size
            num_val = len(val_loader) * val_loader.batch_size
            num_params = self.trainable_params()
            print('Training on {:,d} samples and '\
                  'validating on {:,d} samples'\
                  .format(num_train, num_val))
            print('Number of trainable parameters: {:,d}'\
                .format(num_params))

        # Training loop
        for epoch in count(start = start_epoch):

            # Enable training mode
            self.train()

            # Stop if we have reached the total number of epochs
            if epoch >= start_epoch + epochs:
                break

            # Epoch loop
            num_samples = len(train_loader) * train_loader.batch_size
            with tqdm(total = num_samples) as epoch_pbar:
                epoch_pbar.set_description('Epoch {:2d}'.format(epoch))
                acc_loss = 0
                for xtrain, ytrain in train_loader:

                    # Reset the gradients
                    optimizer.zero_grad()

                    # Do a forward pass, calculate loss and backpropagate
                    yhat = self.forward(xtrain)
                    if len(yhat.shape) == 0:
                        yhat = yhat.unsqueeze(0)
                    loss = criterion(yhat, ytrain)
                    loss.backward()
                    optimizer.step()

                    # Exponentially moving average of loss
                    # Note: The float() is there to copy the loss by value
                    #       and not by reference, to allow it to be garbage
                    #       collected and avoid an excessive memory leak
                    avg_loss = ema * avg_loss + (1 - ema) * float(loss)

                    # Exponentially moving average of accuracy
                    syl_hat = torch.round(torch.sum(yhat, dim = 0).float())
                    syl_train = torch.sum(ytrain, dim = 0).float()
                    batch_acc = torch.sum(syl_train == syl_hat).float()
                    batch_acc /= ytrain.shape[1]
                    avg_acc = ema * avg_acc + (1 - ema) * batch_acc

                    # Bias correction
                    acc_batch += 1
                    avg_loss /= 1 - ema ** (acc_batch * ema_bias)
                    avg_acc /= 1 - ema ** (acc_batch * ema_bias)

                    # Update progress bar description
                    desc = 'Epoch {:2d} - loss {:.4f} - acc {:.4f}'\
                           .format(epoch, avg_loss, avg_acc)
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(train_loader.batch_size)

                # Compute validation metrics
                with torch.no_grad():

                    # Enable validation mode
                    self.eval()

                    val_loss = 0.
                    val_acc = 0.
                    TP, TN, FP, FN = 0, 0, 0, 0
                    for xval, yval in val_loader:
                        probs = self.forward(xval)
                        if len(probs.shape) == 0:
                            probs= probs.unsqueeze(0)

                        val_loss += criterion(probs, yval)

                        yval = yval > pred_threshold
                        yhat = probs > pred_threshold

                        TP += torch.sum(yhat & yval).float()
                        TN += torch.sum(~yhat & ~yval).float()
                        FP += torch.sum(yhat & ~yval).float()
                        FN += torch.sum(~yhat & yval).float()

                        syl_hat = torch.sum(probs, dim = 0).float()
                        syl_hat = torch.round(syl_hat)
                        syl_val = torch.sum(yval, dim = 0).float()
                        batch_acc = torch.sum(syl_val == syl_hat).float()
                        batch_acc /= yval.shape[1]
                        val_acc += batch_acc

                    # Calculate average validation loss
                    val_loss /= len(val_loader)

                    # Calculate syllable accuracy
                    val_acc /= len(val_loader)

                    # Calculate more scores
                    epoch_scores = self.get_scores(TP,TN,FP,FN)
                    val_prec = epoch_scores['precision']
                    val_rec = epoch_scores['recall']
                    val_f1 = epoch_scores['F_score']

                    # Add to history
                    self.history['loss'].append(avg_loss)
                    self.history['acc'].append(avg_acc)
                    self.history['val_loss'].append(val_loss.item())
                    self.history['val_prec'].append(val_prec)
                    self.history['val_rec'].append(val_rec)
                    self.history['val_f1'].append(val_f1)
                    self.history['val_acc'].append(val_acc)

                    # Update progress bar description
                    desc = 'Epoch {:2d} - '\
                           'loss {:.4f} - '\
                           'acc {:.4f} - '\
                           'val_loss {:.4f} - '\
                           'val_acc {:.4f} - '\
                           'val_f1 {:.4f}'\
                           .format(epoch, avg_loss, avg_acc, 
                                   val_loss, val_acc, val_f1)
                    epoch_pbar.set_description(desc)

            # Add score to learning scheduler
            if scheduler is not None:
                scheduler.step(scores[-1])

            # Save model if score is best so far
            if score_cmp(scores[-1], best_score):
                best_score = scores[-1]

                # Delete older models and save the current one
                if save_model:
                    for p in Path('.').glob('counter*.pt'):
                        p.unlink()
                    self.save(f'counter_{scores[-1]:.4f}_{monitor}.pt',
                        optimizer = optimizer, scheduler = scheduler)

            # Stop if score has not improved for <patience> many epochs
            if score_cmp(best_score, scores[-1]):
                bad_epochs += 1
                if bad_epochs > patience:
                    print('Model is not improving, stopping training.')

                    # Load the best score
                    if save_model:
                        path = next(Path('.').glob('counter*.pt'))
                        checkpoint = torch.load(path)
                        self.history = checkpoint['history']
                        self.load_state_dict(checkpoint['model_state_dict'])

                    break
            else:
                bad_epochs = 0

            # Stop when we perfom better than <target_value>
            if target_value is not None:
                if score_cmp(scores[-1], target_value):
                    print('Reached target performance, stopping training.')
                    break

        return self.history

    def report(self, dataloader, pred_threshold = 0.5):
        with torch.no_grad():

            # Enable validation mode
            self.eval()

            acc = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            for xval, yval in dataloader:
                probs = self.forward(xval)

                yval = yval > pred_threshold
                yhat = probs > pred_threshold
                if len(yhat.shape) == 0:
                    yhat = yhat.unsqueeze(0)

                TP += torch.sum(yhat & yval).float()
                TN += torch.sum(~yhat & ~yval).float()
                FP += torch.sum(yhat & ~yval).float()
                FN += torch.sum(~yhat & yval).float()
                
                syl_hat = probs / torch.max(probs, dim = 0)[0]
                syl_hat = torch.round(torch.sum(syl_hat, dim = 0))
                syl_val = torch.sum(yval, dim = 0).float()
                batch_acc = torch.sum(syl_val == syl_hat).float()
                batch_acc /= yval.shape[1]
                acc += batch_acc

            # Calculate syllable accuracy
            acc /= len(dataloader)

            # Calculate other scores
            scores = self.get_scores(TP,TN,FP,FN)
            prec = scores['precision']
            rec = scores['recall']
            f1 = scores['F_score']

            print('\n' + '~' * 23 + '  REPORT  ' + '~' * 23)
            print('Accuracy\tPrecision\tRecall\t\tF1 score')
            print('{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}'\
                .format(acc, prec, rec, f1))

            scores = {
                'accuracy': acc, 
                'precision': prec, 
                'recall': rec, 
                'f1': f1
                }

            return scores

class InjectPosition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        seq_len, batch_size, hdim = inputs.shape

        # (seq_len, 1)
        pos_enc = torch.arange(seq_len, dtype = torch.float).unsqueeze(-1)

        # (seq_len, 1) --> (seq_len, hdim)
        pos_enc = pos_enc.expand(seq_len, hdim)

        # (seq_len, hdim) --> (seq_len, hdim)
        factor = 10000 ** (2 * torch.arange(hdim, dtype = torch.float) / hdim)
        pos_enc = pos_enc / factor.unsqueeze(0)

        # (seq_len, hdim) --> (seq_len, 1, hdim)
        pos_enc = torch.sin(pos_enc).unsqueeze(1)

        # (seq_len, 1, hdim) --> (seq_len, batch_size, hdim)
        pos_enc = pos_enc.expand_as(inputs)

        # Check if the encoding is correct
        assert np.around(pos_enc[2, 0, 20], 2) == \
               np.around(np.sin(2 / 10000 ** (40 / hdim)), 2)

        # (seq_len, batch_size, hdim) + (seq_len, batch_size, hdim)
        #   --> (seq_len, batch_size, hdim)
        return pos_enc + inputs

class AttentionEncoderBlock(nn.Module):
    def __init__(self, idim, hdim, num_lin = 1):
        super().__init__()
        self.attn = SelfAttention()
        self.attn_norm = nn.BatchNorm1d(idim)
        self.lins = nn.ModuleList([nn.Linear(idim, hdim)])
        for _ in range(num_lin - 1):
            self.lins.append(nn.Linear(hdim, hdim))
        self.lin_norm = nn.BatchNorm1d(hdim)
        self.initialise()

    def initialise(self):
        for lin in self.lins:
            init.kaiming_normal_(lin.weight)
        return self

    def forward(self, inputs):
        attn = self.attn(inputs)
        attn = torch.sum(torch.stack([attn, inputs], dim = 0), dim = 0)
        attn = self.attn_norm(attn.permute(1, 2, 0)).permute(2, 0, 1)

        x = attn
        for lin in self.lins:
            x = F.relu(lin(x))
        x = torch.sum(torch.stack([x, attn], dim = 0), dim = 0)
        x = self.lin_norm(x.permute(1, 2, 0)).permute(2, 0, 1)
        return x

class AttentionEncoder(nn.Module):
    def __init__(self, idim, hdim, num_lin = 1, blocks = 1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.blocks.append(AttentionEncoderBlock(idim, hdim, num_lin))
        for _ in range(blocks - 1):
            self.blocks.append(AttentionEncoderBlock(hdim, hdim, num_lin))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SelfAttention(nn.Module):
    ''' Implementation of scaled dot-product self-attention, as described
        in "Attention is All You Need" by Vaswani et al.

        INPUT
            A tensor of shape (seq_len, batch_size, hdim)

        OUTPUT
            A tensor of the same shape, obtained as follows for every batch:
                X -> X * softmax(X * X.T / sqrt(hdim), dim = 0)
    '''
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        seq_len, batch_size, hdim = inputs.shape

        # (seq_len, batch_size, hdim) --> (batch_size, seq_len, hdim)
        inputs = inputs.permute(1, 0, 2)

        # (batch_size, seq_len, hdim) x (batch_size, hdim, seq_len)
        #   --> (batch_size, seq_len, seq_len)
        scores = torch.bmm(inputs, inputs.transpose(1, 2))
        scalar = torch.sqrt(torch.FloatTensor([hdim]))
        scores = torch.div(scores, scalar)

        # (batch_size, seq_len, seq_len) --> (batch_size, seq_len, seq_len)
        weights = F.softmax(scores, dim = 1)

        # Checking that the first row sums to 1
        assert (torch.sum(weights[0, :, 0]) * 100).round() / 100 == 1

        # (batch_size, seq_len, seq_len) x (batch_size, seq_len, hdim)
        #   --> (batch_size, seq_len, hdim)
        attn = torch.bmm(weights, inputs)

        # (batch_size, seq_len, hdim) --> (seq_len, batch_size, hdim)
        return attn.permute(1, 0, 2)

class TBatchNorm(nn.Module):
    ''' A temporal batch normalisation.

    INPUT
        Tensor of shape (seq_len, batch_size, hidden_size)

    OUTPUT
        Tensor of shape (seq_len, batch_size, hidden_size)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.bn(x)
        x = x.permute(2, 0, 1)
        return x
