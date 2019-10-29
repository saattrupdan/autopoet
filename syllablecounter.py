import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torchnlp.nn import LockedDropout
from core import BaseModel, TBatchNorm, TConv

class SyllableCounter(BaseModel):
    def __init__(self, **params):
        super().__init__(**params)

        # Model parameters
        emb_dim = params['emb_dim']
        rnn_dim = params['rnn_dim']
        lin1_dim = params['lin1_dim']
        lin2_dim = params['lin2_dim']
        num_layers = params.get('num_layers', 1)
        emb_drop = params.get('emb_drop', 0.0)
        rnn_drop = params.get('rnn_drop', 0.0)
        lin_drop = params.get('lin_drop', 0.0)

        # Layers
        self.embed = nn.Embedding(self.vocab_size, emb_dim)
        self.embed_drop = LockedDropout(emb_drop)

        self.rnn = nn.GRU(emb_dim, rnn_dim, bidirectional = True,
            num_layers = num_layers, dropout = rnn_drop * (num_layers > 1))
        self.rnn_bn = TBatchNorm(2 * rnn_dim)
        self.rnn_drop = LockedDropout(rnn_drop)

        self.lin1 = nn.Linear(2 * rnn_dim, lin1_dim)
        self.lin1_bn = TBatchNorm(lin1_dim)
        self.lin_drop = LockedDropout(lin_drop)

        self.lin2 = nn.Linear(lin1_dim, lin2_dim)
        self.lin2_bn = TBatchNorm(lin2_dim)

        self.out = nn.Linear(lin2_dim, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)

        x, _ = self.rnn(x)
        x = self.rnn_bn(x)
        x = self.rnn_drop(x)

        x = F.relu(self.lin1(x))
        x = self.lin1_bn(x)
        x = self.lin_drop(x)

        x = F.relu(self.lin2(x))
        x = self.lin2_bn(x)

        out = torch.sigmoid(self.out(x)).squeeze()
        return out

class BatchWrapper:
    ''' A wrapper around a dataloader to pull out data in a custom format. 
    
    INPUT
        dl
            The dataloader we are wrapping
    '''
    def __init__(self, dl):
        self.dl = dl
        self.batch_size = dl.batch_size

    def __iter__(self):
        for batch in self.dl:
            yield (batch.word, batch.syl_seq)

    def __len__(self):
        return len(self.dl)

def get_data(file_name, batch_size = 32, split_ratio = 0.99):
    ''' Build the datasets to feed into our model.

    INPUT
        file_name
            The name of the tsv file containing the data, located in the
            'data' folder
        batch_size = 32
            The amount of samples (= paragraphs) we load in from the dataset
            at a time
        split_ratio = 0.99
            The proportion of the dataset that we will train on. We will
            use the remaining part for validation
    '''
    from torchtext.data import Field, TabularDataset, BucketIterator
    import random

    # Define fields in dataset
    TXT = Field(tokenize = lambda x: list(x), lower = True)
    SEQ = Field(tokenize = lambda x: list(map(int, x)), pad_token = 0, 
        unk_token = 0, is_target = True, dtype = torch.float)
    NUM = Field(sequential = False, use_vocab = False,
        preprocessing = lambda x: int(x), dtype = torch.float)

    datafields = [('word', TXT), ('syl_seq', SEQ), ('syls', NUM)]
   
    # Load in dataset, applying the preprocessing and tokenisation as
    # described in the fields
    dataset = TabularDataset(
        path = 'data/{}.tsv'.format(file_name),
        format = 'tsv', 
        skip_header = True,
        fields = datafields
        )

    # Split dataset into a training set and a validation set in a stratified
    # fashion, so that both datasets will have the same syllable distribution.
    # Also set a random seed, to ensure that we retain the same train/val
    # split when we are loading a previously saved model
    random.seed(42)
    train, val = dataset.split(split_ratio = split_ratio, stratified = True,
        strata_field = 'syls', random_state = random.getstate())

    # Build vocabulary
    TXT.build_vocab(train)
    SEQ.build_vocab(train)

    # Split the two datasets into batches. This converts the tokenised words 
    # into integer sequences, and also pads every batch so that, within a 
    # batch, all sequences are of the same length
    train_iter, val_iter = BucketIterator.splits(
        datasets = (train, val),
        batch_sizes = (batch_size, batch_size),
        sort_key = lambda x: len(x.word),
        sort_within_batch = False,
        )

    # Apply custom batch wrapper, which outputs the data in the form that
    # the network wants it
    train_dl = BatchWrapper(train_iter)
    val_dl = BatchWrapper(val_iter)

    # Save a few parameters that are used elsewhere
    dparams = {'vocab_size': len(TXT.vocab), 'char2idx': dict(TXT.vocab.stoi)}

    return train_dl, val_dl, dparams

def load_model():
    ''' Helper function to load a previously saved model. '''
    counter, _, _, _ = load()
    return counter

def load(**params):
    ''' Load model, optimizer, scheduler and loss function. '''
    from functools import partial
    from pathlib import Path

    lr = params.get('learning_rate', 3e-4)
    step_size = params.get('decay', (10, 1.0))[0]
    gamma = params.get('decay', (10, 1.0))[1]
    pos_weight = params.get('pos_weight', 1)
    smoothing = params.get('smoothing', 0.0)

    # Build loss function
    criterion = partial(custom_bce, pos_weight = pos_weight, 
        smoothing = smoothing)

    # Get the checkpoint path
    paths = list(Path('.').glob('counter*.pt'))
    if len(paths) > 1:
        print('Multiple models found:')
        for idx, path in enumerate(paths):
            print(idx, path)
        idx = int(input('Type the index of the model you want to load.\n>>> '))
    elif len(paths) == 1:
        idx = 0
    else:
        idx = None

    # Load the model, optimizer and scheduler
    try:
        checkpoint = torch.load(paths[idx])

        counter = SyllableCounter(**checkpoint['params'])
        optimizer = optim.Adam(counter.parameters(), lr = lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
            step_size = step_size, gamma = gamma)

        counter.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        counter.history = checkpoint['history']

    except Exception as e:
        if idx is not None:
            print('Exception happened in trying to load checkpoint:')
            print(f'\tType: {type(e)}')
            print(f'\tDescription: {type(e).__doc__}')
            print(f'\tParameters: {e}')
            print('Ignoring it and training a fresh model')

        counter = SyllableCounter(**params)
        optimizer = optim.Adam(counter.parameters(), lr = lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
            step_size = step_size, gamma = gamma)

    return counter, optimizer, scheduler, criterion

def custom_bce(pred, target, pos_weight = 1, smoothing = 0.0, epsilon = 1e-12):
    ''' 
    Custom version of the binary crossentropy. 

    INPUT
        pred
            A 1-dimensional tensor containing predicted values
        target
            A 1-dimensional tensor containing true values
        pos_weight = 1
            The weight that should be given to positive examples
        smoothing = 0.0
            Smoothing parameter for the presence detection
        epsilon = 1e-12
            A small constant to avoid taking logarithm of zero

    OUTPUT
        The binary crossentropy
    '''
    loss_pos = target * torch.log(pred + epsilon)
    loss_pos *= target - smoothing
    loss_pos *= pos_weight

    loss_neg = (1 - target) * torch.log(1 - pred + epsilon)
    loss_neg *= 1 - target + smoothing

    loss = loss_pos + loss_neg
    return torch.neg(torch.sum(loss)) / target.shape[1]


if __name__ == '__main__':

    # Hyperparameters
    hparams = {
        'emb_dim': 50,
        'rnn_dim': 256,
        'lin1_dim': 512,
        'lin2_dim': 256,
        'num_layers': 1,
        'emb_drop': 0.2,
        'rnn_drop': 0.2,
        'lin_drop': 0.2,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'decay': (10, 0.8), 
        'pos_weight': 1.3,
        'smoothing': 0.15,
        'verbose': 0,
        'monitor': 'val_acc',
        'patience': 9
        }

    # Get data
    train_dl, val_dl, dparams = get_data(file_name = 'gutsyls', 
        batch_size = hparams['batch_size'])

    # Load model, optimizer, scheduler and loss function
    counter, optimizer, scheduler, criterion = load(**hparams, **dparams)

    # Train the model
    H = counter.fit(train_dl, val_dl, criterion = criterion,
        optimizer = optimizer, scheduler = scheduler, 
        monitor = hparams['monitor'], patience = hparams['patience'])

    # Print report and plots
    counter.report(val_dl)
    counter.report(train_dl)
    counter.plot(metrics = {'acc', 'val_acc'})
    counter.plot(metrics = {'val_f1', 'val_prec', 'val_rec'})
    counter.plot(metrics = {'loss', 'val_loss'})
