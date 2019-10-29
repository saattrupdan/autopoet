import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torchnlp.nn import LockedDropout
from core import Module, TBatchNorm

class SyllableCounter(Module):
    def __init__(self, emb_dim, rnn_dim, num_layers, rnn_drop, lin_dim,
        lin_drop, params, verbose = 0):
        super().__init__(verbose)

        self.embed = nn.Embedding(params['vocab_size'], emb_dim)

        self.rnn_drop = LockedDropout(rnn_drop)
        self.rnn = nn.GRU(emb_dim, rnn_dim, bidirectional = True,
            num_layers = num_layers, dropout = rnn_drop)
        self.rnn_bn = TBatchNorm(2 * rnn_dim, verbose = verbose)

        self.lin_drop = LockedDropout(lin_drop)
        self.lin = nn.Linear(2 * rnn_dim, lin_dim)
        self.lin_bn = TBatchNorm(lin_dim, verbose = verbose)
        self.out = nn.Linear(lin_dim, 1)

    def forward(self, x):
        x = self.embed(x)

        x = self.rnn_drop(x)
        x, _ = self.rnn(x)
        x = self.rnn_bn(x)

        x = self.lin_drop(x)
        x = F.relu(self.lin(x))
        x = self.lin_bn(x)

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

def get_data(file_name, batch_size, split_ratio = 0.99):
    ''' Build the datasets to feed into our model.

    INPUT
        file_name
            The name of the tsv file containing the data, located in the
            'data' folder
        batch_size
            The amount of samples (=paragraphs) we load in from the dataset
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
    # fashion, so that both datasets will have the same syllable distribution
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
    params = {'vocab_size': len(TXT.vocab)}

    return train_dl, val_dl, params

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
    if pred.shape != target.shape:
        print(pred, target)
    loss_pos = target * torch.log(pred + epsilon)
    loss_pos *= target - smoothing
    loss_pos *= pos_weight

    loss_neg = (1 - target) * torch.log(1 - pred + epsilon)
    loss_neg *= 1 - target + smoothing

    loss = loss_pos + loss_neg
    return torch.neg(torch.mean(loss))

if __name__ == '__main__':
    from pathlib import Path
    from functools import partial

    # Hyperparameters
    LEARNING_RATE = 3e-4
    DECAY = (10, 0.8)
    POS_WEIGHT = 1.4
    SMOOTHING = 0.1
    BATCH_SIZE = 32
    EMB_DIM = 50
    RNN_DIM = 128
    LIN_DIM = 256
    NUM_LAYERS = 1
    RNN_DROP = 0.0
    LIN_DROP = 0.0

    # Get data
    train_dl, val_dl, params = get_data('gutsyls', batch_size = BATCH_SIZE)

    # Build model, optimizer and scheduler
    counter = SyllableCounter(emb_dim = EMB_DIM, rnn_dim = RNN_DIM,
        num_layers = NUM_LAYERS, rnn_drop = RNN_DROP, lin_dim = LIN_DIM,
        lin_drop = LIN_DROP, params = params, verbose = 0)
    optimizer = optim.Adam(counter.parameters(), lr = LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
        step_size = DECAY[0], gamma = DECAY[1])
    history = None

    # Build loss function
    criterion = partial(custom_bce, pos_weight = POS_WEIGHT, 
        smoothing = SMOOTHING)

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

    # Load the state
    if idx is not None:
        try:
            checkpoint = torch.load(paths[idx])
            counter.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            counter.history = checkpoint['history']
        except RuntimeError as e:
            print(e)

    print(f'Training on {len(train_dl) * train_dl.batch_size:,d} samples and '\
          f'validating on {len(val_dl) * val_dl.batch_size:,d} samples')
    print(f'Number of trainable parameters: {counter.trainable_params():,d}')

    # Train the model
    H = counter.fit(train_dl, val_dl, criterion = criterion,
        optimizer = optimizer, scheduler = scheduler, 
        monitor = 'loss', patience = 9)

    # Print report and plots
    counter.report(val_dl)
    counter.report(train_dl)
    counter.plot(metrics = {'acc, val_acc'})
    counter.plot(metrics = {'loss', 'val_loss'})
