import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import init
from torch.nn import functional as F
from torch.utils import data
from core import BaseModel, SelfAttention

class SyllableCounter(BaseModel):
    ''' Model that finds beginnings of syllables in a single English word.

    INPUT
        Tensor of shape (seq_len, batch_size), where seq_len is the amount
        of characters in the given word

    OUTPUT
        Tensor of shape (seq_len, batch_size) with values in [0, 1], that
        indicate whether a new syllable begins at that character
    '''
    def __init__(self, **params):
        super().__init__(**params)

        # Model parameters
        dim = params['dim']
        rnn_drop = params.get('rnn_drop', 0.0)
        lin_drop = params.get('lin_drop', 0.0)
        num_layers = params.get('num_layers', 1)
        num_linear = params.get('num_linear', 0)

        # Layers
        self.embed = nn.Embedding(self.vocab_size, dim // 4)
        self.rnn = nn.GRU(dim // 4, dim // 2, bidirectional = True,
            num_layers = num_layers, dropout = rnn_drop)
        self.drops = nn.ModuleList([nn.Dropout(lin_drop) 
            for _ in range(num_linear)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(dim)
            for _ in range(num_linear)])
        self.lins = nn.ModuleList([nn.Linear(dim, dim) 
            for _ in range(num_linear)])
        self.out = nn.Linear(dim, 1)

        self.initialise()

    def initialise(self):
        lin_params = [lin.weight for lin in self.lins]
        for param in lin_params:
            init.kaiming_normal_(param)
        rnn_params = self.rnn.parameters()
        for param in rnn_params:
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        return self

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        for idx in range(len(self.lins)):
            x = self.drops[idx](x)
            x = self.norms[idx](x.permute(1, 2, 0)).permute(2, 0, 1)
            prior = x
            x = F.relu(self.lins[idx](x))
            x = torch.sum(torch.stack([prior, x], dim = 0), dim = 0)
        x = torch.sigmoid(self.out(x))
        return x.squeeze()

    def predict(self, doc: str, pred_threshold = 0.5, return_syls = True,
        return_confidence = False, return_sequence = False):
        import re

        doc = str(doc)
        if re.sub(r'[^a-zA-Z]', '', doc) == '':
            return 0

        with torch.no_grad():
            self.eval()

            # Unpack abbreviations
            for abbrev, phrase in self.abbrevs.items():
                doc = re.sub(r'(^|(?<= )){}($|(?=[^a-zA-Z]))'.format(abbrev),
                    phrase, doc)

            # Split input into words and initialise variables
            words = re.sub(r'[^a-z ]', '', doc.lower().strip()).split(' ')
            char_idx, num_syls, confidence = 0, 0, 1
            if return_sequence:
                prob_seq = torch.zeros((sum(len(word) for word in words)))

            # Loop over words and accumulate the specified outputs
            for word in words:
                if re.sub(r'[^a-zA-Z]', '', word) == '':
                    probs = torch.tensor([0])
                else:
                    probs = self.forward(self.word2bits(word))
                syl_seq = probs > pred_threshold
                if len(syl_seq.shape) == 0:
                    syl_seq = syl_seq.unsqueeze(0)
                if return_syls:
                    num_syls += torch.sum(probs).item()
                    num_syls += 0.1 * sum(syl_seq) - 0.1 * sum(~syl_seq)
                    num_syls = int(np.around(num_syls))
                if return_confidence:
                    confidence *= torch.prod(probs[syl_seq])
                    confidence *= torch.prod(1 - probs[~syl_seq])
                if return_sequence:
                    prob_seq[range(char_idx, char_idx + len(word))] = probs.T
                    char_idx += len(word)

            # Return the specified outputs
            out = {}
            if return_syls:
                out['num_syls'] = num_syls
            if return_confidence:
                out['confidence'] = np.around(confidence.item(), 2)
            if return_sequence:
                chars = list(''.join(words))
                prob_seq = np.around(prob_seq.numpy(), 2)
                out['probs'] = list(zip(chars, prob_seq))
            if len(out) == 1:
                return list(out.values())[0]
            else:
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

def load_model(name = 'counter'):
    ''' Load a previously saved model. '''
    from pathlib import Path

    # Get the checkpoint path
    paths = list(Path('.').glob('{}*.pt'.format(name)))
    if len(paths) > 1:
        print('Multiple models found:')
        for idx, path in enumerate(paths):
            print(idx, path)
        idx = int(input('Type the index of the model you want to load.\n>>> '))
    elif len(paths) == 1:
        idx = 0
    else:
        idx = None

    # Load the model
    checkpoint = torch.load(paths[idx])
    counter = SyllableCounter(**checkpoint['params'])
    counter.load_state_dict(checkpoint['model_state_dict'])
    counter.history = checkpoint['history']

    return counter

def load(name = 'counter', **params):
    ''' Load model, optimizer, scheduler and loss function. '''
    from functools import partial
    from pathlib import Path

    lr = params.get('learning_rate', 3e-4)
    step_size = params.get('decay', (10, 1.0))[0]
    gamma = params.get('decay', (10, 1.0))[1]
    pos_weight = params.get('pos_weight', 1)
    smoothing = params.get('smoothing', 0.0)

    # Build loss function
    criterion = partial(bce_rmse, pos_weight = pos_weight, 
        smoothing = smoothing)

    # Get the checkpoint path
    paths = list(Path('.').glob('{}*.pt'.format(name)))
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
            print('\tType: {}'.format(type(e)))
            print('\tDescription: {}'.format(type(e).__doc__))
            print('\tParameters: {}'.format(e))
            print('Ignoring it and training a fresh model')

        counter = SyllableCounter(**params)
        optimizer = optim.Adam(counter.parameters(), lr = lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
            step_size = step_size, gamma = gamma)

    return counter, optimizer, scheduler, criterion

def bce_rmse(pred, target, pos_weight = 1, smoothing = 0.0, epsilon = 1e-12):
    ''' A combination of binary crossentropy and root mean squared error.

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
            A small constant to avoid dividing by zero

    OUTPUT
        The average of the character-wise binary crossentropy and the
        word-wise root mean squared error
    '''
    target = target * (1 - smoothing) + (1 - target) * smoothing
    loss_pos = target * torch.log(pred + epsilon)
    loss_neg = (1 - target) * torch.log(1 - pred + epsilon)

    bce = torch.mean(torch.neg(pos_weight * loss_pos + loss_neg))
    mse = (torch.sum(pred, dim = 0) - torch.sum(target, dim = 0)) ** 2
    rmse = torch.mean(torch.sqrt(mse + epsilon))
    return (bce + rmse) / 2

    # Comparison of different loss functions after training for one epoch:
    #
    # bce = 82.42% accuracy, took 06:52 minutes
    # bce + mse = 81.47% accuracy, took 06:51 minutes
    # bce + rmse = 85.90% accuracy, took 06:51 minutes  <--- best
    # seq_bce = 82.66% accuracy, took 06:57 minutes
    # seq_bce + mse = 83.29%, took 06:50 minutes
    # seq_bce + rmse = 83.22%, took 06:57 minutes


if __name__ == '__main__':

    # Hyperparameters
    hparams = {
        'dim': 256,
        'num_layers': 3,
        'num_linear': 2,
        'rnn_drop': 0.1,
        'lin_drop': 0.5,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'decay': (5, 0.5), 
        'pos_weight': 1.2,
        'smoothing': 0.1,
        'verbose': 0,
        'monitor': 'val_acc',
        'patience': np.inf,
        'ema': 0.999, # With batch size 32 this averages over 32,000 samples
        'ema_bias': 200
        }

    # Get data
    train_dl, val_dl, dparams = get_data(file_name = 'gutsyls', 
        batch_size = hparams['batch_size'])

    # Load model, optimizer, scheduler and loss function
    counter, optimizer, scheduler, criterion = load(**hparams, 
        **dparams)

    # Train the model
    counter.fit(train_dl, val_dl, criterion = criterion,
        optimizer = optimizer, scheduler = scheduler, 
        monitor = hparams['monitor'], patience = hparams['patience'],
        ema = hparams['ema'], ema_bias = hparams['ema_bias'])

    # Print report and plots
    counter.report(val_dl)
    counter.report(train_dl)
    counter.plot(metrics = {'acc', 'val_acc'})
    counter.plot(metrics = {'val_f1', 'val_prec', 'val_rec'})
    counter.plot(metrics = {'loss', 'val_loss'})
