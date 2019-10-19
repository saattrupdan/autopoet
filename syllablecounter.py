import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

class SyllableDataset(data.Dataset):
    ''' Dataset with words and syllables. '''

    def __init__(self, tsv_fname, nrows = None):
        import string
        
        self.tsv_fname = tsv_fname
        self.nrows = nrows
        self.words = None
        self.syllables = None
        self.seq_len = None
        
        # Create a char -> idx dictionary
        alphabet = list(string.ascii_lowercase)
        self.vocab_size = len(alphabet) + 1
        self.char2idx = {char: idx for (idx, char) in 
            enumerate(alphabet, start = 1)}

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.words[idx], self.syllables[idx]

    def compile(self):
        import pandas as pd

        # Load the words
        self.words = pd.read_csv(self.tsv_fname, sep = '\t', usecols = [0],
                                nrows = self.nrows)
        
        # Convert the words to lists of characters
        def list_or_nan(x):
            try:
                return list(x)
            except TypeError:
                return np.nan
        self.words = self.words['words'].apply(list_or_nan).dropna()

        # Convert the characters to integers
        self.words = [torch.tensor([self.char2idx.get(char) 
            for char in word if self.char2idx.get(char) is not None])
            for word in self.words]
        
        # Pad the integer sequences to form a matrix
        self.words = nn.utils.rnn.pad_sequence(self.words, batch_first = True)
                
        # Save the length of the padded sequences
        self.seq_len = self.words.shape[1]

        # Load the syllables
        self.syllables = pd.read_csv(self.tsv_fname, sep = '\t', usecols = [1])
        self.syllables = torch.tensor(self.syllables.squeeze())

        return self

class ConvGrp(nn.Module):
    ''' Sequence of alternating convolutional layers and batch
        normalisations, ending with a skip connection and a max pool. '''

    def __init__(self, ch_in, ch_out, kernel_size, depth = 1):
        super().__init__()
        
        self.depth = depth
        self.convs = []
        self.bns = []
        self.skip = nn.Conv1d(ch_in, ch_out, kernel_size = 1)
 
        self.convs.append(nn.Conv1d(ch_in, ch_out, kernel_size = kernel_size,
            padding = (kernel_size - 1) // 2))
        self.bns.append(nn.BatchNorm1d(ch_out))
        
        for _ in range(depth - 1):
            self.convs.append(nn.Conv1d(ch_out, ch_out, 
                kernel_size = kernel_size, padding = (kernel_size - 1) // 2))
            self.bns.append(nn.BatchNorm1d(ch_out))

    def forward(self, x):
        inputs = x
        for i in range(self.depth):
            x = self.convs[i](x)
            x = F.relu(x)
            x = self.bns[i](x)
        x = torch.sum(torch.stack([x, self.skip(inputs)], dim = 0), dim = 0)
        return F.max_pool1d(x, 2)

class SyllableCounter(nn.Module):
    ''' Model that predicts the number of syllables in English documents. '''
    
    def __init__(self, dataset):
        super().__init__()
        self.seq_len = dataset.seq_len
        self.vocab_size = dataset.vocab_size
        self.char2idx = dataset.char2idx
        
        embedding_dim = 50
        conv_groups = 3
        filters = 128
        kernel_size = 3
        depth = 1
        fc_units = 512

        self.embed = nn.Embedding(num_embeddings = self.vocab_size, 
            embedding_dim = embedding_dim)

        # Convolutional layers
        self.convs = []
        self.convs.append(ConvGrp(embedding_dim, filters, 
            kernel_size = kernel_size, depth = depth))
        for i in range(conv_groups - 1):
            self.convs.append(ConvGrp(filters * 2 ** i, filters * 2 ** (i + 1),
                kernel_size = kernel_size, depth = depth))

        # Calculate how many flattened units the conv output has
        conv_units = self.seq_len 
        for _ in range(conv_groups):
            conv_units //= 2
        conv_units *= (filters * 2 ** (conv_groups - 1))

        self.fc = nn.Linear(conv_units, fc_units)
        self.out = nn.Linear(fc_units, 1)

    def forward(self, x):           
        x = self.embed(x)
        x = x.view(-1, x.shape[2], x.shape[1])
        for conv in self.convs:
            x = conv(x)
        x = torch.flatten(x, start_dim = 1)
        x = F.relu(self.fc(x))
        return F.elu(self.out(x)) + 1
   
    def fit(self, dataset, optimizer, val_split = 0.01, epochs = np.inf, 
        criterion = nn.MSELoss(), scheduler = None, batch_size = 32, 
        monitor = 'val_loss', target_value = 0, patience = 10,
        min_delta = 1e-4, ema = 0.99, history = None):
        from tqdm import tqdm
        from itertools import count
        from pathlib import Path
        from functools import cmp_to_key
        from sklearn.model_selection import train_test_split

        if history is not None:
            self.history = history
        else:
            self.history = {'loss': [], 'val_loss': []}

        bad_epochs = 0
        start_epoch = len(self.history['loss'])
        acc_batch = start_epoch * len(dataset) // batch_size

        # Working with all scores rather than the current score since
        # lists are mutable and floats are not, allowing us to update
        # the score on the fly
        scores = self.history[monitor]

        if monitor == 'loss' or monitor == 'val_loss':
            score_cmp = lambda x, y: x < y
        else:
            score_cmp = lambda x, y: x > y

        if start_epoch:
            avg_loss = self.history['loss'][-1]
            val_loss = self.history['val_loss'][-1]
            best_score = max(scores, key = cmp_to_key(score_cmp))
        else:
            avg_loss = 0
            val_loss = 0
            best_score = np.inf

        # Define data loaders
        train_indices, val_indices = train_test_split(range(len(dataset)),
            test_size = val_split)
        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = data.sampler.SubsetRandomSampler(val_indices)
        train_loader = data.DataLoader(dataset, batch_size = batch_size,
            sampler = train_sampler)
        val_loader = data.DataLoader(dataset, batch_size = batch_size,
            sampler = val_sampler)

        # Training loop
        for epoch in count():

            # Enable training mode
            self.train()

            # Save old score for later
            if len(scores):
                old_score = scores[-1]
            else:
                old_score = np.inf

            # Stop if we have reached the total number of epochs
            if epoch >= epochs:
                break

            # Epoch loop
            with tqdm(total = len(train_indices)) as epoch_pbar:
                epoch_pbar.set_description(f'Epoch {epoch}')
                acc_loss = 0
                for xtrain, ytrain in train_loader:

                    # Reshape and cast target tensor to float32,
                    # since this is required by the loss function
                    ytrain = ytrain.view(-1, 1).to(torch.float32)

                    # Reset the gradients
                    optimizer.zero_grad()

                    # Do a forward pass, calculate loss and backpropagate
                    yhat = self.forward(xtrain)
                    loss = torch.sqrt(criterion(yhat, ytrain))
                    loss.backward()
                    optimizer.step()

                    # Exponentially moving average with bias correction
                    # Note: The float() is there to copy the loss by value
                    #       and not by reference, to allow it to be garbage
                    #       collected and avoid an excessive memory leak
                    avg_loss = ema * avg_loss + (1 - ema) * float(loss)

                    # Bias correction
                    # Note: The factor 30 was found empirically
                    acc_batch += 1
                    avg_loss /= 1 - ema ** (acc_batch * 30)

                    # Update progress bar description
                    desc = f'Epoch {epoch} - loss {avg_loss:.4f}'
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(xtrain.shape[0])

                # Calculate average validation loss
                with torch.no_grad():

                    val_loss = 0
                    for xval, yval in val_loader:

                        # Enable validation mode
                        self.eval()

                        # Reshape and cast target tensor to float32, 
                        # since this is required by the loss function
                        yval = yval.view(-1, 1).to(torch.float32)
                    
                        yhat = self.forward(xval)
                        val_loss += torch.sqrt(criterion(yhat, yval))
                   
                    num_batches = len(val_indices) // batch_size
                    if len(val_indices) % batch_size:
                        num_batches += 1
                    val_loss /= num_batches

                # Add to history and update progress bar description
                self.history['loss'].append(avg_loss)
                self.history['val_loss'].append(val_loss.item())
                desc = f'Epoch {epoch} - loss {avg_loss:.4f} - '\
                       f'val loss {val_loss:.4f}'
                epoch_pbar.set_description(desc)

            # Stop if score has not improved for <patience> many epochs
            if score_cmp(old_score, scores[-1] - min_delta):
                bad_epochs += 1
                if bad_epochs > patience:
                    print('Model is not improving, stopping training.')
                    break
            else:
                bad_epochs = 0

            if scheduler is not None:
                scheduler.step(scores[-1])

            # Stop when we get better than <target_value>
            if score_cmp(scores[-1], target_value):
                print('Reached target performance, stopping training.')
                break

            # Save model if score is best so far
            if score_cmp(scores[-1], best_score):
                best_score = scores[-1]

                # Delete older models
                for p in Path('.').glob('counter*.pt'):
                    p.unlink()

                torch.save({
                    'history': self.history,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                    }, f'counter_{val_loss:.4f}.pt')

        return self

    def predict(self, doc):
        import re

        # Enable evaluation mode
        self.eval()

        # Stop calculating gradients
        with torch.no_grad():

            # Make lower case and remove all non-letter symbols
            words = doc.split(' ')
            words = [re.sub(r'[^a-z]', '', word.lower()) for word in words]

            # Convert words to integer sequences of the correct length
            chars = torch.zeros(len(words), self.seq_len, dtype = torch.long)
            for word_idx in range(len(words)):
                for char_idx in range(len(words[word_idx])):
                    char = words[word_idx][char_idx]
                    chars[word_idx, char_idx] = self.char2idx.get(char, 0)

            # Get predictions
            preds = self.forward(chars)

        # Return the sum of the predictions, rounded to nearest integer
        return int(torch.sum(preds))

    def plot(self, save_to = None):
        import matplotlib.pyplot as plt

        plt.style.use('ggplot')
        fig, ax = plt.subplots()

        ax.plot(self.history['loss'], color = 'orange', label = 'loss')
        ax.plot(self.history['val_loss'], color = 'blue', label = 'val_loss')
        ax.legend()

        if save_to is not None and isinstance(save_to, str):
            plt.savefig(save_to)

        plt.show()
        return self


if __name__ == '__main__':
    from pathlib import Path

    dataset = SyllableDataset('data/gutsyls.tsv').compile()
    counter = SyllableCounter(dataset)
    optimizer = optim.Adam(counter.parameters(), lr = 1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',
        factor = 0.1, patience = 3, min_lr = 1e-6)
    history = None

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
        checkpoint = torch.load(paths[idx])
        counter.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history = checkpoint['history']

    counter.fit(dataset, optimizer = optimizer, scheduler = scheduler,
        history = history, patience = 10, monitor = 'loss')

    counter.plot()
