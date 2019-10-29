import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import init
from torch.nn import functional as F

class Module(nn.Module):
    ''' A PyTorch module with logging and training built-in.

    INPUT
        verbose = 0
            Verbosity level, can be set to 0, 1 or 2
    '''
    def __init__(self, verbose = 0, **kwargs):
        super().__init__()
        import logging

        self.verbose = verbose
        self.history = None

        logging.basicConfig()
        logging.root.setLevel(logging.NOTSET)
        self.logger = logging.getLogger()
        if not verbose:
            self.logger.setLevel(logging.WARNING)
        elif verbose == 1:
            self.logger.setLevel(logging.INFO)
        elif verbose == 2:
            self.logger.setLevel(logging.DEBUG)

    def trainable_params(self):
        return sum(param.numel() for param in self.parameters() 
                if param.requires_grad)

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
        target_value = None, patience = 10, ema = 0.99, pred_threshold = 0.5):
        from tqdm import tqdm
        from itertools import count
        from pathlib import Path
        from functools import cmp_to_key

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
                epoch_pbar.set_description(f'Epoch {epoch:2d}')
                acc_loss = 0
                syl_acc = 0
                for xtrain, ytrain in train_loader:

                    # Reset the gradients
                    optimizer.zero_grad()

                    # Do a forward pass, calculate loss and backpropagate
                    yhat = self.forward(xtrain)
                    loss = criterion(yhat, ytrain)
                    loss.backward()
                    optimizer.step()

                    # Exponentially moving average of loss
                    # Note: The float() is there to copy the loss by value
                    #       and not by reference, to allow it to be garbage
                    #       collected and avoid an excessive memory leak
                    avg_loss = ema * avg_loss + (1 - ema) * float(loss)

                    # Exponentially moving average of accuracy
                    yhat_pred = yhat > 0.5
                    syl_train = torch.sum(ytrain, dim = 0).float()
                    syl_hat = torch.sum(yhat_pred, dim = 0).float()
                    batch_acc = torch.sum(syl_train == syl_hat).float()
                    batch_acc /= ytrain.shape[1]
                    avg_acc = ema * avg_acc + (1 - ema) * batch_acc

                    # Bias correction
                    # Note: The factors 20 and  25 was found empirically
                    acc_batch += 1
                    avg_loss /= 1 - ema ** (acc_batch * 25)
                    avg_acc /= 1 - ema ** (acc_batch * 20)

                    # Update progress bar description
                    desc = f'Epoch {epoch:2d} - loss {avg_loss:.4f}'\
                           f' - acc {avg_acc:.4f}'
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(train_loader.batch_size)

                # Compute validation metrics
                with torch.no_grad():

                    # Enable validation mode
                    self.eval()

                    val_loss = 0
                    syl_acc = 0
                    TP, TN, FP, FN = 0, 0, 0, 0
                    for xval, yval in val_loader:
                        yhat = self.forward(xval)

                        val_loss += criterion(yhat, yval)

                        yhat = yhat > pred_threshold
                        yval = yval > pred_threshold
                        TP += torch.sum(yhat & yval).float()
                        TN += torch.sum(~yhat & ~yval).float()
                        FP += torch.sum(yhat & ~yval).float()
                        FN += torch.sum(~yhat & yval).float()

                        syl_val = torch.sum(yval, dim = 0).float()
                        syl_hat = torch.sum(yhat, dim = 0).float()
                        batch_acc = torch.sum(syl_val == syl_hat).float()
                        batch_acc /= yval.shape[1]
                        syl_acc += batch_acc

                    # Calculate average validation loss
                    val_loss /= len(val_loader)

                    # Calculate syllable accuracy
                    syl_acc /= len(val_loader)

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
                    self.history['val_acc'].append(syl_acc)

                    # Update progress bar description
                    desc = f'Epoch {epoch:2d} - '\
                           f'loss {avg_loss:.4f} - '\
                           f'acc {avg_acc:.4f} - '\
                           f'val_loss {val_loss:.4f} - '\
                           f'val_acc {syl_acc:.4f} - '\
                           f'val_f1 {val_f1:.4f}'
                    epoch_pbar.set_description(desc)

            # Add score to learning scheduler
            if scheduler is not None:
                scheduler.step(scores[-1])

            # Save model if score is best so far
            if score_cmp(scores[-1], best_score):
                best_score = scores[-1]

                # Delete older models
                for p in Path('.').glob('counter*.pt'):
                    p.unlink()

                self.save(f'counter_{scores[-1]:.4f}_{monitor}.pt',
                    optimizer = optimizer, scheduler = scheduler)

            # Stop if score has not improved for <patience> many epochs
            if score_cmp(best_score, scores[-1]):
                bad_epochs += 1
                if bad_epochs > patience:
                    print('Model is not improving, stopping training.')
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

            syl_acc = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            for xval, yval in dataloader:
                yhat = self.forward(xval)

                yhat = yhat > pred_threshold
                yval = yval > pred_threshold
                TP += torch.sum(yhat & yval).float()
                TN += torch.sum(~yhat & ~yval).float()
                FP += torch.sum(yhat & ~yval).float()
                FN += torch.sum(~yhat & yval).float()

                syl_val = torch.sum(yval, dim = 0).float()
                syl_hat = torch.sum(yhat, dim = 0).float()
                batch_acc = torch.sum(syl_val == syl_hat).float()
                batch_acc /= yval.shape[1]
                syl_acc += batch_acc

            # Calculate syllable accuracy
            syl_acc /= len(dataloader)

            # Calculate other scores
            scores = self.get_scores(TP,TN,FP,FN)
            prec = scores['precision']
            rec = scores['recall']
            f1 = scores['F_score']

            print('\n' + '~' * 23 + '  REPORT  ' + '~' * 23)
            print('Accuracy\tPrecision\tRecall\t\tF1 score')
            print(f'{syl_acc:.4f}\t\t{prec:.4f}\t\t{rec:.4f}\t\t{f1:.4f}')

            scores = {
                'accuracy': syl_acc, 
                'precision': prec, 
                'recall': rec, 
                'f1': f1
                }

            return scores

class TBatchNorm(Module):
    ''' Applies temporal batch normalisation.

    INPUT
        hidden_size
            The number of hidden features
        verbose = 0
            Verbosity mode, can be 0, 1 or 2
    '''
    def __init__(self, hidden_size, verbose = 0):
        super().__init__(verbose)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        ''' Forward pass of the network.

        INPUT
            Tensor of shape (seq_len, batch_size, hidden_size)

        OUTPUT
            Tensor of shape (seq_len, batch_size, hidden_size)
        '''
        x = x.permute(1, 2, 0)
        x = self.bn(x)
        x = x.permute(2, 0, 1)
        return x
            
if __name__ == '__main__':
    pass
