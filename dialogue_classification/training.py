import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from .config import device

random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


def masking(preds, labels, input_lengths):
    """Mask sequences of various lengths before applying CrossEntropy Loss."""
    batch_size, seq_len = preds.size(0), preds.size(1)
    indexes = torch.arange(0, seq_len).expand(batch_size, seq_len)
    mask = indexes < input_lengths.view(-1, 1)
    mask = mask.to(device)
    return preds[mask].view(-1, preds.size(2)), labels[mask].view(-1).long()


def train_epoch(net, train_loader, optimizer, criterion):
    net.train()
    losses = []
    for batch in train_loader:
        for k, v in batch.items():
            if k != 'input_lengths':
                batch[k] = v.to(device)
        optimizer.zero_grad()
        outputs = net(**batch)
        masked_preds, masked_labels = masking(outputs, batch['labels'], batch['input_lengths'])
        loss = criterion(masked_preds, masked_labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return losses


def val_epoch(net, valid_loader, criterion):
    net.eval()
    losses = []
    all_labels, all_preds = [], []
    for batch in valid_loader:
        for k, v in batch.items():
            if k != 'input_lengths':
                batch[k] = v.to(device)
        with torch.no_grad():
            outputs = net(**batch)
        masked_preds, masked_labels = masking(outputs, batch['labels'], batch['input_lengths'])
        loss = criterion(masked_preds, masked_labels)
        losses.append(loss.item())
        all_labels.append(masked_labels.detach().cpu().numpy())
        all_preds.append(torch.max(masked_preds.data, 1)[1].detach().cpu().numpy())
        
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    f1 = f1_score(all_labels, all_preds, average='micro')
    
    return losses, f1


def train_model(net, train_loader, valid_loader, test_loader, optimizer, bert_finetuning_epochs=3, n_epochs=10,
                verbose=False, plot=False):
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, min_lr=1e-7, verbose=verbose)

    best_score = 0
    epochs_without_improvement = 0
    all_train_losses, all_val_losses, all_scores = [], [], []

    for epoch in range(n_epochs):
        if bert_finetuning_epochs and epoch == bert_finetuning_epochs:  # freeze Bert weights after n epochs
            net.freeze_bert()
        
        # Train
        losses = train_epoch(net, train_loader, optimizer, criterion)
        all_train_losses += losses
        if verbose:
            print(f'[EPOCH {epoch}] Train loss: {np.mean(losses)}')
            
        # Val
        losses, f1_val = val_epoch(net, valid_loader, criterion)
        all_val_losses.append(np.mean(losses))
        scheduler.step(np.mean(losses))
        all_scores.append(f1_val)
        if verbose:
            print(f'[EPOCH {epoch}] Val loss: {np.mean(losses)}, F1-score: {f1_val}')
            
        # Test
        losses, f1_test = val_epoch(net, test_loader, criterion)
        if verbose:
            print(f'[EPOCH {epoch}] Test loss: {np.mean(losses)}, F1-score: {f1_test}')

        # Early stopping
        if f1_val > best_score:
            best_score = f1_val
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 3:
                print("Early stopping")
                break

    if plot:
        steps_per_epoch = len(train_loader) * train_loader.batch_size // train_loader.batch_size + 1
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
        ax1.plot(all_train_losses, label='train_loss')
        ax1.plot(steps_per_epoch * np.arange(len(all_val_losses)), all_val_losses, marker='o', label='val_loss')
        ax1.set_xticks(steps_per_epoch * np.arange(len(all_val_losses)))
        ax1.set_xticklabels(np.arange(len(all_val_losses)))
        ax2.plot(all_scores, label='F1', marker='o')
        ax1.legend()
        ax2.legend()
        plt.show()
                                     