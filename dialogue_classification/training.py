import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import MAX_UTTERANCE_LEN, device
from .train_test_split import train_set_idx, valid_set_idx, test_set_idx
from .dataset import DialogueDataset
from .models import Net

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


def train_model(data, talk_names, net_conf=None, lr_bert=0.0001, lr_head=0.0005, pct_start=0.2, weight_decay=0,
                class_weight=None, bert_finetuning_epochs=3, batch_size=32, n_epochs=10, verbose=False, plot=False):
    net_conf = net_conf or {}

    net = Net(**net_conf).to(device)

    train_idx = np.isin(talk_names, train_set_idx)
    valid_idx = np.isin(talk_names, valid_set_idx)
    test_idx = np.isin(talk_names, test_set_idx)

    train_set = DialogueDataset(*[arr[train_idx] for arr in data])
    valid_set = DialogueDataset(*[arr[valid_idx] for arr in data])
    test_set = DialogueDataset(*[arr[test_idx] for arr in data])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.Adam(
        [
            {'params': [p for p in net.bert.parameters()], 'lr': lr_bert},
            {'params': [p for p in net.not_bert_params.parameters()], 'lr': lr_head},
        ],
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_set) // batch_size + 1,
                                              epochs=n_epochs, anneal_strategy='cos', max_lr=[lr_bert, lr_head],
                                              pct_start=pct_start)

    best_score = 0
    epochs_without_improvement = 0
    all_train_losses, all_val_losses, all_scores = [], [], []

    for epoch in range(n_epochs):
        if epoch == bert_finetuning_epochs:  # freeze Bert weights after n epochs
            net.freeze_bert()

        # TRAIN EPOCH
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        all_train_losses += losses

        if verbose:
            print(f'[EPOCH {epoch}] Train loss: {np.mean(losses)}')

        # VAL EPOCH
        net.eval()
        losses = []
        total, correct = 0, 0
        for batch in valid_loader:
            for k, v in batch.items():
                if k != 'input_lengths':
                    batch[k] = v.to(device)
            with torch.no_grad():
                outputs = net(**batch)
            masked_preds, masked_labels = masking(outputs, batch['labels'], batch['input_lengths'])
            loss = criterion(masked_preds, masked_labels)
            losses.append(loss.item())
            _, predicted = torch.max(masked_preds.data, 1)
            total += masked_labels.size(0)
            correct += (predicted == masked_labels).sum().item()

        score = correct / total
        all_scores.append(score)
        all_val_losses.append(np.mean(losses))
        if verbose:
            print(f'[EPOCH {epoch}] Val loss: {np.mean(losses)}, Accuracy: {score}')

        # TEST EPOCH
        net.eval()
        losses = []
        total, correct = 0, 0
        for batch in test_loader:
            for k, v in batch.items():
                if k != 'input_lengths':
                    batch[k] = v.to(device)
            with torch.no_grad():
                outputs = net(**batch)
            masked_preds, masked_labels = masking(outputs, batch['labels'], batch['input_lengths'])
            loss = criterion(masked_preds, masked_labels)
            losses.append(loss.item())
            _, predicted = torch.max(masked_preds.data, 1)
            total += masked_labels.size(0)
            correct += (predicted == masked_labels).sum().item()

        test_score = correct / total
        if verbose:
            print(f'[EPOCH {epoch}] Test loss: {np.mean(losses)}, Accuracy: {test_score}')

        # EARLY_STOPPING
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 3:
                break

    if plot:
        steps_per_epoch = len(train_set) // batch_size + 1
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
        ax1.plot(all_train_losses, label='train_loss')
        ax1.plot(steps_per_epoch * np.arange(len(all_val_losses)), all_val_losses, marker='o', label='val_loss')
        ax1.set_xticks(steps_per_epoch * np.arange(len(all_val_losses)))
        ax1.set_xticklabels(np.arange(len(all_val_losses)))
        ax2.plot(all_scores, label='accuracy', marker='o')
        ax1.legend()
        ax2.legend()
        plt.show()

    del net

    return test_score
