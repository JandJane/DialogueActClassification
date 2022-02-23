import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel

from .config import MAX_UTTERANCE_LEN, device

random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

INF = torch.tensor(100_000).float().to(device)
EPS = 1e-5


class Net(nn.Module):
    def __init__(self, input_size=100, output_size=10, rnn_hidden_size=64, input_dropout=0, bert_layers_to_finetune=0,
                 use_attention=True, attention_dropout=0, attention_concat=True, attention_params=None,
                 use_layer_norm=True):
        super().__init__()

        attention_params = attention_params or {}

        self.use_attention = use_attention
        self.attention_concat = attention_concat
        self.use_layer_norm = use_layer_norm

        if self.use_attention:
            pass # TODO
        else:
            self.attention = None

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False

        if bert_layers_to_finetune:
            for param in self.bert.encoder.layer[-bert_layers_to_finetune:].parameters():
                param.requires_grad = True

        self.input_dropout = nn.Dropout(input_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.rnn = nn.GRU(input_size, rnn_hidden_size, batch_first=True, bidirectional=True)

        self.norm1 = nn.LayerNorm([2 * rnn_hidden_size])

        if use_attention:
            pass
            final_size = self.attention.output_size + 2 * rnn_hidden_size * attention_concat
        else:
            final_size = 2 * rnn_hidden_size

        self.norm2 = nn.LayerNorm([final_size])

        self.head = nn.Sequential(
            nn.Linear(final_size, output_size),
            nn.Softmax()
        )

        self.not_bert_params = nn.ModuleList([self.attention, self.rnn, self.norm1, self.norm2, self.head])

    def forward(self, input_ids, attention_masks, input_lengths, labels=None):
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)

        # Get Bert utterance embeddings
        bert_output = self.bert(
            input_ids=input_ids.reshape(-1, MAX_UTTERANCE_LEN),
            attention_mask=attention_masks.reshape(-1, MAX_UTTERANCE_LEN)
        ).last_hidden_state
        attention_masks = attention_masks.reshape(-1, MAX_UTTERANCE_LEN, 1)
        bert_output = (bert_output * attention_masks).sum(dim=1) / (attention_masks.sum(dim=1) + EPS)
        bert_output = bert_output.view(batch_size, seq_len, -1)

        bert_output = self.input_dropout(bert_output)
        rnn_output, _ = self.rnn(bert_output)

        if self.use_layer_norm:
            rnn_output = self.norm2(rnn_output)

        return self.head(rnn_output)

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False
