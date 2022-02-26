import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
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


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimension of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch_size, seq_len, dim): tensor containing projection vector for decoder.
        - **key** (batch_size, seq_len, dim): tensor containing projection vector for encoder.
        - **value** (batch_size, seq_len, dim): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context** (batch_size, seq_len, dim): tensor containing the context vector from attention mechanism.
        - **attn** (batch_size, seq_len, seq_len): tensor containing the attention (alignment) from the encoder outputs.
    Reference:
        - https://github.com/sooftware/attentions/blob/master/attentions.py
    """
    def __init__(self, dim: int):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(self.query_proj(query), self.key_proj(key).transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            # -inf becomes 0 after softmax
            mask = mask.unsqueeze(-1).expand((mask.size(0), mask.size(1), score.size(2)))
            score.masked_fill_(~mask, -INF)

        attn = score.softmax(dim=-1)
        context = torch.bmm(attn, value)
        return context, attn


class Net(nn.Module):
    def __init__(self, input_size=100, output_size=10, rnn_hidden_size=64, input_dropout=0, bert_layers_to_finetune=0,
                 use_attention=True, attention_dropout=0, attention_concat=True, use_layer_norm=True):
        super().__init__()

        self.use_attention = use_attention
        self.attention_concat = attention_concat
        self.use_layer_norm = use_layer_norm

        if self.use_attention:
            self.attention = ScaledDotProductAttention(dim=2 * rnn_hidden_size)
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

        if use_attention and attention_concat:
            final_size = 4 * rnn_hidden_size
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

        # Masks for input sequences
        indexes = torch.arange(0, seq_len).expand(batch_size, seq_len)
        mask = indexes < input_lengths.view(-1, 1)
        mask = mask.to(device)

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

        if self.use_attention:
            normed_rnn_output = self.norm1(rnn_output)
            output, _ = self.attention(normed_rnn_output, normed_rnn_output, normed_rnn_output, mask)
            output = self.attention_dropout(output)
            if self.attention_concat:
                output = torch.cat((output, rnn_output), dim=-1)
        else:
            output = rnn_output

        if self.use_layer_norm:
            output = self.norm2(output)

        return self.head(output)

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False
