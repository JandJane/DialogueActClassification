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
    
    
class DialogueRNN(nn.Module):
    def __init__(self, input_size=300, output_size=10, rnn_hidden_size=64, use_layer_norm=True):
        super().__init__()

        self.rnn = nn.GRU(input_size, rnn_hidden_size, batch_first=True, bidirectional=False)
        
        self.use_layer_norm = use_layer_norm
        self.norm = nn.LayerNorm([rnn_hidden_size])

        self.head = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size),
            nn.ReLU(),
            nn.Linear(rnn_hidden_size, output_size),
        )

    def forward(self, utterance_embs, input_lengths):
        batch_size, seq_len = utterance_embs.size(0), utterance_embs.size(1)

        # Masks for input sequences
        indexes = torch.arange(0, seq_len).expand(batch_size, seq_len)
        mask = indexes < input_lengths.view(-1, 1)
        mask = mask.to(device)

        rnn_output, _ = self.rnn(utterance_embs)

        if self.use_layer_norm:
            rnn_output = self.norm(rnn_output)

        return self.head(rnn_output)


class BertEmbedder(nn.Module):
    def __init__(self, bert_layers_to_finetune=0, max_utterance_len=MAX_UTTERANCE_LEN):
        super().__init__()
        
        self.max_utterance_len = max_utterance_len
        self.bert = BertModel.from_pretrained('bert-base-cased')

        for param in self.bert.parameters():
            param.requires_grad = False

        if bert_layers_to_finetune:
            for param in self.bert.encoder.layer[-bert_layers_to_finetune:].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_masks):
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)

        # Get Bert utterance embeddings
        bert_output = self.bert(
            input_ids=input_ids.reshape(-1, self.max_utterance_len),
            attention_mask=attention_masks.reshape(-1, self.max_utterance_len)
        ).last_hidden_state
        attention_masks = attention_masks.reshape(-1, self.max_utterance_len, 1)
        bert_output = (bert_output * attention_masks).sum(dim=1) / (attention_masks.sum(dim=1) + EPS)
        bert_output = bert_output.view(batch_size, seq_len, -1)   # Pooling ?
        
        return bert_output

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False
            
            
class RNNEmbedder(nn.Module):
    def __init__(self, input_size=300, hidden_size=300, max_utterance_len=MAX_UTTERANCE_LEN):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_utterance_len = max_utterance_len
        
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        batch_size, seq_len = inputs.size(0), inputs.size(1)
        
        _, rnn_output = self.rnn(
            inputs.reshape(batch_size * seq_len, self.max_utterance_len, self.input_size)
        )
        rnn_output = rnn_output.transpose(0, 1).reshape(batch_size, seq_len, 2 * self.hidden_size)

        return rnn_output


class DialogueNetBert(nn.Module):
    def __init__(self, input_size=100, output_size=10, rnn_hidden_size=64, bert_layers_to_finetune=0,
                 use_layer_norm=True, max_utterance_len=MAX_UTTERANCE_LEN):
        super().__init__()
        
        self.bert_embedder = BertEmbedder(bert_layers_to_finetune=bert_layers_to_finetune, max_utterance_len=max_utterance_len)
        
        self.dialogue_rnn = DialogueRNN(input_size=768, output_size=output_size, rnn_hidden_size=rnn_hidden_size, 
                                        use_layer_norm=use_layer_norm)

    def forward(self, input_ids, attention_masks, input_lengths, labels=None):
        bert_embs = self.bert_embedder(input_ids, attention_masks)
        outputs = self.dialogue_rnn(bert_embs, input_lengths)
        return outputs
    
    
class DialogueNetFasttext(nn.Module):
    def __init__(self, input_size=300, output_size=10, rnn_hidden_size=64, use_layer_norm=True, max_utterance_len=MAX_UTTERANCE_LEN):
        super().__init__()
        
        self.rnn_embedder = RNNEmbedder(input_size=input_size, hidden_size=rnn_hidden_size, max_utterance_len=max_utterance_len)
        
        self.dialogue_rnn = DialogueRNN(input_size=2 * rnn_hidden_size, output_size=output_size, rnn_hidden_size=rnn_hidden_size,
                                        use_layer_norm=use_layer_norm)

    def forward(self, inputs, input_lengths, labels=None):
        fasttext_embs = self.rnn_embedder(inputs)
        outputs = self.dialogue_rnn(fasttext_embs, input_lengths)
        return outputs
        