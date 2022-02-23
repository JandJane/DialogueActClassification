import torch

MAX_UTTERANCE_LEN = 25

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
