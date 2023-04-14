from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, input_ids, attention_masks, input_lengths, y):
        super().__init__()
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.input_lengths = input_lengths
        self.y = y

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_masks': self.attention_masks[idx],
            'input_lengths': self.input_lengths[idx],
            'labels': self.y[idx],
        }

    def __len__(self):
        return len(self.input_ids)

    
class FasttextDataset(Dataset):
    def __init__(self, input_embs, input_lengths, y):
        super().__init__()
        self.input_embs = input_embs
        self.input_lengths = input_lengths
        self.y = y

    def __getitem__(self, idx):
        return {
            'inputs': self.input_embs[idx],
            'input_lengths': self.input_lengths[idx],
            'labels': self.y[idx],
        }

    def __len__(self):
        return len(self.input_embs)