from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle

from AbcTokenizer import AbcTokenizer


class AbcDataset(Dataset):
    def __init__(self, root='./abc_dataset', max_len=512, split="train"):
        # self.data = datasets.load_dataset('sander-wood/irishman', split='validation')
        #
        # data_dict = self.data.to_dict()
        #
        # with open('validation.pkl', 'wb') as handle:
        #     pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{root}/{split}.pkl', 'rb') as handle:
            self.data = pickle.load(handle)

        self.max_len = max_len

        self.tokenizer = AbcTokenizer()

        self.vocab_size = 128

    def __len__(self):
        return len(self.data['abc notation'])

    def __getitem__(self, idx):
        processed_abc = torch.tensor(self.tokenizer(self.data['abc notation'][idx]), dtype=torch.int64)
        processed_abc = processed_abc.reshape(-1)

        processed_abc = processed_abc[:self.max_len]

        pad_length = self.max_len - len(processed_abc)
        padded_seq = torch.cat([processed_abc, torch.tensor([0] * pad_length, dtype=processed_abc.dtype)], dim=0)

        return padded_seq
