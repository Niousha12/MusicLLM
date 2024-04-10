import argparse
import sys

import torch
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel, ErnieConfig, ErnieForMaskedLM
from dataset import AbcDataset
from torch.utils.data.dataloader import DataLoader

from utils import remove_extra_pre_fix
from tqdm import tqdm

# Set seeds
import torch
import numpy as np

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# If using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

if __name__ == '__main__':
    max_len = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AbcDataset(max_len=max_len, split='validation')
    dataloader = DataLoader(dataset, batch_size=1)

    batch = next(iter(dataloader))
    batch = batch.to(device)

    print(dataset.tokenizer.decode(batch[0].tolist()))

    masked_input = batch.clone()
    random_mask = torch.rand(masked_input.shape).to(device)  # I can only do this for non-overlapping
    random_mask[:, 80:] = (random_mask[:, 80:] < 0.3) * (masked_input[:, 80:] != 0) * (masked_input[:, 80:] != 32)  # Cannot mask the [<UNK>] token
    mask_idx = (random_mask.flatten() == True).nonzero().view(-1)
    print(mask_idx)
    masked_input = masked_input.flatten()
    masked_input[mask_idx] = 1
    masked_input = masked_input.view(batch.size())
    print(dataset.tokenizer.decode(masked_input[0].tolist()))
    # Load the model

    # configuration = BertConfig(vocab_size=dataset.vocab_size)
    # model = BertForMaskedLM(configuration).to(device)
    # model.load_state_dict(remove_extra_pre_fix(torch.load('pretrained_model/bert_50.pth', map_location=device)))

    configuration = ErnieConfig(vocab_size=dataset.vocab_size)
    model = ErnieForMaskedLM(configuration).to(device)
    model.load_state_dict(remove_extra_pre_fix(torch.load('pretrained_model/ernie_50.pth', map_location=device)))

    model.eval()

    with torch.no_grad():
        predictions = model(masked_input, labels=batch).logits
        # Get the predicted token id
        predicted_token_id = predictions.argmax(axis=-1)

        grand_truth = batch.flatten()[mask_idx]
        prediction = predicted_token_id.flatten()[mask_idx]

        masked_input = masked_input.flatten()
        masked_input[mask_idx] = prediction
        masked_input = masked_input.view(batch.size())

    print(dataset.tokenizer.decode(masked_input[0].tolist()))
