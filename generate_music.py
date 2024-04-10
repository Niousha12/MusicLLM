import argparse
import sys

import torch
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
from dataset import AbcDataset
from torch.utils.data.dataloader import DataLoader

from evaluation_metrics import evaluate_mask_prediction_accuracy, evaluate_next_token_accuracy
from utils import remove_extra_pre_fix
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_len', action='store', type=int, default=512)

    args = vars(parser.parse_args())

    dataset = AbcDataset(max_len=args['max_len'], split='validation')
    dataloader = DataLoader(dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configuration = GPT2Config(vocab_size=dataset.vocab_size)
    model = GPT2LMHeadModel(configuration).to(device)

    checkpoint = torch.load('pretrained_model/gpt2.pth', map_location=device)

    model.load_state_dict(remove_extra_pre_fix(checkpoint))
    model.eval()

    batch = next(iter(dataloader))
    batch = batch.to(device)

    print(dataset.tokenizer.decode(batch[0].tolist()))

    initial_index = 80
    batch[0, initial_index:] = 0

    print(dataset.tokenizer.decode(batch[0].tolist()))

    with torch.no_grad():
        for i in tqdm(range(initial_index, len(batch[0]))):
            outputs = model(batch)
            prediction = outputs.logits[0, i - 1]
            predicted_token_id = prediction.argmax()
            batch[0, i] = predicted_token_id

    print(dataset.tokenizer.decode(batch[0].tolist()))
