import argparse
import sys

import torch
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel, ErnieForMaskedLM, ErnieConfig
from tqdm import tqdm
from dataset import AbcDataset
from torch.utils.data.dataloader import DataLoader

from evaluation_metrics import evaluate_mask_prediction_accuracy, evaluate_next_token_accuracy
from utils import remove_extra_pre_fix


def main(args, model_weights):
    dataset = AbcDataset(max_len=args['max_len'], split='validation')
    dataloader = DataLoader(dataset, batch_size=args['batch_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # configuration = BertConfig(vocab_size=dataset.vocab_size, output_hidden_states=True)
    # # Initializing a model (with random weights) from the bert-base-uncased style configuration
    # model = BertForMaskedLM(configuration).to(device)
    # Initializing a model (with random weights) from the bert-base-uncased style configuration
    if args['model'] == 'bert':
        configuration = BertConfig(vocab_size=dataset.vocab_size, output_hidden_states=True)
        model = BertForMaskedLM(configuration).to(device)
        accuracy_function = evaluate_mask_prediction_accuracy
    elif args['model'] == 'gpt2':
        configuration = GPT2Config(vocab_size=dataset.vocab_size)
        model = GPT2LMHeadModel(configuration).to(device)
        accuracy_function = evaluate_next_token_accuracy
    elif args['model'] == 'ernie':
        configuration = ErnieConfig(vocab_size=dataset.vocab_size)
        model = ErnieForMaskedLM(configuration).to(device)
        accuracy_function = evaluate_mask_prediction_accuracy
    else:
        raise ValueError("Model is not supported")

    if args['multi_gpu']:
        model.load_state_dict(remove_extra_pre_fix(model_weights))
    else:
        model.load_state_dict(model_weights)

    model.eval()

    acc = accuracy_function(model, dataloader, device)

    print(f'Accuracy: {acc}')

    neg_log_likelihood = []
    for i, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss
        # perplexity = torch.exp(loss)
        neg_log_likelihood.append(loss.item())
        # perplexity += perplexity.item()
        # print(f'Perplexity: {perplexity.item()}')

    perplexity = torch.exp(torch.tensor(neg_log_likelihood).mean())
    sys.stdout.write(f'Perplexity: {perplexity}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store', type=str, default='outputs/2024-04-10_18-51-11/checkpoint_1.pth')
    parser.add_argument('--batch_size', action='store', type=int, default=12)

    args = vars(parser.parse_args())
    batch_size = args['batch_size']
    checkpoint = torch.load(args['checkpoint'])
    args.update(checkpoint['args'])
    args['batch_size'] = batch_size

    main(args, checkpoint['model'])
