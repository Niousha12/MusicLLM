import os
import sys
import argparse
import pickle
import datetime

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    BertForMaskedLM, BertConfig,
    GPT2LMHeadModel, GPT2Config,
    ErnieForMaskedLM, ErnieConfig
)
from dataset import AbcDataset
from tqdm import tqdm


# Define training function
def train_loop(args, dataloader, device, model, optimizer, scheduler):
    model.train()
    epoch_loss_list = []
    saving_path = f"outputs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

    # Ensure saving path exists
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)

    for epoch in range(1, args['epochs'] + 1):
        epoch_loss = 0.0
        if args['multi_gpu']:
            dataloader.sampler.set_epoch(epoch)

        for _, batch in enumerate(tqdm(dataloader)):
            batch = batch.to(device)
            optimizer.zero_grad()
            masked_input, mask_idx = apply_mask(args, batch, device=device)
            out = model(masked_input, labels=batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Logging and saving
        epoch_loss /= len(dataloader)

        epoch_loss_list.append(epoch_loss)

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}: lr {before_lr} -> {after_lr}")
        print(f"Epoch {epoch}, Device {device}: Loss is {epoch_loss}")

        checkpoint = {
            'args': args,
            'model': model.module.state_dict() if args['multi_gpu'] else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss_list': epoch_loss_list,
            'epoch': epoch}

        torch.save(checkpoint, os.path.join(saving_path, f"checkpoint_{epoch}.pth"))


# Apply masking based on model type
def apply_mask(args, batch, device):
    # This function should return the masked_input and mask_idx based on the model type
    mask_percentage = args['mask_percentage']
    mask_idx = None
    if args['model'] == 'bert':
        masked_input = batch.clone().to(device)
        random_mask = torch.rand(masked_input.shape).to(device)  # I can only do this for non-overlapping
        random_mask = (random_mask < mask_percentage) * (masked_input != 0)  # Cannot mask the [<UNK>] token
        mask_idx = (random_mask.flatten() == True).nonzero().view(-1)
        masked_input = masked_input.flatten()
        masked_input[mask_idx] = 1
        masked_input = masked_input.view(batch.size())
    elif args['model'] == 'ernie':
        masked_input = batch.clone().to(device)
        random_mask = torch.rand((masked_input.shape[0], (masked_input.shape[1] // 20) + 1)).to(
            device)  # I can only do this for non-overlapping
        random_mask = (random_mask < mask_percentage)
        random_mask = random_mask.repeat_interleave(20, dim=1)  # Cannot mask the [<UNK>] token
        random_mask = random_mask[:, :masked_input.shape[1]]
        mask_idx = (random_mask.flatten() == True).nonzero().view(-1)
        masked_input = masked_input.flatten()
        masked_input[mask_idx] = 1
        masked_input = masked_input.view(batch.size())
    else:
        masked_input = batch.clone().to(device)

    return masked_input, mask_idx


# Setup function for distributed data parallel (DDP)
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# DataLoader preparation with DistributedSampler
def prepare_dataloader(dataset, rank, world_size, batch_size, pin_memory, num_workers):
    sampler = distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=num_workers, sampler=sampler
    )
    return dataloader


# Main training loop
def run_multi_gpu(rank, world_size, args):
    ddp_setup(rank, world_size)
    dataset = AbcDataset(max_len=args['max_len'], split='train')
    dataloader = prepare_dataloader(dataset, rank, world_size, args['batch_size'], False, 0)
    model, optimizer, scheduler = initialize_model_and_optimizers(args, rank, dataset.vocab_size)
    model = DDP(model, device_ids=[rank])
    train_loop(args, dataloader, rank, model, optimizer, scheduler)
    torch.distributed.destroy_process_group()


def run(device, args):
    dataset = AbcDataset(max_len=args['max_len'], split='train')
    dataloader = DataLoader(dataset, batch_size=args['batch_size'])
    model, optimizer, scheduler = initialize_model_and_optimizers(args, device, dataset.vocab_size)
    train_loop(args, dataloader, device, model, optimizer, scheduler)


# Initialize model and optimizers based on args
def initialize_model_and_optimizers(args, device, vocab_size):
    # Return the model, optimizer, and scheduler
    # Initializing a model (with random weights) from the bert-base-uncased style configuration
    if args['model'] == 'bert':
        configuration = BertConfig(vocab_size=vocab_size, output_hidden_states=True)
        model = BertForMaskedLM(configuration).to(device)
    elif args['model'] == 'gpt2':
        configuration = GPT2Config(vocab_size=vocab_size)
        model = GPT2LMHeadModel(configuration).to(device)
    elif args['model'] == 'ernie':
        configuration = ErnieConfig(vocab_size=vocab_size)
        model = ErnieForMaskedLM(configuration).to(device)
    else:
        raise ValueError("Model is not supported")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, total_iters=5)

    return model, optimizer, scheduler


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', action='store', type=int, default=512)
    parser.add_argument('--model', action='store', type=str, default='bert')
    parser.add_argument('--mask_percentage', action='store', type=float, default=0.5)
    parser.add_argument('--batch_size', action='store', type=int, default=4)
    parser.add_argument('--lr', action='store', type=float, default=1e-4)
    parser.add_argument('--weight_decay', action='store', type=float, default=1e-05)
    parser.add_argument('--epochs', type=int, default=15, help='Total number of training epochs')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='Use multiple GPUs for training')

    args_parsed = parser.parse_args()
    args_dict = vars(args_parsed)

    args_log = "\n".join("{}\t{}".format(k, v) for k, v in sorted(args_dict.items(), key=lambda t: str(t[0])))

    sys.stdout.write(args_log)

    if args_dict['multi_gpu']:
        gpu_count = torch.cuda.device_count()
        torch.multiprocessing.spawn(run_multi_gpu, args=(gpu_count, args_dict), nprocs=gpu_count)
    else:
        run(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), args=args_dict)
