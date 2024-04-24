import os
import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from trainers import DPOTrainer
from configs import get_configs
from gpt import GPT
from dataset import DahoasRMStaticDataset

import os
os.environ["TRITON_PTXAS_PATH"] = "/data/lily/dp823/minChatGPT/my_minChatGPT/env/lib/python3.9/site-packages/nvidia/cuda_nvcc/bin/ptxas"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def train(pretrain, batch_size, exp_name, gpuid):
    device = f'cuda:{gpuid}'
    cfg = get_configs("gpt2-medium")
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    cfg.total_epochs = 1
    cfg.exp_name = exp_name

    policy = GPT.from_checkpoint(
        cfg,
        "./runs/SFT-sft_default_202404141028/sft_default_202404141028_final.pt"
    )
    reference = GPT.from_checkpoint(
        cfg,
        "./runs/SFT-sft_default_202404141028/sft_default_202404141028_final.pt"
    )

    train_ds = DahoasRMStaticDataset(block_size=1024,
                                     split='train',
                                     max_examples=20, # TODO tmp
                                     tokenizer_name="tiktoken/gpt2")
    test_ds = DahoasRMStaticDataset(block_size=1024,
                                    split='test',
                                    max_examples=10, # TODO tmp
                                    tokenizer_name="tiktoken/gpt2")
    trainer = DPOTrainer(cfg, device, policy, reference, train_ds, test_ds)
    trainer.fit()


@click.command()
@click.option('--strategy', '-s', default="naive")
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
@click.option('--gpuid', '-g', default='1')
def main(strategy, pretrain, batch_size, exp_name, gpuid):
    torch.manual_seed(1234)
    train(pretrain, batch_size, exp_name, gpuid)


if __name__ == "__main__":
    main()
