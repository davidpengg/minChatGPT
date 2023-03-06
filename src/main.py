from gpt import GPT, GPTRewardModel, HFGPTRewardModel
from llama import LLaMA, ModelArgs
from dataset import AnthropicHHRLHFDataset, DahoasRMStaticDataset, DahoasSFTStaticDataset, EYLSFTStaticDataset
import torch
import tiktoken
import click
from tokenizer import LLaMATokenizer
from torch.utils.data import DataLoader
from trainers import RewardModelTrainer, SFTTrainer
from torchinfo import summary
import json


@click.command()
@click.option('--task', '-t')
def main(task):
    device = 'cuda'

    if task == 'gpt':
        max_new_tokens = 20
        num_samples = 8
        temperature = 0.9
        top_k = 200
        prompt = "Hello, my name is"
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        indices = encode(prompt)
        x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
        model = GPT.from_pretrained()
        model.eval()
        model.to(device)
        for k in range(num_samples):
            y = model.generate(x,
                               max_new_tokens,
                               temperature=temperature,
                               top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
    elif task == 'llama':
        num_samples = 3
        max_new_tokens = 20
        prompt = """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:"""
        ckpt_path = '../models/7B/consolidated.00.pth'
        params_path = '../models/7B/params.json'
        tokenizer_path = '../models/tokenizer.model'
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        tokenizer = LLaMATokenizer(model_path=tokenizer_path)
        with open(params_path) as fp:
            params = json.loads(fp.read())
        model_args: ModelArgs = ModelArgs(max_seq_len=1024,
                                          max_batch_size=32,
                                          **params)
        model_args.vocab_size = tokenizer.n_words

        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = LLaMA(model_args)
        # model.to(device)
        # model.eval()
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

        with torch.inference_mode():
            for k in range(num_samples):
                x = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False),
                                 dtype=torch.long,
                                 device=device)[None, ...]
                y = model.generate(x,
                                   max_new_tokens)
                print(tokenizer.decode(y[0].tolist()))
                print('---------------')
    elif task == "reward":
        rm = GPTRewardModel.from_pretrained('gpt2-xl')
        rm.eval()
        rm.to(device)
        score = rm(x)
        print(score)
    elif task == "dataset":
        from datasets import load_dataset
        dataset = load_dataset("Anthropic/hh-rlhf", split='train')
        print(dataset[0])
    elif task == "test_dataset":
        model = GPT.from_pretrained('gpt2-large')
        model.eval()
        model.to(device)
        ds = AnthropicHHRLHFDataset()
        dl = DataLoader(ds, batch_size=1)
        for d in dl:
            print('Positive---------------')
            x = d[0].to(device)
            y = model.generate(x,
                               max_new_tokens,
                               temperature=temperature,
                               top_k=top_k)
            print(decode(y[0].tolist()))
            print('Negative---------------')
            x = d[1].to(device)
            y = model.generate(x,
                               max_new_tokens,
                               temperature=temperature,
                               top_k=top_k)
            print(decode(y[0].tolist()))
            print('End---------------')
    elif task == "sft":
        model_name = 'gpt2-medium'
        model = GPT.from_pretrained(model_name)
        summary(model, input_data=torch.ones(1, 1024).long())
        train_ds = EYLSFTStaticDataset(block_size=1024,
                                       split='train',
                                       max_examples=None,
                                       tokenizer_name="tiktoken/gpt2")
        test_ds = EYLSFTStaticDataset(block_size=1024,
                                      split='test',
                                      max_examples=None,
                                      tokenizer_name="tiktoken/gpt2")
        trainer = SFTTrainer(device,
                             model,
                             train_ds,
                             test_ds,
                             batch_size=4,
                             max_steps=300000,
                             name=model_name,
                             finetune=False)
        trainer.fit()
    elif task == "train_rm":
        model_name = 'gpt2-large/lora'
        rm = GPTRewardModel.from_pretrained(model_name)
        summary(rm, input_data=torch.ones(1, 1024).long())
        train_ds = DahoasRMStaticDataset(block_size=1024,
                                         split='train',
                                         max_examples=None,
                                         tokenizer_name="tiktoken/gpt2")
        test_ds = DahoasRMStaticDataset(block_size=1024,
                                        split='test',
                                        max_examples=None,
                                        tokenizer_name="tiktoken/gpt2")
        trainer = RewardModelTrainer(device,
                                     rm,
                                     train_ds,
                                     test_ds,
                                     batch_size=1,
                                     total_epochs=1,
                                     name=model_name,
                                     finetune="lora")

        trainer.fit()
    elif task == "test_loss":
        from loss import KPairwiseLoss
        loss_func = KPairwiseLoss()
        scores = torch.tensor([[0.8, 0.4], [0.5, 0.6]])
        loss = loss_func(scores)
        print(loss)
    elif task == "test_tokenizer":
        from dataset import TiktokenTokenizer
        from transformers import GPT2Tokenizer, GPT2TokenizerFast
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.pad_token)
        print(
            tokenizer("How are you?<|endoftext|>",
                      max_length=20,
                      padding="max_length",
                      truncation=True,
                      return_tensors="pt"))
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.pad_token)
        print(
            tokenizer("How are you?",
                      max_length=20,
                      padding="max_length",
                      truncation=True,
                      return_tensors="pt"))

        tokenizer = TiktokenTokenizer('gpt2')
        print(tokenizer.pad_token)
        print(
            tokenizer("How are you?",
                      max_length=20,
                      padding="max_length",
                      truncation=True,
                      return_tensors="pt"))


if __name__ == "__main__":
    main()
