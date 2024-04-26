import tiktoken
import torch
from configs import get_configs
from gpt import GPT
import argparse
import json
from tqdm import tqdm

cfg = get_configs("gpt2-medium")

gpuid = 0
device = f"cuda:{gpuid}"

print("loading models...")
import time
start = time.time()
base = GPT.from_checkpoint(
    cfg,
    "./runs/BASE-sft_default_202404240102/sft_default_202404240102_step0.pt",
    map_location=device
)
print(f"took {time.time() - start:.2f} to load base")
start = time.time()
sft = GPT.from_checkpoint(
    cfg,
    "./runs/SFT-sft_default_202404141028/sft_default_202404141028_final.pt",
    map_location=device
)
print(f"took {time.time() - start:.2f} to load sft")

def wrap_prompt(prompt):
    if prompt.startswith("Instruction: "):
        prompt = prompt[len("Instruction: "):]
    if prompt.endswith("\nResponse: "):
        prompt = prompt[:len("\nResponse: ")]
    return f"Human: {prompt}\nAssistant:"

def prepare_gpt2_input(prompt, device):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode

def process_decode(decoded):
    if "<|endoftext|>" in decoded:
        eos_id = decoded.find("<|endoftext|>")
        decoded = decoded[:eos_id]
    assistant_split = decoded.split('Human:')
    if len(assistant_split) > 2:
        decoded = ''.join(assistant_split[:2]).strip()
    return decoded

def generate_gpt2(prompt, device="cuda:0"):
    model_a = base
    model_b = sft

    model_a.eval()
    model_a.to(device)
    model_b.eval()
    model_b.to(device)
    max_new_tokens = 500
    temperature = 0.9
    top_k = 200
    x, decode = prepare_gpt2_input(prompt, device)
    print("prepared")

    y_a = model_a.generate(x,
                        max_new_tokens,
                        temperature=temperature,
                        top_k=top_k)
    print("generated A")
    y_b = model_b.generate(x,
                        max_new_tokens,
                        temperature=temperature,
                        top_k=top_k)
    print("generated B")
    response_a = decode(y_a[0].tolist())
    response_b = decode(y_b[0].tolist())
    
    response_a = process_decode(response_a)
    response_b = process_decode(response_b)
    print("decoded")

    return response_a, response_b

def generate_from_text(data_path, output_path):
    with open(data_path, 'r') as f_data, open(output_path, 'w') as f_write:
        for example in tqdm(f_data, total=100):
            data = json.loads(example)
            prompt = wrap_prompt(data['instruction'])

            response_a, response_b = generate_gpt2(prompt, device)
            result = {
                "base_gpt2": response_a,
                "sft_gpt2": response_b,
                "instruction": prompt
            }
            print(json.dumps(result), file=f_write, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    if args.data_path is not None and args.output_path is not None:
        generate_from_text(args)
    else:
        while (True):
            prompt = input("Enter prompt: ")
            prompt = wrap_prompt(prompt)

            response_a, response_b = generate_gpt2(prompt, device)
            print("BASE")
            print(response_a)
            print("==========================")
            print("SFT")
            print(response_b)
            print("...")
