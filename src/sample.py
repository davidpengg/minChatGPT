import tiktoken
import torch
from configs import get_configs
from gpt import GPT

cfg = get_configs("gpt2-medium")

gpuid = input("gpuid: ")
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



def prepare_gpt2_input(prompt, device):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode

def generate_gpt2(model, prompt, device, samples=1):
    model.eval()
    model.to(device)
    max_new_tokens = 500
    temperature = 0.9
    top_k = 200
    x, decode = prepare_gpt2_input(prompt, device)

    for k in range(samples):
        y = model.generate(x,
                           max_new_tokens,
                           temperature=temperature,
                           top_k=top_k)
        decoded = decode(y[0].tolist())
        
        print(decoded)
        print('  -----------  ')
        if "<|endoftext|>" in decoded:
            eos_id = decoded.find("<|endoftext|>")
            decoded = decoded[:eos_id]
        assistant_split = decoded.split('Human:')
        if len(assistant_split) > 2:
            decoded = ''.join(assistant_split[:2]).strip()
        print(decoded)
        print('---------------')

while (True):
    prompt = input("Enter prompt: ")
    prompt = f"Human: {prompt}\nAssistant:"

    print("BASE")
    generate_gpt2(base, prompt, device)
    print("SFT")
    generate_gpt2(sft, prompt, device)
    print("...")