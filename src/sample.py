from main import generate_gpt2
from configs import get_configs
from gpt import GPT

cfg = get_configs("gpt2-medium")

print("loading models...")
base = GPT.from_pretrained(cfg)
sft = GPT.from_checkpoint(
    cfg,
    "./runs/sft_default_202404141028/sft_default_202404141028_final.pt"
)

device = "cuda:0"

while (True):
    prompt = input("Enter prompt: ")
    prompt = f"Human: {prompt}\nAssistant:"

    print("BASE")
    generate_gpt2(base, prompt, device)
    print("SFT")
    generate_gpt2(sft, prompt, device)
    print("...")