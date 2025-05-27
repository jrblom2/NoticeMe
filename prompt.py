from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Set the device
device = torch.device("cuda:0")

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./tinyllama-finetuned").to(device)

# Load the tokenizer (you can reuse the same one used during training)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

prompt = (
    "The following is a conversation with me and my waifu girlfriend\n\n"
    "Me: Hello\nGirlfriend: Hello\n"
    "Me: How are you?\nGirlfriend: I am good\n"
    "Me: I love you.\nGirlfriend: I love you too."
    "Me: What should we do now?\n"
)
response = generator(prompt, max_length=100, do_sample=True, temperature=0.8)

print(response[0]["generated_text"])
