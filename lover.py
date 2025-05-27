from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class Lover:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained("./tinyllama-finetuned").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0)

    def chat(self, prompt):
        response = self.generator(prompt, max_length=100, do_sample=True, temperature=0.8)
        return response[0]["generated_text"]
    
lover = Lover()
prompt = (
    "The following is a conversation with me and my waifu girlfriend\n\n"
    "Me: Hello\nGirlfriend: Hello\n"
    "Me: How are you?\nGirlfriend: I am good\n"
    "Me: I love you.\nGirlfriend: I love you too.\n"
    "Me: What should we do now?\n"
)
print(lover.chat(prompt))