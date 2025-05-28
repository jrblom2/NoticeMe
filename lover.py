from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json

class Lover:
    def __init__(self, initial_prompt="", initial_summary="", saved_memory=None):
        self.device = torch.device("cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained("./tinyllama-finetuned").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0)
        self.initial_prompt = initial_prompt
        self.initial_summary = initial_summary
        self.past_summ = initial_summary
        self.past_output = ["" for i in range(self.num_blocks)]
        self.cur_prompt = initial_prompt
        self.disc_json = dict(
            past_summ=self.past_summ,
            past_output=self.past_output,
            cur_prompt=self.cur_prompt
        )
        if saved_memory is not None:
            self.load_memory(saved_memory)

    def chat(self, prompt):
        response = self.generator(prompt, max_length=100, do_sample=True, temperature=0.8)
        return response[0]["generated_text"]

    def load_memory(self, filename):
        try:
            self.disc_json = json.load(open(filename, "r"))
            assert "past_summ" in self.disc_json, "Loaded file must have past_summ key"
            assert "past_output" in self.disc_json, "Loaded file must have past_output key"
            assert "cur_prompt" in self.disc_json, "Loaded file must have cur_prompt key"
            self.past_summ = self.disc_json["past_summ"]
            self.past_output = self.disc_json["past_output"]
            self.cur_prompt = self.disc_json["cur_prompt"]
            return f"Successfully loaded memory from {filename}"
        except:
            return "Files to load memory file."
    
lover = Lover()
prompt = (
    "The following is a conversation with me and my waifu girlfriend\n\n"
    "Me: Hello\nGirlfriend: Hello\n"
    "Me: How are you?\nGirlfriend: I am good\n"
    "Me: I love you.\nGirlfriend: I love you too.\n"
    "Me: What should we do now?\n"
)
print(lover.chat(prompt))