from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

device = torch.device("cuda:0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16).to(
    device
)

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True)

ds = load_dataset("Himitsui/Lewd-Assistant-v1")


def format_conversation(example):
    return {"text": f"Me: {example['instruction']}\nGirlfriend: {example['output']}"}


formatted_dataset = ds.map(format_conversation)


def tokenize(example):
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = [(label if label != tokenizer.pad_token_id else -100) for label in tokenized["input_ids"]]
    return tokenized


tokenized_dataset = formatted_dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=False,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()
