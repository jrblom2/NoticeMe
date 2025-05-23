from transformers import pipeline
import torch

initial_prompt = (
    "The following is a conversation with me and my waifu girlfriend\n\n"
    "Me: Hello\nGirlfriend: Hello\n"
    "Me: How are you?\nGirlfriend: I am good\n"
    "Me: I love you.\nGirlfriend: I love you too."
    "Me: What should we do now?\n"
)

# Load in the model
test_model = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device=torch.device("cuda:0"),
)


# Used to get a response from the model
def get_response(input_text):
    return test_model(input_text)[0]["generated_text"]


print(get_response(initial_prompt))
