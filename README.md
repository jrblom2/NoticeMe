# NoticeMe

Some Notes about approach:

This article has some good guidance but we will end up doing something pretty different:

https://gmongaras.medium.com/coding-a-virtual-ai-girlfriend-f951e648aa46

This project will really have two parts that should equal a conversational horn dog. The model we use to start with and then how we build the context

1. The model:
    After some research, any model we could write or train will not be able to do this task. As such we should focus on using the most powerful Hugging face model
    that our computers can load and either doing some sort of cross analysis between a bunch of them and the performance at the end or some sort of fine-tuning task. Fine-tuning is something we covered in class so I think David would like to see it and that could be a good focus of the project but we would need the right data and it would probably have to happen on the sheep, these models are BIG. I think 1.0-1.5B params is the sweet spot for our laptops.

    Some modesl to consider:

    GPTNeo https://huggingface.co/EleutherAI/gpt-neo-1.3B

    Cursed Waifu GPTNeo https://huggingface.co/gmongaras/gpt-anime-sub-1.3B

    TinyLlama (my fav) https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

    Datasets (they are all what you expect):

    This one is pretty good

    https://huggingface.co/datasets/Himitsui/Lewd-Assistant-v1?not-for-all-audiences=true

2. The context:
    A very common context length I have seen is 2048, so we should assume that is how many characters we have to work with. Tokens are usually bigger than one character but we should be conservative to be safe. In order to answer a prompt in an acceptably horny fashion, the model needs three things. It needs to be primed to be thirsty, the previous conversation, and the actual prompt it is being asked. I think while there are some more advanced ways to accomplish these goals (maybe with an adapter?) we should definetly stick to context building for now.

    We should split our 2048 into several parts, one part for each thing we need it to know.
    1. A static part representing its personalitly, probably the largest part
    2. The conversation so far, maybe just the previous prompts and responses that will fit?
    3. Our prompt. We should keep it short so the other sections have more room.

    We should then be able to just deadass concat all these things together and update the 2 and 3 as needed and get some decent results.