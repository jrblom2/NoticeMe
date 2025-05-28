from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from nltk.corpus import stopwords
from string import punctuation
from copy import deepcopy
import torch
import json

class Lover:
    def __init__(self, initial_prompt="", initial_summary="", saved_memory=None):
        self.device = torch.device("cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained("./tinyllama-finetuned").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0)
        self.summarizer = pipeline(
            "summarization",
            'pszemraj/led-large-book-summary',
            device=0 if torch.cuda.is_available() else -1,
            framework="pt",
            torch_dtype=torch.float16,
        )

        # Set up memory
        self.initial_prompt = initial_prompt
        self.initial_summary = initial_summary
        self.summ_size_max = 256 # Used for 1
        self.block_size = 150 # Used for 2 and 3
        self.num_blocks = 4 # Used for 2
        self.past_summ = initial_summary  # 1
        self.past_output = ["" for i in range(self.num_blocks)]  # 2
        self.cur_prompt = initial_prompt  # 3
        self.disc_json = dict(
            past_summ=self.past_summ,
            past_output=self.past_output,
            cur_prompt=self.cur_prompt
        )
        if saved_memory is not None:
            self.load_memory(saved_memory)

    def basic_chat(self, prompt):
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

    def get_model_response(self, text):
        # How many newlines are there?
        num = text.count("\n")

        # Get the model output. at the correct position
        output = self.other_text_model(text)[0]['generated_text'].split("\n")
        output_new = output[num].strip()

        # Make sure the output is not blank
        tmp = 1
        #output_new = output_new.replace("You:", "").replace("Person:", "")
        while output_new == "":
            output_new = output[num+tmp].strip()
            tmp += 1
            
        # If the model is generating newlines after its text,
        # it may want to say more
        cur_out = output_new
        more_max = 0 # Max limit on how much more to add
        more_added = 0 # Current extra added
        while more_added < more_max:
            try:
                if output[num+tmp].strip() == "":
                    break # Break is a \n\n is reached. Keep going if only \n
                out_new = output[num+tmp].strip()
                if out_new not in punctuation:
                    out_new += "."
                cur_out += f" {out_new}"
                more_added += 1
                tmp += 1

                # If a question make was the last letter,
                # stop adding more lines
                if cur_out[-1] == "?":
                    break
            except IndexError:
                break

        return cur_out

    def get_response(self):
        """
        The text used to respond is creafted upon
        all three components in the history.
        It will look like the following:
        [summary of the past]\n\n\n\n
        [saved prompts from the past][current prompt]
        """
        text = self.past_summ + "\n\n" +\
            "".join(self.past_output)+\
            self.cur_prompt

        resp = self.get_model_response(text)

        # Sometimes a stupid output will be placed at the
        # beginning like [Random name]: [words].
        # let's remove these
        resp = resp.split(":")[-1].strip()

        # Add the new text to the prompt
        self.cur_prompt += f"Girlfriend: {resp}\n"

        # Before returning the respnse, we need to make sure
        # the text is being summarized
        self.summarize_text()

        # After the text has been update, update the
        # dictionary and save it
        self.disc_json["cur_prompt"] = self.cur_prompt
        json.dump(self.disc_json, open("config_file.json", "w"))

        # Return the response
        return resp

    # Summary function for a single line using a small model
    def summarize_single(self, text):
        # Get the keywords
        keywords = self.summ_model.extract_keywords(text)

        # Get keywords above a threshold
        words = ", ".join([word[0] for word in keywords])

        return words


    # Get the summary of the text using the large model
    def get_summ(self, text):
        # Get the summary
        summary = self.summarize_single(text)
        
        # Remove stopwords and puncuation from the summary
        filtered = [word for word in self.tokenizer.tokenize(summary) \
            if word not in stopwords.words('english')]
        
        return " ".join(filtered)
    
    def summarize_text(self):
        # If the prompt is over the block size, save it
        # to memory. Summarization comes later
        if len(self.cur_prompt.split(" ")) > self.block_size:
            # Get a subset which of the block size
            splt = self.cur_prompt.split(" ")
            subset = " ".join(splt[:self.block_size]) + " "

            # The rest is the current prompt
            self.cur_prompt = " ".join(splt[self.block_size:])

            # Get the oldest item in the past output and clean it
            oldest_item = self.past_output[0]
            oldest_item = oldest_item.replace("Girlfriend: ", "").replace("Me: ", "").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")

            # Store the subset and move all subsets
            # up in the queue
            self.past_output = self.past_output[1:] + [subset]




            # If the oldest item is not "", summarize it
            # Summarize the subset
            if oldest_item != "":
                # Summarize it as the current summary
                self.past_summ = self.summarizer(
                    self.past_summ + "\n\n" + oldest_item,
                    min_length=16,
                    max_length=512,
                    no_repeat_ngram_size=3,
                    repetition_penalty=5.0,
                    num_beams=4, # Note: Over 4 beams and the model kills my computer
                    early_stopping=True,
                )[0]["summary_text"]
        
        # When saving is done, save files to disk
        self.disc_json = dict(
            past_summ=self.past_summ,
            past_output=self.past_output,
            cur_prompt=self.cur_prompt,
        )

    def reset_memory(self):
        self.past_summ = deepcopy(self.initial_summary)
        self.past_output = ["" for i in range(self.num_blocks)]
        self.cur_prompt = deepcopy(self.initial_prompt)

    def chat(self):
        """
        The main chat function. It will return the response
        from the model.
        """
        # Get the input from the user
        user_input = input("Me: ")

        # Add it to the prompt
        self.cur_prompt += f"Me: {user_input}\n"

        # Get the response from the model
        response = self.get_response()

        # Print the response
        print(f"Girlfriend: {response}")

initial_summ = "The following is a conversation with me and my waifu girlfriend\n\n"
initial_prompt = (
    "Me: Hello\nGirlfriend: Hello\n"
    "Me: How are you?\nGirlfriend: I am good\n"
    "Me: I love you.\nGirlfriend: I love you too.\n"
    "Me: What should we do now?\n"
)
memory_file = None
MyLover = Lover(
    initial_prompt=initial_prompt,
    initial_summary=initial_summ,
    saved_memory=memory_file
)
MyLover.chat()