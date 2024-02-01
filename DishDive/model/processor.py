import os
import transformers
from dotenv import load_dotenv

load_dotenv()

hf_auth = os.getenv("HF_AUTH")


class Tokenizer:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_id = model_id

    def tokenizer(self):
        return transformers.AutoTokenizer.from_pretrained(
            self.model_id, use_auth_token=hf_auth
        )
