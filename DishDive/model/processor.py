import os
import transformers
from dotenv import load_dotenv

load_dotenv()

hf_auth = os.getenv("HF_AUTH")
model = os.getenv("MODEL")


class Tokenizer:
    def __init__(self, model_id=model):
        self.model_id = model_id

    def get_tokenizer(self):
        return transformers.AutoTokenizer.from_pretrained(
            self.model_id, use_auth_token=hf_auth
        )
