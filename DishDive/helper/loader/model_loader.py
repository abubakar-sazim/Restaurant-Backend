import os
from torch import cuda, bfloat16
import transformers
from dotenv import load_dotenv

load_dotenv()

hf_auth = os.getenv("HF_AUTH")


class LoadModel:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_id = model_id

    def load(self):
        device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )

        model_config = transformers.AutoConfig.from_pretrained(
            self.model_id, use_auth_token=hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
            use_auth_token=hf_auth,
        )
        model.eval()
        print(f"Model loaded on {device}")
