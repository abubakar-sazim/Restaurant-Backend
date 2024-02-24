from transformers import pipeline
from langchain.llms import HuggingFacePipeline


class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def standalone_query(self):
        standalone_query_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample = False,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )
        return HuggingFacePipeline(pipeline=standalone_query_generation_pipeline)

    def response(self):
        response_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample = True,
            temperature=0.25,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )
        return HuggingFacePipeline(pipeline=response_generation_pipeline)
