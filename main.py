from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from DishDive.helper.loader.db_loader import LoadDB
from DishDive.helper.loader.model_loader import LoadModel
from DishDive.model.processor import Tokenizer
from DishDive.query_pipeline.generate import Generator
from DishDive.instructor.prompt import Prompt
from DishDive.conversation.chain import LLMChain
from DishDive.conversation.chat import Chat
import re

app = FastAPI()
prompts = Prompt()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


db_loader = LoadDB("./chroma_db")
vectorstore = db_loader.load()

model_loader = LoadModel()
model = model_loader.load()

tokenizer = Tokenizer()
tokenizer = tokenizer.get_tokenizer()

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 3},
)
generator = Generator(model, tokenizer)
standalone_query_generation_llm = generator.standalone_query()
response_generation_llm = generator.response()

CONDENSE_QUESTION_PROMPT = prompts.question_promt()
ANSWER_PROMPT = prompts.response_prompt()
DEFAULT_DOCUMENT_PROMPT = prompts.document_prompt()

llm_chain = LLMChain(
    CONDENSE_QUESTION_PROMPT,
    ANSWER_PROMPT,
    retriever,
    standalone_query_generation_llm,
    response_generation_llm,
)
memory, final_chain = llm_chain.get_chain()


def process_response(entries):
    context_list = []
    for entry in entries:
        business_dict = {}
        lines = entry.split("\n")
        current_key = None
        current_value = []
        for line in lines:
            if ": " in line:
                if current_key is not None:
                    business_dict[current_key] = "\n".join(current_value).strip()
                current_key, current_value = line.split(": ", 1)
                current_value = [current_value]
            else:
                current_value.append(line)

        if current_key is not None:
            business_dict[current_key] = "\n".join(current_value).strip()

        context_list.append(business_dict)
    context = {
        f"id_{str(i)}": business_dict for i, business_dict in enumerate(context_list)
    }

    return context


@app.get("/")
async def read_root():
    return {"Hello": "Foodies"}


@app.get("/chat")
async def inference(question: str):
    try:
        chatbot = Chat(question, final_chain, memory)
        response = chatbot.ask_LLM()

        entries = re.split(r"\n(?=business_id:)", response["context"].strip())

        result = {
            "question": response["question"],
            "answer": response["answer"],
            "context": process_response(entries),
        }

        return JSONResponse(content=result)
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
