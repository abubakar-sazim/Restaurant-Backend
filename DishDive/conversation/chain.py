from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
from DishDive.instructor.prompt import Prompt
from langchain.schema import format_document

prompt = Prompt()


def _combine_documents(
    docs, document_prompt=prompt.document_prompt(), document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


class LLMChain:
    def __init__(
        self,
        CONDENSE_QUESTION_PROMPT,
        ANSWER_PROMPT,
        retriever,
        standalone_query_generation_llm,
        response_generation_llm,
    ):
        self.CONDENSE_QUESTION_PROMPT = CONDENSE_QUESTION_PROMPT
        self.ANSWER_PROMPT = ANSWER_PROMPT
        self.retriever = retriever
        self.standalone_query_generation_llm = standalone_query_generation_llm
        self.response_generation_llm = response_generation_llm

    def get_chain(self):
        memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )

        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables)
            | itemgetter("history"),
        )

        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | self.CONDENSE_QUESTION_PROMPT
            | self.standalone_query_generation_llm,
        }

        retrieved_documents = {
            "docs": itemgetter("standalone_question") | self.retriever,
            "question": lambda x: x["standalone_question"],
        }

        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"]),
            "question": itemgetter("question"),
        }

        answer = {
            "answer": final_inputs | self.ANSWER_PROMPT | self.response_generation_llm,
            "question": itemgetter("question"),
            "context": final_inputs["context"],
        }

        final_chain = loaded_memory | standalone_question | retrieved_documents | answer

        return memory, final_chain
