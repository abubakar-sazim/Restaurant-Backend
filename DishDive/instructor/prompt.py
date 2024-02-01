from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

_template = """
[INST]

Given the user's request for restaurant suggestions based on reviews and possible attributes, create a standalone question in the original language.
This query will retrieve relevant documents containing restaurant reviews and specific attributes.

Let me share a couple examples that will be important.

If there's no prior interaction, return the "Follow Up Input" as is.

Now, with those examples, here is the actual chat history and input question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:
[your response here]
[/INST]
"""

template = """
[INST]
You are a helpful agent that gives restaurants suggestion based on context.

Provide an answer based solely on the provided context:
{context}
Don't say "Based on the provided context". Answer naturally.

Format the answer like this:
Name of the restaurant
Address
Why do you thik this is the answer.

Do Not give any false suggestion. If there is any questions that can't be answer from contex just tell
"Sorry I don't have enough information to suggest you a restaurants."

If user asks questions other than restaurants suggestion you tell them to ask only about restaurants suggestion.

If user say "hi", "Hello"or other casual questions respond accordingly.

Question: {question}
[/INST]
"""


class Prompt:
    def __init__(self):
        pass

    def question_promt(self):
        return PromptTemplate.from_template(_template)

    def response_prompt(self):
        return ChatPromptTemplate.from_template(template)
    
    def combine_document_prompt(self):
        return PromptTemplate.from_template(template="{page_content}")
