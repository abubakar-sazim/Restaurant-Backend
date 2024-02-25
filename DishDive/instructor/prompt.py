from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

_template = """
[INST]

Given the user's request for restaurant suggestions based on reviews, create a standalone question.
This query will retrieve relevant documents containing restaurant reviews and specific attributes.

The database has attributes like location, Alcohol(if the restauant has alcohol or not),
wifi(has free wifi or not), Accepts Credit Cards(rue/false),  Good for Kids(yes/no),
Has TV(true/false), Noise Level, Outdoor Seating(true/false), Parking(true/false), 
Delivery (true/false), Good for Groups (true/false).
The standalone question must be generated using these features whenever needed.

The question should be one line question.
Don't use "based on the context" in the question and no additional text. 

Let me share a couple examples that will be important.

If there's no prior interaction, return the "Follow Up Input" as is.

here is the actual chat history and input question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:
[your response here]
[/INST]
"""

template = """
[INST]
You are a helpful agent that gives restaurants suggestion based on context. Don't say "Based on the provided context" when answering.

Provide an answer based solely on the provided context:
{context}

If there is no documents in the context then say "Sorry I don't have enough information to suggest you a restaurants."
Do Not give any suggestion out of the context.

Format the answer like this:
Name of the restaurant:
Address:
Reason:

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

    def document_prompt(self):
        return PromptTemplate.from_template(template="{page_content}")
