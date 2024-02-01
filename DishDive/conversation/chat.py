class Chat:
    def __init__(self, question, chain, memory):
        self.question = question
        self.chain = chain
        self.memory = memory

    def ask_LLM(self):
        inputs = {"question": self.question}
        result = self.chain.invoke(inputs)
        self.memory.save_context(inputs, {"answer": result["answer"]})
        return result
