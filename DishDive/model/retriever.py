class Retriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve_k(self, k=3):
        return self.vectorstore.as_retriever(k)
