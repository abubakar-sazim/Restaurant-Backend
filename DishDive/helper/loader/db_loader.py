from torch import cuda
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


class LoadDB:
    def __init__(self, path, embed_model_id="sentence-transformers/all-mpnet-base-v2"):
        self.path = path
        self.embed_model_id = embed_model_id

    def load(self):
        device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
        embed_model = HuggingFaceEmbeddings(
            model_name=self.embed_model_id,
            model_kwargs={"device": device},
            encode_kwargs={"device": device, "batch_size": 32},
        )
        vectorstore = Chroma(
            persist_directory=self.path, embedding_function=embed_model
        )
        return vectorstore
