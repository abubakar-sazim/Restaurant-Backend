from langchain_community.document_loaders.csv_loader import CSVLoader


class LoadData:
    def __init__(self, path):
        self.path = path

    def load(self):
        loader = CSVLoader(file_path=self.path, encoding="utf8")
        return loader.load()
