import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # noqa
import pytest
from langchain_core.documents import Document
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_community.vectorstores import FAISS
from graph import LangGraphClass
from config import BaseLLMSettings


class FakeGraphClass(LangGraphClass):

    def _init_vector_db(self):
        doc = Document(page_content="fake document") # Hack to init the FAISS vectorstore
        embedding_function = DeterministicFakeEmbedding(size=8)
        self.vector_db = FAISS.from_documents(
            documents=[doc],
            embedding=embedding_function,
            normalize_L2=True
        )
        return self.vector_db


@pytest.fixture
def fake_graph():
    settings = BaseLLMSettings(TEST_TRUE=True)
    return FakeGraphClass(settings)
