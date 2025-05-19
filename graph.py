import os
from typing import Literal

from pydantic import BaseModel, Field
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from config import BaseLLMSettings
from prompts import (
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_USER_PROMPT,
    ANSWER_CHECKING_SYSTEM_PROMPT,
    ANSWER_CHECKING_USER_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_USER_PROMPT,
)


class GraphState(dict):
    """State container for the LangGraph workflow."""
    question: str
    context: str
    answer: str
    iterations: int
    answer_validity: str
    history: list[tuple[str, str]] = []


class CheckingDecision(BaseModel):
    """Structured output model for answer checking."""
    decision: Literal['Y', 'N'] = Field(
        description="One character (Y or N), for decision if the answer is correct or not"
    )
    reasoning: str = Field(
        description='Reasoning behind the decision'
    )


class LangGraphClass:

    def __init__(self,
                 settings: BaseLLMSettings) -> None:
        self.settings = settings

        self._init_prompts()
        self._init_llm()
        self._init_embedding_function()
        self._init_vector_db()
        self._init_retriever()

        self.answer_checker_chain = self.check_answer_prompt | self.structured_llm_checking

    def _init_llm(self):
        """Set up the main and structured-checking LLMs, with fake models if testing."""

        if self.settings.TEST_TRUE:
            from langchain_core.messages import AIMessage
            from langchain_community.chat_models.fake import FakeMessagesListChatModel
            self.llm = FakeMessagesListChatModel(
                responses=[
                    AIMessage(content="Fake response",),
                ]
            )

            self.structured_llm_checking = FakeMessagesListChatModel(
                responses=[
                    AIMessage(
                        content="Fake checking response",
                        decision='N'
                    )
                ]
            )
        else:
            self.llm = ChatOpenAI(
                model_name=self.settings.LLM_MODEL_NAME,
                api_key=self.settings.OPENAI_API_KEY,
                temperature=0
            )
            self.structured_llm_checking = self.llm.with_structured_output(CheckingDecision)
        self.generator_chain = self.generator_prompt | self.llm
        self.reflection_chain = self.reflection_prompt | self.llm

    def _init_embedding_function(self):
        """Initialize OpenAI embedding model."""
        self.embedding_function = OpenAIEmbeddings(
            model=self.settings.EMBEDDING_MODEL,
            openai_api_key=self.settings.OPENAI_API_KEY
        )

    def _init_vector_db(self):
        """Load existing FAISS vector index"""
        if not os.path.exists(self.settings.INDEX_PATH):
            raise FileNotFoundError(f"FAISS index path does not exist: {self.settings.INDEX_PATH}")
        self.vector_db = FAISS.load_local(
            self.settings.INDEX_PATH,
            self.embedding_function,
            allow_dangerous_deserialization=True
        )

    def _init_retriever(self):
        """Create retriever object from FAISS index."""
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": self.settings.RETRIEVE_DOCS_NUMBER}
        )

    def _init_prompts(self):
        """Initialize all system/user prompts for generation, checking, and reflection."""
        self.check_answer_prompt = ChatPromptTemplate.from_messages([
            ('system', ANSWER_CHECKING_SYSTEM_PROMPT),
            ('user', ANSWER_CHECKING_USER_PROMPT),
        ])

        self.generator_prompt = ChatPromptTemplate.from_messages([
            ('system', GENERATOR_SYSTEM_PROMPT),
            ('user', GENERATOR_USER_PROMPT),
        ])
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            ('system', REFLECTION_SYSTEM_PROMPT),
            ('user', REFLECTION_USER_PROMPT),
        ])

    def create_vector_index(self, index_path: str, csv_path: str):
        """
        Build and save a new FAISS index from a CSV file.

        Args:
            index_path (str): Directory path to store the FAISS index.
            csv_path (str): Path to the CSV file containing question-answer data.
        """

        os.makedirs(index_path, exist_ok=True)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")

        df = pd.read_csv(csv_path)

        docs = [
            Document(page_content=f"Q: {row['question']}\nA: {row['answer']}")
            for _, row in df.iterrows()
        ]
        print("Number of documents:", len(docs))

        if not docs:
            raise ValueError("No documents found to embed")

        vector_db = FAISS.from_documents(docs, self.embedding_function)
        vector_db.save_local(index_path)
        print(f"Vector DB saved to: {index_path}")

    def retrieve_context(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents for the given question and update context."""
        state['iterations'] = 0
        docs = self.retriever.get_relevant_documents(state["question"])
        state["context"] = "\n\n".join([doc.page_content for doc in docs])
        return state

    def generate_answer(self, state: GraphState) -> GraphState:
        """Generate answer using the retrieved context and input question and history"""
        history = ""
        if state.get('history'):
            history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in state.get('history')])

        state["answer"] = self.generator_chain.invoke({
            "context": state["context"],
            "question": state["question"],
            "history": history,
        }).content
        return state

    def check_answer(self, state: GraphState) -> GraphState:
        """Validate the answer using an LLM structured output chain."""
        response = self.answer_checker_chain.invoke({
            "question": state["question"],
            "context": state["context"],
            "answer": state["answer"]
        })
        state["answer_validity"] = response.decision
        return state

    def decide_to_finish(self, state: GraphState) -> str:
        """
        Decide if the answer is valid or needs reflection (re-retrieval and regeneration)

        Returns:
            str: "end" to terminate or "reflect" to retry with improved answer.
        """
        if state["answer_validity"] == 'Y':
            return "end"
        if state["iterations"] > self.settings.MAX_ITERATIONS:
            return "end"

        return "reflect"

    def reflect_and_retry(self, state: GraphState) -> GraphState:
        """Improve the current answer by applying reflection logic."""
        state['iterations'] += 1

        improved_answer = self.reflection_chain.invoke({
            "question": state["question"],
            "context": state["context"],
            "answer": state["answer"]
        })
        state["answer"] = improved_answer.content
        return state

    def setup_workflow(self):
        """
        Define and compile the graph workflow.
        Returns:
            RunnableGraph: Compiled workflow graph for execution.
        """

        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve_context)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("check", self.check_answer)
        workflow.add_node("reflect", self.reflect_and_retry)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "check")

        workflow.add_conditional_edges(
            "check",
            self.decide_to_finish,
            {
                "end": END,
                "reflect": "reflect",
            }
        )
        workflow.add_edge("reflect", "check")

        return workflow.compile()
