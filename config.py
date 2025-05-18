from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseLLMSettings(BaseSettings):

    TEST_TRUE: bool = False
    MAX_ITERATIONS: int = 1
    RETRIEVE_DOCS_NUMBER: int = 4
    INDEX_PATH: str = 'faiss_index'
    CSV_DATA_PATH: str = "data/rag_dataset.csv"
    LLM_MODEL_NAME: str = 'gpt-4o'
    MINI_LLM_MODEL: str = 'gpt-4o-mini'
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    OPENAI_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )


settings = BaseLLMSettings()
