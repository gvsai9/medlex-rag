"""
Central config — reads from .env
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # =========================
    # Embeddings
    # =========================
    embedding_provider: str = "local"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # =========================
    # LLM
    # =========================
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_temperature: float = 0.1
    ollama_smalltalk_temperature: float = 0.3
    ollama_top_p: float = 0.9
    ollama_num_ctx: int = 4096
        # Gemini / Cloud LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Common LLM generation settings
    llm_temperature: float = 0.1
    llm_smalltalk_temperature: float = 0.3
    llm_top_p: float = 0.9
    llm_max_tokens: int = 1200
    # Graph extraction LLM
    graph_llm_provider: str = "ollama"
    graph_llm_model: str = "qwen2.5:3b"

    # =========================
    # Pinecone
    # =========================
    vector_db_provider: str = "pinecone"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "medlex-rag-bge-final"
    pinecone_host: str = ""
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "medlex"

    # =========================
    # Neo4j Aura
    # =========================
    graph_db_provider: str = "neo4j"
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""
    neo4j_database: str = ""
    aura_instanceid: str = ""
    aura_instancename: str = ""

    # =========================
    # MySQL state DB
    # =========================
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "medlex"

    # =========================
    # App
    # =========================
    app_env: str = "development"
    max_concurrent_workers: int = 4
    chunk_size: int = 512
    chunk_overlap: int = 51
    top_k_retrieval: int = 5
    top_k_rerank: int = 2

    # Optional old fields
    sqlite_db_path: str = "data/medlex.db"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "models/text-embedding-004"
    faiss_index_path: str = "data/index/faiss.index"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "medlex-rag"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()