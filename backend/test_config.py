from config import get_settings
import config as config_module

print("Loaded config file from:", config_module.__file__)

s = get_settings()

print("Embedding provider:", s.embedding_provider)
print("Embedding model:", s.embedding_model_name)
print("Embedding dim:", s.embedding_dim)

print("LLM provider:", s.llm_provider)
print("Ollama base URL:", s.ollama_base_url)
print("Ollama model:", s.ollama_model)

print("Vector DB:", s.vector_db_provider)
print("Pinecone index:", s.pinecone_index_name)
print("Pinecone host:", s.pinecone_host)
print("Pinecone namespace:", s.pinecone_namespace)

print("Graph DB:", s.graph_db_provider)
print("Neo4j URI:", s.neo4j_uri)
print("Neo4j username:", s.neo4j_username)
print("Neo4j database:", s.neo4j_database)
print("Aura instance:", s.aura_instancename)

print("MySQL host:", s.mysql_host)
print("MySQL database:", s.mysql_database)