from config import get_settings
from pinecone import Pinecone

s = get_settings()

pc = Pinecone(api_key=s.pinecone_api_key)

print("Indexes:")
print(pc.list_indexes())

index = pc.Index(s.pinecone_index_name)
print("Pinecone index connected:", s.pinecone_index_name)
print(index.describe_index_stats())