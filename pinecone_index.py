import pinecone
import argparse

from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import PineconeVectorStore

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--company_name", help="Company name")

args = argParser.parse_args()
print(args)

# Creating a Pinecone index
api_key = ""
pinecone.init(api_key=api_key, environment="us-west1-gcp")

pinecone.create_index(
    f'{args.company_name.replace("_", "-")}',
    dimension=1536,
    metric="euclidean",
    pod_type="p1"
)
index = pinecone.Index(f'{args.company_name.replace("_", "-")}')

# construct vector store
vector_store = PineconeVectorStore(pinecone_index=index)

# create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# A document is created for each page in a pdf or a text file
documents = SimpleDirectoryReader(f'{args.company_name}_data', filename_as_id=True).load_data()

# create index, which will insert documents/vectors to pinecone
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)