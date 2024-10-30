import pinecone
import argparse
import json

from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Type

from llama_index import Document
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import PineconeVectorStore

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--company_name", help="Company name")

args = argParser.parse_args()

# Creating a Pinecone index
api_key = ""
pinecone.init(api_key=api_key, environment="us-west1-gcp")

# pinecone.create_index(
#     "demo",
#     dimension=1536,
#     metric="euclidean",
#     pod_type="p1"
# )
index = pinecone.Index("demo")

# construct vector store
vector_store = PineconeVectorStore(pinecone_index=index, namespace=f'{args.company_name.replace("_", "-")}')

# create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

input_dir = f'{args.company_name}_data'
input_dir_path = Path(input_dir)

# Iterate over this subtree and yield all existin
#  files (of any kind, including directories) matching the given relative pattern.
file_refs = Path(input_dir).glob("*")

all_files = set()
for ref in file_refs:
    if not ref.is_dir():
        all_files.add(ref)

#print(all_files)

with open(f"scrapers/{args.company_name}/{args.company_name}_file_link_map.txt", "r") as fp:
    # Load the dictionary from the file
    filename_to_link_map = json.load(fp)
#print(filename_to_link_map)

documents = []
for input_file in all_files:
    metadata: Optional[dict] = None
    filename = str(input_file)
    if filename in filename_to_link_map:
        metadata = {'article-link': filename_to_link_map[filename]}
    #print(filename_to_link_map[str(input_file).replace('.txt', '')])
    with open(input_file, "r", errors="ignore", encoding="utf-8") as f:
        data = f.read()

    doc = Document(text=data, metadata=metadata or {})
    doc.id_ = str(input_file)
    #print(doc.metadata)
    documents.append(doc)


# create index, which will insert documents/vectors to pinecone
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
# index = VectorStoreIndex.from_documents(documents)
# print(index.ref_doc_info)