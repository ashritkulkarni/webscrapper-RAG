import pinecone
import argparse
import json
import os
import google.auth
from google.cloud import storage

from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Type

from llama_index import Document
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import PineconeVectorStore

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--company_name", help="Company name")

args = argParser.parse_args()

company_name = f'{args.company_name}'

# Creating a Pinecone index
api_key = "7f751400-6170-468b-9be9-23a21b63f3e6"
pinecone.init(api_key=api_key, environment="us-west1-gcp")

#Connection to GCP
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'config/rejoy_gcp_secret.json'
# Create a service account key file.
key_file_path = "/config/rejoy_gcp_secret.json"
# Create a Google Cloud Storage client object.
client = storage.Client()

# Calling the index
index = pinecone.Index("sleepnum-poc")

# construct vector store
vector_store = PineconeVectorStore(pinecone_index=index, namespace=f'{company_name.replace("_", "-")}')

# create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Fetching files from GCP

# Get the bucket name.
bucket_name = "sleepnumber_text_files"
# Get a list of all the objects in the bucket.
objects = client.bucket(bucket_name).list_blobs()
# Iterate over the objects and print the name of each text file.

documents = []
count = 0
for object in objects:
    # print(f'count: {count}')
    count = count + 1
    metadata: Optional[dict] = None
    if object.name == "Total_Encasement_Instruction_Sheet_360.pdf.txt" or object.name == "360_Bases.pdf.txt" or object.name == "SCS_Replacement_Work_Instruction.pdf.txt":
        metadata = {'file-name': object.name}
        data = object.open("rb").read().decode("utf-8")
        doc = Document(text=data, metadata=metadata or {})
        doc.id_ = str(object)
        print(doc.metadata)
        documents.append(doc)
print(documents)
print(count)
print(len(documents))

print('indexing into vector DB')

# # create index, which will insert documents/vectors to pinecone
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
# index = VectorStoreIndex.from_documents(documents)
# # print(index.ref_doc_info)