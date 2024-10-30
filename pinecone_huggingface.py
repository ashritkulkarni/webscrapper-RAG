from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from google.cloud import storage
import pinecone
import os
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai
from typing import Callable, Dict, Generator, List, Optional, Type

hf_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
OPENAI_API_KEY = openai.api_key = "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"

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
index = pinecone.Index("sleepnum-hf")
index_name = "sleepnum-hf"


client = storage.Client()
# Get the bucket name.
bucket_name = "sleepnumber_text_files"
# # Get a list of all the objects in the bucket.
objects = client.bucket(bucket_name).list_blobs()

documents = []
count = 0
for object in objects:
    count = count + 1
    metadata: Optional[dict] = None
    if object.content_type == "text/plain" and ".pdf" in object.name:
        metadata = {'file-name': object.name}
        data = object.open("rb").read().decode("utf-8")
        doc = Document(page_content=data, metadata=metadata or {})
        documents.append(doc)
# print(documents[0])


print('------------------indexing into vector DB----------------')

# text_splitter = CharacterTextSplitter(
#         chunk_size=1000,      # Specify chunk size
#         chunk_overlap=200,    # Specify chunk overlap to prevent loss of information
#     )

# docs_split = text_splitter.split_documents(documents)

# # hf_docsearch = Pinecone.from_texts([
# #     t['text'] for t in documents], 
# #     hf_embeddings, 
# #     index_name = index_name,
# #     namespace = 'ash-sleepnum-hf-v1')

# # from langchain.document_loaders import GCSDirectoryLoader


# # bucket_name = "sleepnumber_text_files"

# # # Create a GCSDirectoryLoader object, specifying the GCS bucket name and the prefix of the documents to load
# # loader = GCSDirectoryLoader(bucket = bucket_name, project_name='rejoycs')

# # # Call the `load()` method on the loader object to load the documents
# # documents_lang = loader.load()
# # docs = []

# print("-------------------------This is document loader first doc----------------------------------")
# print(documents[0])

# text_splitter = CharacterTextSplitter(
#         chunk_size=1000,      # Specify chunk size
#         chunk_overlap=200,    # Specify chunk overlap to prevent loss of information
#     )

# docs_split = text_splitter.split_documents(documents)

# # create new embedding to upsert in vector store
# doc_db = Pinecone.from_documents(
#           docs_split,
#           hf_embeddings,
#           index_name=index_name,
#           namespace='ash-sleepnum-hf-v1'
#         )

# query = "How to install air chamber"

# # search for matched entities and return score
# search_docs = doc_db.similarity_search_with_score(query)

# print("------------------------This is similary search-------------------------------")
# print(search_docs)




# # repo_id = "uni-tianyan/Uni-TianYan" 
# # llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={
# #                                   "temperature":0, 
# #                                   "max_length":64})

# # completion llm
# llm = ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model_name='gpt-3.5-turbo-16k',
#     temperature=0.0
# )

# template = """Question: {query}

# Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["query"])

# llm_chain = LLMChain(prompt=prompt, llm=llm)



# qa = RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type='stuff',
#     retriever=doc_db.as_retriever(),
# )
# query = "How to install air chamber"
# result = qa.run(query)

# print("-----------------------------This is actual LLM output------------------------")

# print(result)

# print("-----------------------------This is LLM output with prompt ------------------------")

# print(llm_chain.run(query))