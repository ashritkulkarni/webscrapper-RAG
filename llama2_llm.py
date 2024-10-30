import pinecone
from torch import cuda, bfloat16
import transformers
import argparse
import json
import os
import timeit
import time
from google.cloud import storage
import openai
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Type


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from tqdm.auto import tqdm
from uuid import uuid4
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA                                                           
from typing import Callable, Dict, Generator, List, Optional, Type
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

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

# # Calling the index
# index = pinecone.Index("sleepnum-hf")
# index_name = "sleepnum-hf"

index = pinecone.Index('sleepnum-poc')
index_name = 'sleepnum-poc'

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)


text_field = "text"

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

model_id = 'meta-llama/Llama-2-13b-chat-hf'
# model_id = 'meta-llama/Llama-2-7b-chat-hf'
hf_auth = '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'



# device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# # set quantization configuration to load large model with less GPU memory
# # this requires the `bitsandbytes` library
# # bnb_config = transformers.BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_quant_type='nf4',
# #     bnb_4bit_use_double_quant=True,
# #     bnb_4bit_compute_dtype=bfloat16
# # )

# bnb_config = transformers.BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

# # begin initializing HF items, need auth token for these


# model_config = transformers.AutoConfig.from_pretrained(
#     model_id,
#     use_auth_token=hf_auth
# )

# model = transformers.AutoModelForCausalLM.from_pretrained(
#     model_id,
#     trust_remote_code=True,
#     config=model_config,
#     quantization_config=bnb_config,
#     device_map='auto',
#     use_auth_token=hf_auth
# )
# model.eval()
# print(f"Model loaded on {device}")

from langchain.llms import CTransformers

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin', # Location of downloaded GGML model
                    model_type='llama', # Model type Llama
                    config={'max_new_tokens': 1200,
                            'temperature': 0.01})

# from langchain import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS

qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

# # Wrap prompt template in a PromptTemplate object
def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['question'])
    return prompt

query = 'How to install air chamber?'

similarity_docs = vectorstore.similarity_search(
    query,  # the search query
    k=2  # returns top 3 most relevant chunks of text
)

print("---------------------------------------This is similarity results-----------------------------------")
print(similarity_docs)

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
start = time.time()
print("----------------------------------------This is LLM response-----------------------------------------")

print(rag_pipeline(query))
end = time.time()
print("LLAMA LLM took this much time to respond :",
      (end-start), "mins")

# # Build RetrievalQA object
# def build_retrieval_qa(llm, prompt, vectordb):
#     dbqa = RetrievalQA.from_chain_type(llm=llm,
#                                        chain_type='stuff',
#                                        retriever=vectordb.as_retriever(search_kwargs={'k':2}),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={'prompt': prompt})
#     return dbqa


# # Instantiate QA object
# def setup_dbqa():
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
#                                        model_kwargs={'device': 'cpu'})
#     vectorstore = Pinecone(
#     index, embeddings.embed_query, text_field, namespace = 'ash-sleepnum-hf-v1'
# )
#     qa_prompt = set_qa_prompt()
#     dbqa = build_retrieval_qa(llm, qa_prompt, vectorstore)

#     return dbqa

# parser = argparse.ArgumentParser()
# parser.add_argument('input', type=str)
# args = parser.parse_args()
# start = timeit.default_timer() # Start timer

# # Setup QA object
# dbqa = setup_dbqa()

# # Parse input from argparse into QA object
# response = dbqa({'query': args.input})
# end = timeit.default_timer() # End timer

# # Print document QA response
# print(f'\nAnswer: {response["result"]}')
# print('='*50) # Formatting separator

# # Process source documents for better display
# source_docs = response['source_documents']
# for i, doc in enumerate(source_docs):
#     print(f'\nSource Document {i+1}\n')
#     print(f'Source Text: {doc.page_content}')
#     print(f'Document Name: {doc.metadata["source"]}')
#     print(f'Page Number: {doc.metadata["page"]}\n')
#     print('='* 50) # Formatting separator
    
# # Display time taken for CPU inference
# print(f"Time to retrieve response: {end - start}")