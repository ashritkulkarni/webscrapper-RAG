import pinecone

import os
import uvicorn
import json

import openai

from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from typing import Callable, Dict, Generator, List, Optional, Type



os.environ["OPENAI_API_KEY"] = ""
# get api key from platform.openai.com
openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

embed_model = "text-embedding-ada-002"


# initialize pinecone
api_key = ""
pinecone.init(api_key=api_key, environment="us-west1-gcp")




index_name = "ash-poc"
# index = pinecone.Index("ash-poc")
index = pinecone.GRPCIndex(index_name)
# view index stats
print(index.describe_index_stats())

query = "What is the step 4 in the installation of Lifestyle Collection Furniture HD?"

res = openai.Embedding.create(
    input=[query],
    engine=embed_model
)
res

from langchain.embeddings.openai import OpenAIEmbeddings

# get openai api key from platform.openai.com
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)
from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)
# retrieve from Pinecone
xq = res['data'][0]['embedding']

# index = pinecone.Index("ash-poc")
# # index = pinecone.Index("ash-poc")
# vector_store = PineconeVectorStore(pinecone_index=index, namespace='ash-poc')
# final_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# # get relevant contexts (including the questions)
res = index.query(xq, top_k=5, include_metadata=True)

# get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]


print(vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
))

print(query)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print(qa_with_sources(query))

# system message to 'prime' the model
primer = f"""You are Q&A bot. A highly intelligent system that answers
user questions based on the information provided by the user above
each question. Only answer from pdf.txt files that you have access to from the vector metadata.
If the information can not be found in the information
provided by the user you truthfully say "I don't know".
"""

res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": query}
    ]
)

from IPython.display import Markdown, display

print(res['choices'][0]['message']['content'])

links = []
prompt_res = "How to install air chamber"
prompt = f'You are an AI assistant that returns response in prettified html format to any QUERY. Only answer from pdf.txt files that you have access to from the vector metadata. Highlight links, use numbered ordering, style the text in bold, italics, or underlined where needed to make it readable. Here is the QUERY: {prompt_res}'
loaded_index = load_index("ash-hf-v1")
query_engine = loaded_index.as_query_engine()
response = query_engine.query(prompt)
for source in response.source_nodes:
    if 'file-name' in source.node.extra_info:
        links.append(source.node.extra_info['file-name'])
print(links)

print(response)

