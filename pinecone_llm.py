import pinecone
import os
import uvicorn
import json

from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse



os.environ["OPENAI_API_KEY"] = "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"

class Query(BaseModel):
    dataset: str
    prompt: str

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize pinecone
api_key = "7f751400-6170-468b-9be9-23a21b63f3e6"
pinecone.init(api_key=api_key, environment="us-west1-gcp")

# load index
def load_index(dataset: str):
    index = pinecone.Index("demo")
    vector_store = PineconeVectorStore(pinecone_index=index, namespace=dataset)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

# query_engine = loaded_index.as_query_engine()
# response = query_engine.query("How do I adjust my temperature?")
# print(response)

@app.get("/")
async def root():
    return {"message": "Hello World. I am a customer support AI assistant."}

@app.post("/get")
def read_root(query: Query):
    prompt = f'You are an AI assistant that returns response in prettified html format to any QUERY. Highlight links, use numbered ordering, style the text in bold, italics, or underlined where needed to make it readable. Here is the QUERY: {query.prompt}'
    loaded_index = load_index(query.dataset)
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query(prompt)
    return {"prompt": query.prompt, "response": response}

async def stream_response(query: Query):
    links = []
    prompt = f'You are an AI assistant that returns response in prettified html format to any QUERY. Highlight links, use numbered ordering, style the text in bold, italics, or underlined where needed to make it readable. Here is the QUERY: {query.prompt}'
    loaded_index = load_index(query.dataset)
    query_engine = loaded_index.as_query_engine(streaming=True)
    response_stream = query_engine.query(prompt)
    for source in response_stream.source_nodes:
        if 'article-link' in source.node.extra_info:
            links.append(source.node.extra_info['article-link'])
    yield(json.dumps(links))
    for text in response_stream.response_gen:
        yield(text)
    

@app.post("/getstream")
async def getstream(query: Query):
    return StreamingResponse(stream_response(query))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)