from contextlib import asynccontextmanager
from http.client import HTTPException
import json
from fastapi import FastAPI, UploadFile
from langchain_community.document_loaders import SeleniumURLLoader
from fastapi.middleware.cors import CORSMiddleware
from utils import OllamaClient, ChromaClient, Chunker, RAGPipeline, build_prompt, clean_text
from schemas import ArduinoExperiment, IngestUrls, LLMResponse, Chat, Metadata, Sources
from dotenv import load_dotenv

import os

load_dotenv()

rag_builder = None
chroma_client = None
ingester = None
ollama_client = None
rag_builder = None
context = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, chroma_client, ingester, ollama_client, rag_builder
    ollama_client = OllamaClient(os.getenv("OLLAMA_URL"), os.getenv("OLLAMA_MODEL"))
    chroma_client = ChromaClient(os.getenv("CHROMADB_URL"), os.getenv("CHROMADB_PORT")).chroma_client
    rag_builder = RAGPipeline(
        os.getenv("OLLAMA_URL"),
        os.getenv("OLLAMA_MODEL"),
        os.getenv("SYSTEM_PROMPT")
    )
    ingester = Chunker(os.getenv("TOKENIZER_PATH"), os.getenv("TOKENIZER_CHUNK_SIZE"))
    yield
    del chroma_client, rag_builder, ollama_client, ingester


app = FastAPI(lifespan=lifespan)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(query: Chat) -> LLMResponse:
    global context
    res = ollama_client.generate(query.query, context)
    if context == []:
        context = res["context"]
    llm_response = LLMResponse(query=query.query, result=res["response"])
    return llm_response


@app.post("/generate_rag")
async def generate_rag(query: Chat) -> LLMResponse:
    res = rag_builder.invoke(ollama_client, chroma_client, query.collection_name, query.query)
    print(res["result"])
    return LLMResponse(
        query=res["query"],
        result=ArduinoExperiment(**json.loads(res["result"])),
        source_documents=[
            Sources(
                metadata=Metadata(
                    source=src.metadata["source"]
                ),
            )
            for src in res["source_documents"]
        ],
    )

@app.post("/generate_context")
async def generate_context(query: Chat) -> LLMResponse:
    data_context = chroma_client.get_collection(query.collection_name).query(ollama_client.embed_service.embed_query(query.query), include=["metadatas", "documents"], n_results=1)
    res = ollama_client.generate(
        build_prompt(os.getenv("SYSTEM_PROMPT"), query.query), 
        context = ingester.tokenizer.encode(" ".join([doc for doc in data_context["documents"][0]])).ids)
    print(res["response"])
    return LLMResponse(
        query=build_prompt(os.getenv("SYSTEM_PROMPT"), query.query),
        result=ArduinoExperiment(**json.loads(res["response"])),
        source_documents=[
            Sources(
                metadata=Metadata(
                    source=src["source"]
                ),
            )
            for src in data_context["metadatas"][0]
        ],
    )

@app.post("/ingest_documents")
async def ingest_documents(file: UploadFile, collection_name: str):
    data = await file.read()
    file_type = None
    if file.filename.split(".")[-1] == "pdf" or file.filename.split(".")[-1] == "PDF":
        file_type = "PDF"
    else:
        raise HTTPException(
            status_code=406, detail="We do not support other file types other than PDFs"
        )
    try:
        ingester.ingest(
            data,
            file.filename,
            file_type,
            ollama_client,
            chroma_client,
            collection_name,
        )
    except:
        raise HTTPException(status_code=500, detail="Problem")
    return {"message": "ok"}


@app.post("/ingest_urls")
async def ingest_urls(urls_to_ingest: IngestUrls, collection_name: str):
    for url in urls_to_ingest.urls:
        file_type = "URL"
        try:
            ingester.ingest(
                SeleniumURLLoader(urls=[url]).load()[0].page_content,
                url,
                file_type,
                ollama_client,
                chroma_client,
                collection_name
            )
        except:
            raise HTTPException(status_code=500, detail="Problem")
        return {"message": "ok"}


if __name__ == "__main__":
    import uvicorn, os

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=os.getenv("PORT", "8000"))
