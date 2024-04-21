from typing import List
from langchain_community.vectorstores import Chroma
import chromadb
import ollama
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from tokenizers import Tokenizer
import fitz
import uuid
from cleantext import clean

def clean_text(chunk):
    return clean(chunk)

def build_prompt(system_prompt, request):
        return """
           {}
        
           {}  

           Keep in mind that you have to give the output in JSON format.
        """.format(
            system_prompt,
            request
        )


class Embed(Embeddings):
    def __init__(self, func, model):
        self.func = func
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(texts)
    
    def embed_query(self, prompt):
        return self.func(self.model, prompt)["embedding"]
        

class OllamaClient:
    def __init__(self, host, model):
        self.ollama_client = ollama.Client(host)
        self.model = model  
        self.embed_service = Embed(self.ollama_client.embeddings, self.model)

    def generate(self, text, context, format='', stream=False):
        return self.ollama_client.generate(model=self.model, prompt=text, context=context, format=format, stream=stream)

# class ChromaClient:
#     def __init__(self, host, port):
#         self.chroma_client = chromadb.HttpClient(host=host + ":" + str(port))

class ChromaClient:
    def __init__(self, host, port):
        self.chroma_client = chromadb.PersistentClient(path="./chromadb")


class Chunker:
    def __init__(self, tokenizer_path, max_tokens):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size = int(max_tokens), chunk_overlap = 100)

    def chunk_it(self, text):
        chunks = self.splitter.split_text(text)
        return chunks

    def ingest(
        self,
        text,
        filename,
        file_type,
        ollama_client,
        chroma_client,
        chroma_collection_name,
    ):
        chunks = None
        collection = chroma_client.get_or_create_collection(chroma_collection_name)
        if file_type == "PDF":
            pdf = fitz.Document(stream=text, filetype="pdf")
            data = ""
            for page in range(0, pdf.page_count):
                data = data + pdf[page].get_text()
            chunks = self.chunk_it(data)
            for _, chunk in enumerate(chunks):
                id = uuid.uuid4().hex
                embed = ollama_client.embed_service.embed_query(
                    clean_text(chunk)
                )
                collection.add(
                    [id], [embed], documents=[chunk], metadatas={"source": filename}
                )
        elif file_type == "URL":
            chunks = self.chunk_it(text)
            for _, chunk in enumerate(chunks):
                id = uuid.uuid4().hex
                embed = ollama_client.embed_service.embed_query(
                   clean_text(chunk)
                )
                collection.add(
                    [id], [embed], documents=[chunk], metadatas={"source": filename}
                )        


class RAGPipeline:
    def __init__(
        self,
        ollama_url,
        ollama_model,
        system_prompt
    ):
        self.ollama_llm = Ollama(base_url=ollama_url, model=ollama_model)
        self.system_prompt = system_prompt

    def invoke(self, ollama_client, chroma_client, chroma_collection_name, prompt):
        db_retriever = Chroma(
            client=chroma_client,
            collection_name=chroma_collection_name,
            embedding_function=ollama_client.embed_service,
        )
        return RetrievalQA.from_chain_type(
                self.ollama_llm,
                retriever=db_retriever.as_retriever(search_kwargs={"k": 10}),
                return_source_documents=True,
            ).invoke({"query": build_prompt(self.system_prompt, prompt)})
