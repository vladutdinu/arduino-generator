from typing import List
from pydantic import BaseModel

class Chat(BaseModel):
    query: str
    collection_name: str = "arduino"

class IngestUrls(BaseModel):
    urls: List[str]

class Metadata(BaseModel):
    source: str

class Sources(BaseModel):
    metadata: Metadata

class ArduinoExperiment(BaseModel):
    Description: str
    Code: str
    Explanation: str

class LLMResponse(BaseModel):
    query: str
    result: ArduinoExperiment | str
    source_documents: List[Sources] = []
