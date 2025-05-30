from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import dotenv

dotenv.load_dotenv()

class LLMEmbedding:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GEMINI_API_KEY")

    def embed_text_openai(self, text: str) -> list[float]:
        return OpenAIEmbeddings(model=self.model_name, api_key=self.openai_api_key).embed_query(text)
    
    def embed_batch_text_openai(self, texts: list[str]) -> list[list[float]]:
        return OpenAIEmbeddings(model=self.model_name, api_key=self.openai_api_key).embed_documents(texts)
    
    def embed_text(self, text: str) -> list[float]:
        return GoogleGenerativeAIEmbeddings(model=self.model_name, api_key=self.google_api_key).embed_query(text)
    
    def embed_batch_text(self, texts: list[str]) -> list[list[float]]:
        return GoogleGenerativeAIEmbeddings(model=self.model_name, api_key=self.google_api_key).embed_documents(texts)
    