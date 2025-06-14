import os

from src.services.llm_service.factory import LLMServiceFactory
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(api_key)

gemini_service = LLMServiceFactory.create_service(name="gemini", model="gemini-1.5-flash")
result = gemini_service.generate(
    system_prompt="You are a reviewer",
    user_prompt="You are very good",
    json_output=False
)
print(result)
