import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_DIR=os.getenv('DATA_DIR', 'data')
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = os.getenv('MILVUS_PORT', 19530)
    
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    DEBUG = True 
    
    DIMENSIONS = 768
    VECTOR_WEIGHT = 0.6
    FULL_TEXT_WEIGHT = 0.2
    UNIT_WEIGHT = 0.2