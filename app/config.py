"""
Configuration for Research Assistant
Fill in your actual API keys here
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys - Replace with your actual keys or use environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_tavily_api_key_here")

# ChromaDB Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

# Application Configuration
APP_NAME = "Research Assistant"
MAX_FILE_SIZE_MB = 10
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_K = 5
TEMPERATURE = 0.7

# Retrieval Configuration
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5
RERANK_TOP_K = 20

# UI Configuration
SIDEBAR_WIDTH = 300
MAX_CHAT_HISTORY = 50

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"

# Vector Database Configuration
VECTOR_DIMENSIONS = 384  # For all-MiniLM-L6-v2
COLLECTION_NAME = "research_docs"

# Web Search Configuration
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_TIMEOUT = 10 