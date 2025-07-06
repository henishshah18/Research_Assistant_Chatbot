"""
Configuration example for Research Assistant
Copy this file to config.py and fill in your API keys
"""

# API Keys - Replace with your actual keys
OPENAI_API_KEY = "your_openai_api_key_here"
TAVILY_API_KEY = "your_tavily_api_key_here"

# LangSmith (Optional - for monitoring)
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = "your_langsmith_api_key_here"
LANGCHAIN_PROJECT = "research-assistant"

# ChromaDB Configuration
CHROMA_DB_PATH = "./data/chroma_db"

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