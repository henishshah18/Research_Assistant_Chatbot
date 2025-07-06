# 🔍 Research Assistant

A sophisticated hybrid search system that combines document processing, real-time web search, and AI-powered response synthesis to provide comprehensive research assistance.

## 🌟 Features

### 🔥 Core Capabilities
- **📄 PDF Document Processing**: Upload and process PDF documents with intelligent chunking
- **🌐 Real-time Web Search**: Integrate live web information using Tavily API
- **🔍 Hybrid Retrieval**: Combine dense (semantic) and sparse (keyword) search methods
- **🎯 Advanced Re-ranking**: Use cross-encoders for improved result precision
- **🤖 AI Response Synthesis**: Generate comprehensive answers using GPT-4o-mini
- **📊 Source Credibility Assessment**: Evaluate and rank source reliability

### 🛠️ Technical Features
- **Vector Database**: ChromaDB for efficient semantic search
- **Multiple Search Modes**: Dense, Sparse, Hybrid, and Web-only search
- **Session Management**: Persistent chat history and user preferences
- **Quality Monitoring**: Real-time response quality assessment
- **Source Citations**: Full transparency with proper source attribution
- **Analytics Dashboard**: Search performance metrics and visualizations

## 🏗️ Architecture

### Search Methods Implemented
1. **Dense Retrieval**: Semantic search using sentence-transformers embeddings
2. **Sparse Retrieval**: Keyword matching using BM25 algorithm
3. **Hybrid Retrieval**: Reciprocal rank fusion combining dense + sparse scores
4. **Re-ranking**: Cross-encoder models for final result optimization

### Technology Stack
- **Frontend**: Streamlit for interactive web interface
- **LLM**: OpenAI GPT-4o-mini for response generation
- **Vector DB**: ChromaDB for document storage and retrieval
- **Web Search**: Tavily API for real-time information
- **Embeddings**: Sentence-transformers for semantic understanding
- **Orchestration**: LangChain for workflow management

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API Key
- Tavily API Key (optional, for web search)

### Quick Start

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd research-assistant
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   ```bash
   cp config_example.py app/config.py
   # Edit app/config.py with your API keys
   ```

3. **Run the Application**
   ```bash
   python run_app.py
   ```

The application will start at `http://localhost:8501`

## 🔧 Configuration

### API Keys Required
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get from [Tavily](https://tavily.com/) (optional)

### Environment Variables (Optional)
```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
CHROMA_DB_PATH=./data/chroma_db
```

### Configuration Options
```python
# app/config.py
MAX_CHUNK_SIZE = 1000        # Document chunk size
CHUNK_OVERLAP = 200          # Overlap between chunks
SEARCH_K = 5                 # Number of results to retrieve
DENSE_WEIGHT = 0.5           # Weight for dense retrieval
SPARSE_WEIGHT = 0.5          # Weight for sparse retrieval
```

## 🚀 Usage

### 1. Upload Documents
- Use the sidebar to upload PDF documents
- Wait for processing to complete
- Documents are automatically chunked and indexed

### 2. Search Modes
- **Hybrid (Recommended)**: Best of both document and web search
- **Dense Only**: Semantic similarity search
- **Sparse Only**: Keyword-based search
- **Web Only**: Real-time web information

### 3. Chat Interface
- Ask questions in natural language
- Get comprehensive responses with source citations
- View search analytics and quality metrics

### 4. Advanced Features
- Export chat history (JSON, TXT, Markdown)
- Customize search parameters
- View source credibility assessments
- Monitor response quality metrics

## 📊 Search Performance

### Retrieval Methods Comparison
| Method | Strengths | Use Case |
|--------|-----------|----------|
| Dense | Semantic understanding, conceptual similarity | Complex queries, conceptual search |
| Sparse | Exact keyword matching, fast | Specific terms, names, dates |
| Hybrid | Best of both worlds | General purpose, comprehensive results |
| Re-ranking | Improved precision | Final result optimization |

### Optimization Features
- **Pre-computed Embeddings**: Faster search performance
- **Approximate Nearest Neighbor**: Efficient similarity search
- **Query Caching**: Reduced latency for repeated queries
- **Batch Processing**: Optimized document processing

## 🔍 API Reference

### Core Classes

#### `RetrievalSystem`
```python
# Initialize retrieval system
retrieval = RetrievalSystem()
retrieval.initialize(document_chunks)

# Perform hybrid search
results = retrieval.search(query="your question", mode="hybrid", k=5)
```

#### `WebSearch`
```python
# Initialize web search
web_search = WebSearch()

# Search the web
results = web_search.search(query="latest AI developments", k=3)
```

#### `ResponseSynthesis`
```python
# Generate comprehensive response
synthesis = ResponseSynthesis()
response = synthesis.synthesize(
    query="your question",
    sources=search_results,
    temperature=0.7
)
```

## 🎯 Advanced Configuration

### Custom Embedding Models
```python
# config.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, good quality
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better quality, slower
```

### Vector Database Settings
```python
# config.py
VECTOR_DIMENSIONS = 384      # Model-specific
COLLECTION_NAME = "research_docs"
CHROMA_DB_PATH = "./data/chroma_db"
```

### Web Search Configuration
```python
# config.py
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_TIMEOUT = 10
```

## 📈 Monitoring & Analytics

### Built-in Metrics
- Response quality scores
- Source diversity analysis
- Search performance tracking
- User interaction analytics

### Quality Assessment
- Source credibility evaluation
- Response comprehensiveness scoring
- Citation accuracy tracking
- User satisfaction indicators

## 🛡️ Security & Privacy

### Data Handling
- Documents processed locally
- No data sent to external services except for LLM calls
- Session data stored in browser memory only
- Optional chat history export

### API Security
- API keys stored locally in config files
- No logging of sensitive information
- Secure HTTPS connections for all API calls

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run app/main.py
```

### Code Structure
```
research-assistant/
├── app/
│   ├── main.py                    # Main Streamlit app
│   ├── config.py                  # Configuration settings
│   ├── components/
│   │   ├── pdf_processor.py       # PDF processing
│   │   ├── retrieval_system.py    # Hybrid search system
│   │   ├── web_search.py          # Web search integration
│   │   └── response_synthesis.py  # Response generation
│   └── utils/
│       ├── session_manager.py     # Session management
│       └── ui_components.py       # UI utilities
├── data/                          # Data storage
├── requirements.txt               # Python dependencies
└── run_app.py                    # Application launcher
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For orchestration framework
- **ChromaDB**: For vector database capabilities
- **Sentence Transformers**: For embedding models
- **Tavily**: For real-time web search
- **OpenAI**: For language model capabilities
- **Streamlit**: For the web interface framework

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the configuration guide

---

**Made with ❤️ for researchers, students, and knowledge workers** 