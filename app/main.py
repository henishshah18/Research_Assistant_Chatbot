"""
Research Assistant - Main Streamlit Application
A sophisticated hybrid search system with real-time web integration
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent
sys.path.append(str(app_dir))

# Import configuration
try:
    from config import *
except ImportError:
    st.error("⚠️ Please copy config_example.py to config.py and add your API keys!")
    st.stop()

# Import components
from components.pdf_processor import PDFProcessor
from components.retrieval_system import RetrievalSystem
from components.web_search import WebSearch
from components.response_synthesis import ResponseSynthesis
from utils.session_manager import SessionManager
from utils.ui_components import UIComponents

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .search-result {
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .source-citation {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'retrieval_system' not in st.session_state:
        st.session_state.retrieval_system = None
    if 'web_search' not in st.session_state:
        st.session_state.web_search = None
    if 'response_synthesis' not in st.session_state:
        st.session_state.response_synthesis = None

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced RAG with Real-time Web Integration & Hybrid Retrieval*")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Configuration")
        
        # API Key Status
        ui_components = UIComponents()
        ui_components.display_api_status()
        
        st.divider()
        
        # Document Upload Section
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF files to create your knowledge base"
        )
        
        if uploaded_files:
            process_button = st.button("🔄 Process Documents", type="primary")
            if process_button:
                with st.spinner("Processing documents..."):
                    process_documents(uploaded_files)
        
        st.divider()
        
        # System Settings
        st.header("⚙️ Search Settings")
        search_mode = st.selectbox(
            "Search Mode",
            ["Hybrid (Recommended)", "Dense Only", "Sparse Only", "Web Only"],
            help="Choose your preferred search strategy"
        )
        
        temperature = st.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=TEMPERATURE,
            step=0.1,
            help="Higher values = more creative responses"
        )
        
        max_sources = st.slider(
            "Max Sources",
            min_value=3,
            max_value=10,
            value=SEARCH_K,
            help="Maximum number of sources to retrieve"
        )
        
        st.divider()
        
        # System Stats
        if st.session_state.retrieval_system:
            st.header("📈 System Stats")
            stats = st.session_state.retrieval_system.get_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get('doc_count', 0))
            with col2:
                st.metric("Chunks", stats.get('chunk_count', 0))
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat Interface
        st.header("💬 Research Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"**{source['title']}**")
                            st.markdown(f"*{source['source']}*")
                            st.markdown(source['content'][:200] + "...")
                            st.divider()
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents or any topic..."):
            handle_user_query(prompt, search_mode, temperature, max_sources)
    
    with col2:
        # Information Panel
        st.header("ℹ️ System Information")
        
        # Query Analysis
        if st.session_state.messages:
            last_query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else ""
            if last_query:
                with st.expander("🔍 Last Query Analysis"):
                    st.markdown(f"**Query:** {last_query}")
                    st.markdown("**Search Strategy:** Hybrid Retrieval")
                    st.markdown("**Sources:** Document + Web")
        
        # Help Section
        with st.expander("❓ How to Use"):
            st.markdown("""
            **Getting Started:**
            1. Upload PDF documents using the sidebar
            2. Wait for processing to complete
            3. Ask questions in the chat interface
            
            **Features:**
            - **Document Search**: Query your uploaded PDFs
            - **Web Search**: Get real-time information
            - **Hybrid Retrieval**: Best of both worlds
            - **Source Citations**: Full transparency
            """)
        
        # System Architecture
        with st.expander("🏗️ Architecture"):
            st.markdown("""
            **Components:**
            - Dense Retrieval (Semantic)
            - Sparse Retrieval (Keyword)
            - Web Search (Tavily)
            - Re-ranking (Cross-encoder)
            - Response Synthesis (GPT-4o-mini)
            """)

def process_documents(uploaded_files):
    """Process uploaded PDF documents"""
    try:
        processor = PDFProcessor()
        retrieval_system = RetrievalSystem()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_chunks = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Process PDF
            chunks = processor.process_pdf(file)
            all_chunks.extend(chunks)
        
        # Initialize retrieval system
        status_text.text("Building search indexes...")
        retrieval_system.initialize(all_chunks)
        
        # Store in session state
        st.session_state.retrieval_system = retrieval_system
        st.session_state.documents = uploaded_files
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        st.success(f"Successfully processed {len(uploaded_files)} documents with {len(all_chunks)} chunks!")
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")

def handle_user_query(prompt, search_mode, temperature, max_sources):
    """Handle user query and generate response"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                # Initialize components if needed
                if not st.session_state.web_search:
                    st.session_state.web_search = WebSearch()
                if not st.session_state.response_synthesis:
                    st.session_state.response_synthesis = ResponseSynthesis()
                
                # Perform search based on mode
                sources = []
                
                if search_mode in ["Hybrid (Recommended)", "Dense Only", "Sparse Only"]:
                    if st.session_state.retrieval_system:
                        doc_sources = st.session_state.retrieval_system.search(
                            prompt, 
                            mode=search_mode.lower().replace(" (recommended)", ""),
                            k=max_sources
                        )
                        sources.extend(doc_sources)
                
                if search_mode in ["Hybrid (Recommended)", "Web Only"]:
                    web_sources = st.session_state.web_search.search(prompt, k=3)
                    sources.extend(web_sources)
                
                # Generate response
                response = st.session_state.response_synthesis.synthesize(
                    query=prompt,
                    sources=sources,
                    temperature=temperature
                )
                
                st.markdown(response)
                
                # Store response with sources
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main() 