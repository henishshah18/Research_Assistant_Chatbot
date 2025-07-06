"""
UI Components Utility
Provides reusable UI components for the Research Assistant
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import time
import plotly.graph_objects as go
import plotly.express as px
from config import OPENAI_API_KEY, TAVILY_API_KEY

class UIComponents:
    """Utility class for UI components"""
    
    def __init__(self):
        pass
    
    def display_api_status(self) -> None:
        """Display API key configuration status"""
        st.subheader("🔑 API Status")
        
        # OpenAI API Status
        openai_status = "✅ Configured" if OPENAI_API_KEY != "your_openai_api_key_here" else "❌ Not Configured"
        openai_color = "green" if OPENAI_API_KEY != "your_openai_api_key_here" else "red"
        
        # Tavily API Status
        tavily_status = "✅ Configured" if TAVILY_API_KEY != "your_tavily_api_key_here" else "❌ Not Configured"
        tavily_color = "green" if TAVILY_API_KEY != "your_tavily_api_key_here" else "red"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**OpenAI**: :{openai_color}[{openai_status}]")
        
        with col2:
            st.markdown(f"**Tavily**: :{tavily_color}[{tavily_status}]")
        
        # Configuration help
        if OPENAI_API_KEY == "your_openai_api_key_here" or TAVILY_API_KEY == "your_tavily_api_key_here":
            with st.expander("🔧 Configuration Help"):
                st.markdown("""
                **To configure API keys:**
                1. Copy `config_example.py` to `config.py`
                2. Add your API keys to `config.py`
                3. Restart the application
                
                **Get API Keys:**
                - OpenAI: https://platform.openai.com/api-keys
                - Tavily: https://tavily.com/
                """)
    
    def display_processing_progress(self, current: int, total: int, message: str = "Processing...") -> None:
        """Display processing progress"""
        progress = current / total if total > 0 else 0
        st.progress(progress, text=f"{message} ({current}/{total})")
    
    def display_search_results(self, results: List[Dict[str, Any]], max_display: int = 5) -> None:
        """Display search results in a formatted way"""
        if not results:
            st.info("No search results found.")
            return
        
        st.subheader(f"📋 Search Results ({len(results)} found)")
        
        for i, result in enumerate(results[:max_display]):
            with st.expander(f"📄 {result.get('title', 'Untitled')} (Score: {result.get('score', 0):.2f})"):
                
                # Source information
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Source**: {result.get('source', 'Unknown')}")
                    if result.get('url'):
                        st.markdown(f"**URL**: {result.get('url')}")
                
                with col2:
                    st.markdown(f"**Type**: {result.get('type', 'document')}")
                    if result.get('score'):
                        st.metric("Relevance", f"{result.get('score'):.2f}")
                
                # Content
                st.markdown("**Content**:")
                st.markdown(result.get('content', 'No content available')[:500] + "...")
                
                # Metadata
                if result.get('metadata'):
                    with st.expander("🔍 Metadata"):
                        st.json(result.get('metadata'))
        
        if len(results) > max_display:
            st.info(f"Showing {max_display} of {len(results)} results")
    
    def display_system_metrics(self, stats: Dict[str, Any]) -> None:
        """Display system performance metrics"""
        st.subheader("📊 System Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Documents",
                stats.get('doc_count', 0),
                help="Number of processed documents"
            )
        
        with col2:
            st.metric(
                "Chunks",
                stats.get('chunk_count', 0),
                help="Total text chunks indexed"
            )
        
        with col3:
            st.metric(
                "Avg Chunk Size",
                f"{stats.get('avg_chunk_size', 0):.0f}",
                help="Average characters per chunk"
            )
        
        with col4:
            st.metric(
                "Total Tokens",
                stats.get('total_tokens', 0),
                help="Total tokens processed"
            )
    
    def display_search_analytics(self, search_results: List[Dict[str, Any]]) -> None:
        """Display search analytics visualization"""
        if not search_results:
            return
        
        st.subheader("📈 Search Analytics")
        
        # Score distribution
        scores = [r.get('score', 0) for r in search_results]
        types = [r.get('type', 'unknown') for r in search_results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution histogram
            fig_scores = go.Figure()
            fig_scores.add_trace(go.Histogram(
                x=scores,
                nbinsx=10,
                name="Relevance Scores",
                marker_color='blue'
            ))
            fig_scores.update_layout(
                title="Relevance Score Distribution",
                xaxis_title="Score",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_scores, use_container_width=True)
        
        with col2:
            # Source type distribution
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            
            fig_types = go.Figure()
            fig_types.add_trace(go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                name="Source Types"
            ))
            fig_types.update_layout(
                title="Source Type Distribution",
                height=300
            )
            st.plotly_chart(fig_types, use_container_width=True)
    
    def display_response_quality(self, quality_assessment: Dict[str, Any]) -> None:
        """Display response quality assessment"""
        if not quality_assessment or 'error' in quality_assessment:
            return
        
        st.subheader("🎯 Response Quality")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quality_score = quality_assessment.get('quality_score', 0)
            st.metric(
                "Quality Score",
                f"{quality_score:.1%}",
                help="Overall response quality assessment"
            )
        
        with col2:
            st.metric(
                "Sources Used",
                quality_assessment.get('source_count', 0),
                help="Number of sources referenced"
            )
        
        with col3:
            st.metric(
                "Citations",
                quality_assessment.get('citation_count', 0),
                help="Number of citations in response"
            )
        
        # Quality assessment details
        with st.expander("📝 Quality Details"):
            st.write(f"**Assessment**: {quality_assessment.get('assessment', 'N/A')}")
            st.write(f"**Document Sources**: {quality_assessment.get('doc_sources', 0)}")
            st.write(f"**Web Sources**: {quality_assessment.get('web_sources', 0)}")
            st.write(f"**Response Length**: {quality_assessment.get('response_length', 0)} words")
    
    def display_follow_up_questions(self, questions: List[str]) -> None:
        """Display follow-up questions"""
        if not questions:
            return
        
        st.subheader("🤔 Follow-up Questions")
        
        for i, question in enumerate(questions, 1):
            if st.button(f"❓ {question}", key=f"followup_{i}"):
                st.session_state.follow_up_query = question
                st.rerun()
    
    def display_source_credibility(self, sources: List[Dict[str, Any]]) -> None:
        """Display source credibility assessment"""
        web_sources = [s for s in sources if s.get('type') in ['web_result', 'web_answer']]
        
        if not web_sources:
            return
        
        st.subheader("🔍 Source Credibility")
        
        for source in web_sources:
            credibility = source.get('credibility', {})
            if credibility:
                with st.expander(f"🌐 {source.get('title', 'Web Source')}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Domain**: {credibility.get('domain', 'Unknown')}")
                        st.write(f"**Assessment**: {credibility.get('assessment', 'N/A')}")
                    
                    with col2:
                        credibility_score = credibility.get('score', 0)
                        color = 'green' if credibility_score >= 0.8 else 'orange' if credibility_score >= 0.6 else 'red'
                        st.metric("Credibility", f"{credibility_score:.1%}", delta_color=color)
    
    def display_search_tips(self) -> None:
        """Display search tips and help"""
        with st.expander("💡 Search Tips"):
            st.markdown("""
            **For better results:**
            - Use specific keywords and phrases
            - Ask complete questions rather than single words
            - Combine document and web search for comprehensive results
            - Use quotation marks for exact phrases
            
            **Search modes:**
            - **Hybrid**: Combines document and web search (recommended)
            - **Dense**: Semantic similarity search
            - **Sparse**: Keyword-based search
            - **Web Only**: Real-time web search
            
            **Examples:**
            - "What are the main benefits of renewable energy?"
            - "How does machine learning work in healthcare?"
            - "Latest developments in artificial intelligence"
            """)
    
    def display_loading_animation(self, message: str = "Processing...") -> None:
        """Display loading animation"""
        with st.spinner(message):
            time.sleep(0.1)  # Small delay for visual effect
    
    def display_error_message(self, error: str, suggestion: str = "") -> None:
        """Display formatted error message"""
        st.error(f"❌ {error}")
        if suggestion:
            st.info(f"💡 {suggestion}")
    
    def display_success_message(self, message: str) -> None:
        """Display success message"""
        st.success(f"✅ {message}")
    
    def display_warning_message(self, message: str) -> None:
        """Display warning message"""
        st.warning(f"⚠️ {message}")
    
    def create_download_button(self, content: str, filename: str, label: str = "Download") -> None:
        """Create download button for content"""
        st.download_button(
            label=label,
            data=content,
            file_name=filename,
            mime='text/plain'
        ) 