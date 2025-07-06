"""
Web Search Component
Handles real-time web search using Tavily API
"""

import streamlit as st
from tavily import TavilyClient
from typing import List, Dict, Any, Optional
import requests
import time
from config import TAVILY_API_KEY, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT

class WebSearch:
    """Web search integration using Tavily API"""
    
    def __init__(self):
        self.client = None
        self.is_initialized = False
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Tavily client"""
        try:
            if TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key_here":
                self.client = TavilyClient(api_key=TAVILY_API_KEY)
                self.is_initialized = True
            else:
                st.warning("⚠️ Tavily API key not configured. Web search will be disabled.")
                self.is_initialized = False
                
        except Exception as e:
            st.error(f"Failed to initialize Tavily client: {str(e)}")
            self.is_initialized = False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for information
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of web search results
        """
        if not self.is_initialized:
            st.warning("Web search not available - API key not configured")
            return []
        
        try:
            with st.spinner("Searching the web..."):
                # Perform web search
                results = self.client.search(
                    query=query,
                    max_results=min(k, WEB_SEARCH_MAX_RESULTS),
                    search_depth="advanced",
                    include_domains=None,
                    exclude_domains=None,
                    include_answer=True,
                    include_raw_content=False
                )
                
                # Format results
                formatted_results = []
                
                # Add direct answer if available
                if results.get('answer'):
                    formatted_results.append({
                        'content': results['answer'],
                        'title': 'Direct Answer',
                        'source': 'Tavily AI',
                        'url': '',
                        'score': 1.0,
                        'type': 'web_answer',
                        'metadata': {
                            'search_query': query,
                            'timestamp': time.time()
                        }
                    })
                
                # Add web results
                for result in results.get('results', []):
                    formatted_results.append({
                        'content': result.get('content', ''),
                        'title': result.get('title', 'No Title'),
                        'source': result.get('url', ''),
                        'url': result.get('url', ''),
                        'score': result.get('score', 0.5),
                        'type': 'web_result',
                        'metadata': {
                            'search_query': query,
                            'timestamp': time.time(),
                            'published_date': result.get('published_date', ''),
                        }
                    })
                
                return formatted_results[:k]
                
        except Exception as e:
            st.error(f"Web search failed: {str(e)}")
            return []
    
    def search_news(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for recent news articles
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of news search results
        """
        if not self.is_initialized:
            return []
        
        try:
            # Add news-specific search terms
            news_query = f"{query} news recent"
            
            results = self.client.search(
                query=news_query,
                max_results=min(k, WEB_SEARCH_MAX_RESULTS),
                search_depth="advanced",
                include_domains=["news.google.com", "reuters.com", "bbc.com", "cnn.com", "apnews.com"],
                include_answer=False,
                include_raw_content=False
            )
            
            # Format results
            formatted_results = []
            
            for result in results.get('results', []):
                formatted_results.append({
                    'content': result.get('content', ''),
                    'title': result.get('title', 'No Title'),
                    'source': result.get('url', ''),
                    'url': result.get('url', ''),
                    'score': result.get('score', 0.5),
                    'type': 'news_result',
                    'metadata': {
                        'search_query': query,
                        'timestamp': time.time(),
                        'published_date': result.get('published_date', ''),
                    }
                })
            
            return formatted_results[:k]
            
        except Exception as e:
            st.error(f"News search failed: {str(e)}")
            return []
    
    def search_academic(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for academic/scholarly content
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of academic search results
        """
        if not self.is_initialized:
            return []
        
        try:
            # Add academic-specific search terms
            academic_query = f"{query} research paper academic study"
            
            results = self.client.search(
                query=academic_query,
                max_results=min(k, WEB_SEARCH_MAX_RESULTS),
                search_depth="advanced",
                include_domains=["scholar.google.com", "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "jstor.org"],
                include_answer=False,
                include_raw_content=False
            )
            
            # Format results
            formatted_results = []
            
            for result in results.get('results', []):
                formatted_results.append({
                    'content': result.get('content', ''),
                    'title': result.get('title', 'No Title'),
                    'source': result.get('url', ''),
                    'url': result.get('url', ''),
                    'score': result.get('score', 0.5),
                    'type': 'academic_result',
                    'metadata': {
                        'search_query': query,
                        'timestamp': time.time(),
                        'published_date': result.get('published_date', ''),
                    }
                })
            
            return formatted_results[:k]
            
        except Exception as e:
            st.error(f"Academic search failed: {str(e)}")
            return []
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """
        Get search suggestions for a query
        
        Args:
            query: Partial search query
            
        Returns:
            List of suggested search terms
        """
        if not self.is_initialized:
            return []
        
        try:
            # Simple query expansion suggestions
            suggestions = [
                f"{query} definition",
                f"{query} examples",
                f"{query} latest news",
                f"{query} research",
                f"how to {query}",
                f"{query} best practices",
                f"{query} comparison",
                f"{query} benefits risks"
            ]
            
            return suggestions[:5]
            
        except Exception as e:
            return []
    
    def verify_source_credibility(self, url: str) -> Dict[str, Any]:
        """
        Assess the credibility of a web source
        
        Args:
            url: URL to assess
            
        Returns:
            Credibility assessment
        """
        try:
            # Simple domain-based credibility scoring
            high_credibility_domains = [
                'wikipedia.org', 'britannica.com', 'nature.com', 'science.org',
                'reuters.com', 'bbc.com', 'apnews.com', 'npr.org',
                'harvard.edu', 'mit.edu', 'stanford.edu', 'oxford.ac.uk',
                'gov', 'edu', 'org'
            ]
            
            medium_credibility_domains = [
                'cnn.com', 'nytimes.com', 'washingtonpost.com', 'theguardian.com',
                'wsj.com', 'bloomberg.com', 'techcrunch.com', 'wired.com'
            ]
            
            # Extract domain
            domain = url.split('//')[1].split('/')[0] if '//' in url else url.split('/')[0]
            
            # Assess credibility
            if any(trusted in domain for trusted in high_credibility_domains):
                credibility = 'high'
                score = 0.9
            elif any(medium in domain for medium in medium_credibility_domains):
                credibility = 'medium'
                score = 0.7
            else:
                credibility = 'unknown'
                score = 0.5
            
            return {
                'credibility': credibility,
                'score': score,
                'domain': domain,
                'assessment': f"Source credibility: {credibility}"
            }
            
        except Exception as e:
            return {
                'credibility': 'unknown',
                'score': 0.5,
                'domain': 'unknown',
                'assessment': 'Unable to assess credibility'
            }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get web search statistics"""
        return {
            'is_initialized': self.is_initialized,
            'client': 'Tavily' if self.is_initialized else 'None',
            'max_results': WEB_SEARCH_MAX_RESULTS,
            'timeout': WEB_SEARCH_TIMEOUT,
            'api_key_configured': TAVILY_API_KEY != "your_tavily_api_key_here"
        } 