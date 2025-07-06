"""
Response Synthesis Component
Generates comprehensive responses using GPT-4o-mini from multiple sources
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any, Optional
import json
import time
from config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE

class ResponseSynthesis:
    """Synthesizes responses from multiple sources using GPT-4o-mini"""
    
    def __init__(self):
        self.llm = None
        self.is_initialized = False
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """Initialize the language model"""
        try:
            if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
                self.llm = ChatOpenAI(
                    model=LLM_MODEL,
                    temperature=TEMPERATURE,
                    api_key=OPENAI_API_KEY,
                    max_tokens=2000,
                    streaming=True
                )
                self.is_initialized = True
            else:
                st.error("⚠️ OpenAI API key not configured. Response synthesis will not work.")
                self.is_initialized = False
                
        except Exception as e:
            st.error(f"Failed to initialize language model: {str(e)}")
            self.is_initialized = False
    
    def synthesize(self, query: str, sources: List[Dict[str, Any]], temperature: float = None) -> str:
        """
        Synthesize a comprehensive response from multiple sources
        
        Args:
            query: User query
            sources: List of source documents and web results
            temperature: Response creativity (overrides default)
            
        Returns:
            Synthesized response
        """
        if not self.is_initialized:
            return "❌ Response synthesis not available - OpenAI API key not configured."
        
        if not sources:
            return "❌ No sources found to generate a response."
        
        try:
            # Update temperature if provided
            if temperature is not None:
                self.llm.temperature = temperature
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with sources
            user_prompt = self._create_user_prompt(query, sources)
            
            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Add source citations
            cited_response = self._add_citations(response.content, sources)
            
            return cited_response
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for response synthesis"""
        return """You are a knowledgeable Research Assistant that synthesizes information from multiple sources to provide comprehensive, accurate, and well-structured responses.

Your responsibilities:
1. **Accuracy**: Only use information directly provided in the sources
2. **Comprehensiveness**: Synthesize information from all relevant sources
3. **Clarity**: Present information in a clear, well-organized manner
4. **Citations**: Reference sources appropriately throughout your response
5. **Transparency**: Distinguish between document sources and web sources
6. **Objectivity**: Present multiple perspectives when they exist

Response Structure:
- Start with a direct answer to the user's question
- Provide detailed explanation with supporting evidence
- Include relevant examples or case studies from sources
- Highlight any conflicting information between sources
- End with a concise summary

Citation Format:
- Use [Source: filename] for document sources
- Use [Source: website/domain] for web sources
- Use [Web: Direct Answer] for Tavily AI responses

Quality Guidelines:
- Write in a professional, academic tone
- Use bullet points or numbered lists for clarity when appropriate
- Avoid speculation beyond what's supported by sources
- If sources are insufficient, clearly state limitations
- Maintain objectivity and present balanced perspectives"""
    
    def _create_user_prompt(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Create user prompt with query and sources"""
        
        # Organize sources by type
        doc_sources = [s for s in sources if s.get('type') == 'document']
        web_sources = [s for s in sources if s.get('type') in ['web_result', 'web_answer', 'news_result', 'academic_result']]
        
        prompt = f"""Please provide a comprehensive response to this query: "{query}"

Based on the following sources:

"""
        
        # Add document sources
        if doc_sources:
            prompt += "📄 **DOCUMENT SOURCES:**\n"
            for i, source in enumerate(doc_sources, 1):
                prompt += f"\n{i}. **Source**: {source.get('source', 'Unknown')}\n"
                prompt += f"   **Content**: {source.get('content', '')[:800]}...\n"
                prompt += f"   **Relevance Score**: {source.get('score', 0):.2f}\n"
        
        # Add web sources
        if web_sources:
            prompt += "\n🌐 **WEB SOURCES:**\n"
            for i, source in enumerate(web_sources, 1):
                prompt += f"\n{i}. **Title**: {source.get('title', 'No Title')}\n"
                prompt += f"   **URL**: {source.get('url', 'No URL')}\n"
                prompt += f"   **Content**: {source.get('content', '')[:800]}...\n"
                prompt += f"   **Type**: {source.get('type', 'web_result')}\n"
                prompt += f"   **Relevance Score**: {source.get('score', 0):.2f}\n"
        
        prompt += f"""
**Instructions:**
1. Synthesize information from all sources to answer the query comprehensively
2. Use proper citations throughout your response
3. If sources provide conflicting information, present both perspectives
4. Structure your response clearly with headings if needed
5. Maintain academic rigor while being accessible
6. If the sources don't fully address the query, clearly state what's missing

**Query to Answer**: {query}
"""
        
        return prompt
    
    def _add_citations(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """Add source citations to response"""
        try:
            # Simple citation enhancement
            # In a more sophisticated implementation, you would use NLP to match
            # content to specific sources
            
            citation_section = "\n\n---\n\n## 📚 Sources Referenced\n\n"
            
            for i, source in enumerate(sources, 1):
                if source.get('type') == 'document':
                    citation_section += f"{i}. **{source.get('source', 'Unknown Document')}** "
                    citation_section += f"(Chunk {source.get('metadata', {}).get('chunk_index', 'N/A')})\n"
                else:
                    citation_section += f"{i}. **{source.get('title', 'Web Source')}** "
                    citation_section += f"- {source.get('url', 'No URL')}\n"
                    
                    # Add credibility assessment for web sources
                    if hasattr(self, 'web_search') and source.get('url'):
                        credibility = self.web_search.verify_source_credibility(source.get('url'))
                        citation_section += f"   *Credibility: {credibility.get('credibility', 'unknown')}*\n"
            
            return response + citation_section
            
        except Exception as e:
            return response + f"\n\n---\n\n*Note: Could not generate detailed citations due to error: {str(e)}*"
    
    def generate_follow_up_questions(self, query: str, response: str) -> List[str]:
        """Generate follow-up questions based on the query and response"""
        if not self.is_initialized:
            return []
        
        try:
            follow_up_prompt = f"""Based on this query and response, generate 3-5 relevant follow-up questions that would help the user explore the topic further.

Original Query: {query}

Response: {response[:1000]}...

Generate follow-up questions that:
1. Dive deeper into specific aspects mentioned
2. Explore related topics
3. Ask for clarification or examples
4. Suggest practical applications

Format: Return only the questions, one per line, without numbers or bullets."""
            
            messages = [
                SystemMessage(content="You are a helpful assistant that generates insightful follow-up questions."),
                HumanMessage(content=follow_up_prompt)
            ]
            
            response_obj = self.llm.invoke(messages)
            questions = [q.strip() for q in response_obj.content.split('\n') if q.strip()]
            
            return questions[:5]
            
        except Exception as e:
            return []
    
    def summarize_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Generate a summary of all sources"""
        if not self.is_initialized or not sources:
            return "No sources available for summary."
        
        try:
            sources_text = "\n\n".join([
                f"**{source.get('title', 'Unknown')}**\n{source.get('content', '')[:500]}..."
                for source in sources
            ])
            
            summary_prompt = f"""Please provide a concise summary of the following sources, highlighting key themes and main points:

{sources_text}

Summary should be:
- 2-3 paragraphs maximum
- Focus on main themes and key findings
- Note any conflicting information
- Maintain objectivity"""
            
            messages = [
                SystemMessage(content="You are a helpful assistant that creates concise, accurate summaries."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def assess_response_quality(self, query: str, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of the generated response"""
        try:
            # Simple quality metrics
            source_count = len(sources)
            doc_sources = len([s for s in sources if s.get('type') == 'document'])
            web_sources = len([s for s in sources if s.get('type') in ['web_result', 'web_answer']])
            
            response_length = len(response.split())
            citation_count = response.count('[Source:') + response.count('[Web:')
            
            # Quality assessment
            quality_score = 0
            
            # Source diversity
            if source_count >= 3:
                quality_score += 0.3
            elif source_count >= 2:
                quality_score += 0.2
            
            # Response comprehensiveness
            if response_length >= 200:
                quality_score += 0.3
            elif response_length >= 100:
                quality_score += 0.2
            
            # Citation usage
            if citation_count >= 2:
                quality_score += 0.2
            elif citation_count >= 1:
                quality_score += 0.1
            
            # Source mix
            if doc_sources > 0 and web_sources > 0:
                quality_score += 0.2
            
            return {
                'quality_score': min(quality_score, 1.0),
                'source_count': source_count,
                'doc_sources': doc_sources,
                'web_sources': web_sources,
                'response_length': response_length,
                'citation_count': citation_count,
                'assessment': 'Good' if quality_score >= 0.7 else 'Fair' if quality_score >= 0.5 else 'Needs Improvement'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get response synthesis statistics"""
        return {
            'is_initialized': self.is_initialized,
            'model': LLM_MODEL,
            'temperature': TEMPERATURE,
            'max_tokens': 2000,
            'api_key_configured': OPENAI_API_KEY != "your_openai_api_key_here"
        } 