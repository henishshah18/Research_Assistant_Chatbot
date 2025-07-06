"""
Session Manager Utility
Handles user sessions, chat history, and state management
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import json
import time
import uuid
from datetime import datetime, timedelta
from config import MAX_CHAT_HISTORY

class SessionManager:
    """Manages user sessions and chat history"""
    
    def __init__(self):
        self.session_id = self._get_or_create_session_id()
        self._initialize_session_state()
    
    def _get_or_create_session_id(self) -> str:
        """Get existing session ID or create new one"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables"""
        default_values = {
            'messages': [],
            'chat_history': [],
            'documents': [],
            'current_query': '',
            'last_search_results': [],
            'session_start_time': time.time(),
            'query_count': 0,
            'search_stats': {
                'total_queries': 0,
                'successful_queries': 0,
                'failed_queries': 0,
                'avg_response_time': 0,
                'total_sources_used': 0
            },
            'user_preferences': {
                'search_mode': 'hybrid',
                'max_sources': 5,
                'temperature': 0.7,
                'show_sources': True,
                'show_analytics': False
            },
            'processing_cache': {}
        }
        
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def add_message(self, role: str, content: str, sources: List[Dict[str, Any]] = None, 
                   metadata: Dict[str, Any] = None) -> None:
        """Add a message to the chat history"""
        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'sources': sources or [],
            'metadata': metadata or {}
        }
        
        st.session_state.messages.append(message)
        st.session_state.chat_history.append(message)
        
        # Limit chat history
        if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
    
    def get_chat_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get chat history with optional limit"""
        history = st.session_state.chat_history
        if limit:
            return history[-limit:]
        return history
    
    def clear_chat_history(self) -> None:
        """Clear all chat history"""
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.success("Chat history cleared!")
    
    def get_conversation_context(self, max_turns: int = 3) -> str:
        """Get conversation context for better responses"""
        recent_messages = self.get_chat_history(limit=max_turns * 2)
        
        context = ""
        for msg in recent_messages:
            if msg['role'] == 'user':
                context += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                context += f"Assistant: {msg['content'][:200]}...\n"
        
        return context
    
    def update_search_stats(self, success: bool, response_time: float, 
                          sources_count: int = 0) -> None:
        """Update search statistics"""
        stats = st.session_state.search_stats
        
        stats['total_queries'] += 1
        st.session_state.query_count += 1
        
        if success:
            stats['successful_queries'] += 1
            stats['total_sources_used'] += sources_count
        else:
            stats['failed_queries'] += 1
        
        # Update average response time
        total_time = stats['avg_response_time'] * (stats['total_queries'] - 1) + response_time
        stats['avg_response_time'] = total_time / stats['total_queries']
        
        st.session_state.search_stats = stats
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get current search statistics"""
        return st.session_state.search_stats.copy()
    
    def save_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """Save user preferences"""
        st.session_state.user_preferences.update(preferences)
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences"""
        return st.session_state.user_preferences.copy()
    
    def cache_processing_result(self, key: str, result: Any, expiry_hours: int = 24) -> None:
        """Cache processing results"""
        cache_entry = {
            'result': result,
            'timestamp': time.time(),
            'expiry': time.time() + (expiry_hours * 3600)
        }
        st.session_state.processing_cache[key] = cache_entry
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached processing result"""
        cache = st.session_state.processing_cache.get(key)
        
        if cache and time.time() < cache['expiry']:
            return cache['result']
        
        # Remove expired cache
        if cache:
            del st.session_state.processing_cache[key]
        
        return None
    
    def clear_cache(self) -> None:
        """Clear all cached results"""
        st.session_state.processing_cache = {}
        st.success("Cache cleared!")
    
    def export_chat_history(self, format: str = 'json') -> str:
        """Export chat history in specified format"""
        history = self.get_chat_history()
        
        if format == 'json':
            return json.dumps(history, indent=2, ensure_ascii=False)
        
        elif format == 'txt':
            text_export = f"Chat History - Session {self.session_id}\n"
            text_export += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            text_export += "=" * 50 + "\n\n"
            
            for msg in history:
                timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M:%S')
                text_export += f"[{timestamp}] {msg['role'].upper()}: {msg['content']}\n"
                
                if msg.get('sources'):
                    text_export += f"Sources: {len(msg['sources'])} references\n"
                
                text_export += "-" * 30 + "\n"
            
            return text_export
        
        elif format == 'markdown':
            md_export = f"# Chat History - Session {self.session_id}\n\n"
            md_export += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for msg in history:
                timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M:%S')
                
                if msg['role'] == 'user':
                    md_export += f"## 👤 User ({timestamp})\n{msg['content']}\n\n"
                else:
                    md_export += f"## 🤖 Assistant ({timestamp})\n{msg['content']}\n\n"
                    
                    if msg.get('sources'):
                        md_export += f"**Sources:** {len(msg['sources'])} references\n\n"
            
            return md_export
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary statistics"""
        session_time = time.time() - st.session_state.session_start_time
        
        return {
            'session_id': self.session_id,
            'duration_minutes': session_time / 60,
            'total_messages': len(st.session_state.messages),
            'user_queries': len([m for m in st.session_state.messages if m['role'] == 'user']),
            'documents_processed': len(st.session_state.documents),
            'search_stats': self.get_search_stats(),
            'start_time': datetime.fromtimestamp(st.session_state.session_start_time).isoformat(),
            'user_preferences': self.get_user_preferences()
        }
    
    def reset_session(self) -> None:
        """Reset the entire session"""
        keys_to_preserve = ['session_id']
        
        # Clear all session state except preserved keys
        for key in list(st.session_state.keys()):
            if key not in keys_to_preserve:
                del st.session_state[key]
        
        # Reinitialize
        self._initialize_session_state()
        st.success("Session reset successfully!")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get approximate memory usage of session data"""
        import sys
        
        messages_size = sys.getsizeof(st.session_state.messages)
        documents_size = sys.getsizeof(st.session_state.documents)
        cache_size = sys.getsizeof(st.session_state.processing_cache)
        
        return {
            'messages_kb': messages_size / 1024,
            'documents_kb': documents_size / 1024,
            'cache_kb': cache_size / 1024,
            'total_kb': (messages_size + documents_size + cache_size) / 1024
        }
    
    def cleanup_old_data(self, days_old: int = 7) -> None:
        """Clean up old session data"""
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        # Clean old messages
        st.session_state.messages = [
            msg for msg in st.session_state.messages 
            if datetime.fromisoformat(msg['timestamp']).timestamp() > cutoff_time
        ]
        
        # Clean expired cache
        expired_keys = [
            key for key, cache in st.session_state.processing_cache.items()
            if cache['timestamp'] < cutoff_time
        ]
        
        for key in expired_keys:
            del st.session_state.processing_cache[key]
        
        st.info(f"Cleaned up data older than {days_old} days") 