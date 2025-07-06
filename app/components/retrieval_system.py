"""
Hybrid Retrieval System Component
Combines dense (semantic) and sparse (keyword) retrieval with re-ranking
"""

import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path
import json
import pickle
from config import (
    CHROMA_DB_PATH, EMBEDDING_MODEL, RERANK_MODEL, VECTOR_DIMENSIONS,
    COLLECTION_NAME, DENSE_WEIGHT, SPARSE_WEIGHT, RERANK_TOP_K
)

class RetrievalSystem:
    """Hybrid retrieval system combining dense, sparse, and re-ranking methods"""
    
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.rerank_model = None
        self.bm25 = None
        self.documents = []
        self.is_initialized = False
        
    def initialize(self, chunks: List[Dict[str, Any]]) -> None:
        """Initialize the retrieval system with document chunks"""
        try:
            st.info("Initializing retrieval system...")
            
            # Store documents
            self.documents = chunks
            
            # Initialize ChromaDB
            self._init_chromadb()
            
            # Initialize embedding model
            self._init_embedding_model()
            
            # Initialize BM25 for sparse retrieval
            self._init_bm25()
            
            # Initialize re-ranking model
            self._init_rerank_model()
            
            # Process and store embeddings
            self._process_and_store_embeddings()
            
            self.is_initialized = True
            st.success("✅ Retrieval system initialized successfully!")
            
        except Exception as e:
            st.error(f"Failed to initialize retrieval system: {str(e)}")
            raise
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection"""
        try:
            # Ensure data directory exists
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
                # Delete existing collection to refresh
                self.chroma_client.delete_collection(name=COLLECTION_NAME)
            except:
                pass
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB: {str(e)}")
    
    def _init_embedding_model(self) -> None:
        """Initialize sentence transformer model"""
        try:
            with st.spinner("Loading embedding model..."):
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def _init_bm25(self) -> None:
        """Initialize BM25 for sparse retrieval"""
        try:
            # Check if we have documents
            if not self.documents:
                st.warning("No documents available for BM25 initialization")
                self.bm25 = None
                return
            
            # Prepare corpus for BM25
            corpus = []
            for doc in self.documents:
                if doc.get("text"):
                    tokens = doc["text"].lower().split()
                    if tokens:  # Only add non-empty token lists
                        corpus.append(tokens)
            
            if not corpus:
                st.warning("No valid text found for BM25 initialization")
                self.bm25 = None
                return
            
            self.bm25 = BM25Okapi(corpus)
            
        except Exception as e:
            st.error(f"Failed to initialize BM25: {str(e)}")
            self.bm25 = None
    
    def _init_rerank_model(self) -> None:
        """Initialize cross-encoder for re-ranking"""
        try:
            with st.spinner("Loading re-ranking model..."):
                self.rerank_model = CrossEncoder(RERANK_MODEL)
                
        except Exception as e:
            st.warning(f"Failed to load re-ranking model: {str(e)}. Re-ranking will be disabled.")
            self.rerank_model = None
    
    def _process_and_store_embeddings(self) -> None:
        """Process documents and store embeddings in ChromaDB"""
        try:
            with st.spinner("Creating embeddings..."):
                # Extract texts and metadata
                texts = [doc["text"] for doc in self.documents]
                ids = [doc["id"] for doc in self.documents]
                metadatas = [doc["metadata"] for doc in self.documents]
                
                # Generate embeddings in batches
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    all_embeddings.extend(batch_embeddings.tolist())
                
                # Store in ChromaDB
                self.collection.add(
                    embeddings=all_embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
        except Exception as e:
            raise Exception(f"Failed to process embeddings: {str(e)}")
    
    def search(self, query: str, mode: str = "hybrid", k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents using specified mode
        
        Args:
            query: Search query
            mode: Search mode ('dense', 'sparse', 'hybrid')
            k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        if not self.is_initialized:
            st.error("Retrieval system not initialized")
            return []
        
        try:
            if mode == "dense":
                return self._dense_search(query, k)
            elif mode == "sparse":
                return self._sparse_search(query, k)
            elif mode == "hybrid":
                return self._hybrid_search(query, k)
            else:
                return self._hybrid_search(query, k)  # Default to hybrid
                
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return []
    
    def _dense_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Dense (semantic) search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k
            )
            
            # Format results
            search_results = []
            for i in range(len(results['documents'][0])):
                doc_id = results['ids'][0][i]
                doc = next((d for d in self.documents if d['id'] == doc_id), None)
                if doc:
                    search_results.append({
                        'content': results['documents'][0][i],
                        'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'source': doc['source'],
                        'title': f"Chunk {doc['chunk_index']} - {doc['source']}",
                        'type': 'document',
                        'metadata': doc['metadata']
                    })
            
            return search_results
            
        except Exception as e:
            raise Exception(f"Dense search failed: {str(e)}")
    
    def _sparse_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Sparse (keyword) search using BM25"""
        try:
            if not self.bm25:
                st.warning("BM25 not available, using simple text matching")
                # Fallback to simple text matching
                search_results = []
                query_lower = query.lower()
                
                for doc in self.documents:
                    if query_lower in doc['text'].lower():
                        # Simple scoring based on term frequency
                        score = doc['text'].lower().count(query_lower) / len(doc['text'].split())
                        search_results.append({
                            'content': doc['text'],
                            'score': score,
                            'source': doc['source'],
                            'title': f"Chunk {doc['chunk_index']} - {doc['source']}",
                            'type': 'document',
                            'metadata': doc['metadata']
                        })
                
                # Sort by score and return top k
                search_results.sort(key=lambda x: x['score'], reverse=True)
                return search_results[:k]
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Get top k results
            top_indices = np.argsort(bm25_scores)[-k:][::-1]
            
            # Format results
            search_results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    search_results.append({
                        'content': doc['text'],
                        'score': float(bm25_scores[idx]),
                        'source': doc['source'],
                        'title': f"Chunk {doc['chunk_index']} - {doc['source']}",
                        'type': 'document',
                        'metadata': doc['metadata']
                    })
            
            return search_results
            
        except Exception as e:
            raise Exception(f"Sparse search failed: {str(e)}")
    
    def _hybrid_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Hybrid search combining dense and sparse retrieval"""
        try:
            # Get more results from each method for better fusion
            expanded_k = min(k * 2, len(self.documents))
            
            # Perform dense and sparse searches
            dense_results = self._dense_search(query, expanded_k)
            sparse_results = self._sparse_search(query, expanded_k)
            
            # Combine results using reciprocal rank fusion
            combined_results = self._reciprocal_rank_fusion(
                dense_results, sparse_results, k
            )
            
            # Apply re-ranking if available
            if self.rerank_model and len(combined_results) > 1:
                combined_results = self._rerank_results(query, combined_results)
            
            return combined_results[:k]
            
        except Exception as e:
            raise Exception(f"Hybrid search failed: {str(e)}")
    
    def _reciprocal_rank_fusion(self, dense_results: List[Dict], sparse_results: List[Dict], k: int) -> List[Dict]:
        """Combine dense and sparse results using reciprocal rank fusion"""
        try:
            # Create separate mappings for scores and documents
            score_map = {}
            doc_map = {}
            
            # Add dense results
            for rank, result in enumerate(dense_results):
                doc_id = result['title']  # Using title as unique identifier
                score_map[doc_id] = score_map.get(doc_id, 0) + DENSE_WEIGHT / (rank + 1)
                doc_map[doc_id] = result
            
            # Add sparse results
            for rank, result in enumerate(sparse_results):
                doc_id = result['title']
                score_map[doc_id] = score_map.get(doc_id, 0) + SPARSE_WEIGHT / (rank + 1)
                if doc_id not in doc_map:
                    doc_map[doc_id] = result
            
            # Create combined results
            combined_results = []
            
            # Sort by combined score
            sorted_docs = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            
            for doc_id, score in sorted_docs:
                if doc_id in doc_map:
                    result = doc_map[doc_id].copy()
                    result['score'] = score
                    combined_results.append(result)
            
            return combined_results
            
        except Exception as e:
            raise Exception(f"Reciprocal rank fusion failed: {str(e)}")
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-rank results using cross-encoder"""
        try:
            if not self.rerank_model or len(results) <= 1:
                return results
            
            # Prepare query-document pairs
            pairs = [[query, result['content']] for result in results]
            
            # Get re-ranking scores
            rerank_scores = self.rerank_model.predict(pairs)
            
            # Update results with new scores
            for i, result in enumerate(results):
                result['rerank_score'] = float(rerank_scores[i])
                result['original_score'] = result['score']
            
            # Sort by re-ranking score
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return results
            
        except Exception as e:
            st.warning(f"Re-ranking failed: {str(e)}")
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.is_initialized:
            return {}
        
        return {
            'doc_count': len(set(doc['source'] for doc in self.documents)),
            'chunk_count': len(self.documents),
            'total_tokens': sum(doc['word_count'] for doc in self.documents),
            'avg_chunk_size': np.mean([doc['char_count'] for doc in self.documents]),
            'embedding_model': EMBEDDING_MODEL,
            'rerank_model': RERANK_MODEL if self.rerank_model else "Disabled",
            'collection_size': self.collection.count() if self.collection else 0
        }
    
    def clear_index(self) -> None:
        """Clear the search index"""
        try:
            if self.collection:
                self.chroma_client.delete_collection(name=COLLECTION_NAME)
            self.is_initialized = False
            st.success("Search index cleared successfully!")
            
        except Exception as e:
            st.error(f"Failed to clear index: {str(e)}") 