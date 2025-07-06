"""
PDF Document Processing Component
Handles PDF upload, text extraction, and chunking for the Research Assistant
"""

import streamlit as st
import fitz  # PyMuPDF
from typing import List, Dict, Any
from pathlib import Path
import hashlib
from config import MAX_CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_SIZE_MB

class PDFProcessor:
    """Handles PDF document processing and text extraction"""
    
    def __init__(self):
        self.max_file_size = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.chunk_size = MAX_CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        
    def process_pdf(self, uploaded_file) -> List[Dict[str, Any]]:
        """
        Process a single PDF file and return chunks
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of document chunks with metadata
        """
        try:
            # Check file size
            if uploaded_file.size > self.max_file_size:
                st.error(f"File {uploaded_file.name} is too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")
                return []
            
            # Extract text from PDF
            text = self._extract_text_from_pdf(uploaded_file)
            
            if not text.strip():
                st.warning(f"No text extracted from {uploaded_file.name}")
                return []
            
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            if not cleaned_text.strip():
                st.warning(f"No text remaining after cleaning from {uploaded_file.name}")
                return []
            
            # Create chunks
            chunks = self._create_chunks(cleaned_text, uploaded_file.name)
            
            if not chunks:
                st.warning(f"No chunks created from {uploaded_file.name}")
                return []
            
            st.success(f"Successfully processed {uploaded_file.name}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return []
    
    def _extract_text_from_pdf(self, uploaded_file) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            # Read PDF bytes
            pdf_bytes = uploaded_file.read()
            
            # Open PDF document
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Add page number for reference
                text += f"\n\n--- Page {page_num + 1} ---\n"
                text += page_text
            
            doc.close()
            return text
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text - ULTRA SIMPLE VERSION"""
        try:
            # Just do basic whitespace cleanup - NO REGEX!
            # Remove excessive whitespace using simple string operations
            text = ' '.join(text.split())
            
            # That's it! No fancy regex needed
            return text.strip()
            
        except Exception as e:
            st.warning(f"Error cleaning text: {str(e)}")
            return text  # Return original text if cleaning fails
    
    def _create_chunks(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Create overlapping chunks from text"""
        chunks = []
        
        # Split by sentences first for better chunk boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    # Create chunk
                    chunk = self._create_chunk_metadata(
                        current_chunk.strip(),
                        filename,
                        chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    current_chunk = self._get_overlap_text(current_chunk, sentence)
                else:
                    # Single sentence is too long, truncate it
                    current_chunk = sentence[:self.chunk_size]
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunk = self._create_chunk_metadata(
                current_chunk.strip(),
                filename,
                chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple string operations"""
        try:
            # Simple sentence splitting - NO REGEX!
            sentences = []
            current_sentence = ""
            
            for char in text:
                current_sentence += char
                
                # Check for sentence ending
                if char in '.!?':
                    # Look ahead to see if this is really the end
                    if len(current_sentence.strip()) > 10:  # Minimum sentence length
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
            
            # Add remaining text
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            
            return [s for s in sentences if len(s.strip()) > 10]
            
        except Exception as e:
            st.warning(f"Error splitting sentences: {str(e)}")
            return [text]  # Return original text as single sentence
    
    def _get_overlap_text(self, current_chunk: str, new_sentence: str) -> str:
        """Get overlap text for the next chunk"""
        if len(current_chunk) < self.chunk_overlap:
            return current_chunk + " " + new_sentence
        else:
            # Take the last part of current chunk for overlap
            words = current_chunk.split()
            overlap_words = words[-self.chunk_overlap//10:]  # Rough estimate
            return " ".join(overlap_words) + " " + new_sentence
    
    def _create_chunk_metadata(self, text: str, filename: str, chunk_index: int) -> Dict[str, Any]:
        """Create metadata for a chunk"""
        # Generate unique chunk ID
        chunk_id = hashlib.md5(f"{filename}_{chunk_index}_{text}".encode()).hexdigest()
        
        return {
            "id": chunk_id,
            "text": text,
            "source": filename,
            "chunk_index": chunk_index,
            "char_count": len(text),
            "word_count": len(text.split()),
            "type": "document",
            "metadata": {
                "filename": filename,
                "chunk_index": chunk_index,
                "processing_timestamp": str(Path(__file__).stat().st_mtime)
            }
        }
    
    def get_processing_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get processing statistics"""
        if not chunks:
            return {}
        
        total_chars = sum(chunk["char_count"] for chunk in chunks)
        total_words = sum(chunk["word_count"] for chunk in chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
            "sources": list(set(chunk["source"] for chunk in chunks))
        } 