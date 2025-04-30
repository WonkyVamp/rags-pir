from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProcessedDocument:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str

class DocumentProcessor:
    
    def __init__(self, config: Dict[str, Any]):

        self.config = config
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.min_chunk_size = config.get("min_chunk_size", 100)
        
    def process_document(self, 
                        content: str,
                        metadata: Optional[Dict[str, Any]] = None,
                        source: Optional[str] = None) -> List[ProcessedDocument]:
       
        if not content:
            return []
            
        # Clean and normalize text
        content = self._clean_text(content)
        
        # Split into chunks
        chunks = self._split_into_chunks(content)
        
        # Create processed documents
        processed_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            processed_docs.append(ProcessedDocument(
                content=chunk,
                metadata=chunk_metadata,
                chunk_id=f"{source}_{i}" if source else f"chunk_{i}",
                source=source or "unknown"
            ))
            
        return processed_docs
    
    def _clean_text(self, text: str) -> str:
       
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        text = text.strip()
        
        return text
    
    def _split_into_chunks(self, text: str) -> List[str]:
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunk = text[start:]
                if len(chunk) >= self.min_chunk_size:
                    chunks.append(chunk)
                break
                
            # Try to find a good breaking point
            break_point = text.rfind(' ', start, end)
            if break_point == -1:
                break_point = end
                
            chunk = text[start:break_point]
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
                
            # Move start position for next chunk
            start = break_point - self.chunk_overlap
            if start < 0:
                start = 0
                
        return chunks
    
    def process_file(self, file_path: str) -> List[ProcessedDocument]:
      
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        content = path.read_text(encoding='utf-8')
        
        metadata = {
            "filename": path.name,
            "file_type": path.suffix,
            "file_size": path.stat().st_size
        }
        
        return self.process_document(content, metadata, str(path)) 