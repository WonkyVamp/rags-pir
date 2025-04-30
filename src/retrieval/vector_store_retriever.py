from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .base_retriever import BaseRetriever, RetrievedDocument

class VectorStoreRetriever(BaseRetriever):
    
    def __init__(self, config: Dict[str, Any]):
     
        super().__init__(config)
        self.model = SentenceTransformer(config.get("model_name", "all-MiniLM-L6-v2"))
        self.dimension = config.get("dimension", 384)
        self.index = None
        self.documents = []
        self._initialize_index()
        
    def _initialize_index(self):
        index_type = self.config.get("index_type", "L2")
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
            
        texts = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        sources = [doc.get("source", "unknown") for doc in documents]
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        self.index.add(np.array(embeddings).astype('float32'))
        
        for text, metadata, source in zip(texts, metadatas, sources):
            self.documents.append({
                "content": text,
                "metadata": metadata,
                "source": source
            })
            
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        if not self.documents:
            return []
            
        query_embedding = self.model.encode([query])
        
        scores, indices = self.index.search(
            np.array(query_embedding).astype('float32'),
            min(top_k, len(self.documents))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(RetrievedDocument(
                    content=doc["content"],
                    metadata=doc["metadata"],
                    score=float(score),
                    source=doc["source"]
                ))
                
        return results
    
    def clear(self) -> None:
        self._initialize_index()
        self.documents = []
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_documents": len(self.documents),
            "index_type": self.config.get("index_type", "L2"),
            "dimension": self.dimension,
            "model_name": self.config.get("model_name", "all-MiniLM-L6-v2")
        } 