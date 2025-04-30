from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RetrievedDocument:
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str

class BaseRetriever(ABC):
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:

        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass 