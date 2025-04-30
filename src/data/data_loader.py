import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import json

class InvestmentDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_investment_documents(self) -> List[Dict[str, Any]]:
        """
        Load investment documents from various sources.
        Returns a list of dictionaries containing document data.
        """
        documents = []
        
        # Load from JSON files
        for json_file in self.data_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                documents.extend(data)
                
        # Load from CSV files
        for csv_file in self.data_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            documents.extend(df.to_dict('records'))
            
        return documents
    
    def preprocess_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess documents for analysis.
        """
        processed_docs = []
        
        for doc in documents:
            processed_doc = {
                'id': doc.get('id', ''),
                'title': doc.get('title', ''),
                'content': doc.get('content', ''),
                'metadata': {
                    'source': doc.get('source', ''),
                    'date': doc.get('date', ''),
                    'type': doc.get('type', '')
                }
            }
            processed_docs.append(processed_doc)
            
        return processed_docs
    
    def get_document_chunks(self, documents: List[Dict[str, Any]], chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Split documents into smaller chunks for processing.
        """
        chunks = []
        
        for doc in documents:
            content = doc['content']
            words = content.split()
            
            for i in range(0, len(words), chunk_size):
                chunk = {
                    'id': f"{doc['id']}_chunk_{i//chunk_size}",
                    'content': ' '.join(words[i:i + chunk_size]),
                    'metadata': doc['metadata']
                }
                chunks.append(chunk)
                
        return chunks 