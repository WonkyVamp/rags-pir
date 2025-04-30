from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

class EnhancedDocumentRetriever:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.bm25 = None
        self.pinecone_index = None
        self.embeddings = OpenAIEmbeddings()
        self._initialize_retrievers()
        
    def _initialize_retrievers(self):
        tokenized_docs = [doc['content'].split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        
        index_name = "investment-docs"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
        
        self.pinecone_index = pinecone.Index(index_name)
        
        self._process_and_index_documents()
        
    def _process_and_index_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        for doc in self.documents:
            chunks = text_splitter.split_text(doc['content'])
            
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        'id': doc['id'],
                        'title': doc.get('title', ''),
                        'source': doc.get('source', ''),
                        'date': doc.get('date', ''),
                        'type': doc.get('type', '')
                    }
                )
                for chunk in chunks
            ]
            
            embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
            
            vectors = [
                (f"{doc.metadata['id']}_{i}", emb, doc.metadata)
                for i, (doc, emb) in enumerate(zip(documents, embeddings))
            ]
            
            self.pinecone_index.upsert(vectors=vectors)
            
    def _get_tfidf_scores(self, query: str) -> List[float]:
        """Get TF-IDF scores for documents"""
        query_vec = self.tfidf_vectorizer.fit_transform([query])
        doc_vecs = self.tfidf_vectorizer.transform([doc['content'] for doc in self.documents])
        return (query_vec * doc_vecs.T).toarray()[0]
    
    def _get_bm25_scores(self, query: str) -> List[float]:
        """Get BM25 scores for documents"""
        tokenized_query = query.split()
        return self.bm25.get_scores(tokenized_query)
    
    def _get_semantic_scores(self, query: str) -> List[float]:
        """Get semantic similarity scores using Pinecone"""
        query_embedding = self.embeddings.embed_query(query)
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=len(self.documents),
            include_metadata=True
        )
        
        score_map = {match.id: match.score for match in results.matches}
        
        return [score_map.get(f"{doc['id']}_0", 0.0) for doc in self.documents]
    
    def retrieve_documents(self, query: str, top_k: int = 5, 
                         weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        if weights is None:
            weights = {'tfidf': 0.33, 'bm25': 0.33, 'semantic': 0.34}
            
        tfidf_scores = self._get_tfidf_scores(query)
        bm25_scores = self._get_bm25_scores(query)
        semantic_scores = self._get_semantic_scores(query)
        
        def normalize_scores(scores):
            if not scores:
                return [0.0] * len(self.documents)
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [0.5] * len(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]
        
        tfidf_scores = normalize_scores(tfidf_scores)
        bm25_scores = normalize_scores(bm25_scores)
        semantic_scores = normalize_scores(semantic_scores)
        
        weighted_scores = [
            weights['tfidf'] * tfidf + 
            weights['bm25'] * bm25 + 
            weights['semantic'] * semantic
            for tfidf, bm25, semantic in zip(tfidf_scores, bm25_scores, semantic_scores)
        ]
        
        top_indices = np.argsort(weighted_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                **doc,
                'scores': {
                    'tfidf': tfidf_scores[idx],
                    'bm25': bm25_scores[idx],
                    'semantic': semantic_scores[idx],
                    'weighted': weighted_scores[idx]
                }
            })
            
        return results
    
    def retrieve_with_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.retrieve_documents(query, top_k)
        
        for result in results:
            query_embedding = self.embeddings.embed_query(result['content'])
            similar_chunks = self.pinecone_index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            result['context'] = [
                {
                    'content': match.metadata.get('content', ''),
                    'score': match.score
                }
                for match in similar_chunks.matches
            ]
            
        return results 