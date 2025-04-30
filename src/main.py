import os
from dotenv import load_dotenv
from data.data_loader import InvestmentDataLoader
from models.retrieval import EnhancedDocumentRetriever
from models.ner import CompanyInsightExtractor
from models.risk_assessment import RiskAssessor
from typing import Dict, Any, List
import json

load_dotenv()

class InvestmentAnalysisSystem:
    def __init__(self, data_dir: str):
        self.data_loader = InvestmentDataLoader(data_dir)
        self.retriever = None
        self.insight_extractor = CompanyInsightExtractor()
        self.risk_assessor = RiskAssessor()
        
    def initialize(self):
        documents = self.data_loader.load_investment_documents()
        processed_docs = self.data_loader.preprocess_documents(documents)
        
        self.retriever = EnhancedDocumentRetriever(processed_docs)
        
    def analyze_investment(self, query: str, weights: Dict[str, float] = None) -> Dict[str, Any]:
        relevant_docs = self.retriever.retrieve_with_context(query, weights=weights)
        
        company_insights = []
        for doc in relevant_docs:
            insights = self.insight_extractor.extract_company_insights(doc['content'])
            for context in doc.get('context', []):
                context_insights = self.insight_extractor.extract_company_insights(context['content'])
                insights['companies'].extend(context_insights['companies'])
                insights['people'].extend(context_insights['people'])
                insights['locations'].extend(context_insights['locations'])
                insights['financial_metrics'].extend(context_insights['financial_metrics'])
            company_insights.append(insights)
            
        risk_assessments = []
        for doc in relevant_docs:
            full_content = doc['content'] + "\n" + "\n".join(
                ctx['content'] for ctx in doc.get('context', [])
            )
            risk_report = self.risk_assessor.generate_risk_report(full_content)
            risk_assessments.append(risk_report)
            
        analysis_results = {
            'query': query,
            'relevant_documents': relevant_docs,
            'company_insights': company_insights,
            'risk_assessments': risk_assessments,
            'summary': self._generate_summary(company_insights, risk_assessments),
            'retrieval_scores': {
                doc['id']: doc['scores']
                for doc in relevant_docs
            }
        }
        
        return analysis_results
    
    def _generate_summary(self, company_insights: List[Dict[str, Any]], 
                         risk_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of analysis results"""
        combined_insights = {
            'companies': set(),
            'people': set(),
            'locations': set(),
            'financial_metrics': []
        }
        
        for insight in company_insights:
            combined_insights['companies'].update(insight['companies'])
            combined_insights['people'].update(insight['people'])
            combined_insights['locations'].update(insight['locations'])
            combined_insights['financial_metrics'].extend(insight['financial_metrics'])
            
        overall_risk_levels = [assessment['overall_risk']['risk_level'] 
                             for assessment in risk_assessments]
        most_common_risk = max(set(overall_risk_levels), key=overall_risk_levels.count)
        
        return {
            'key_companies': list(combined_insights['companies']),
            'key_people': list(combined_insights['people']),
            'key_locations': list(combined_insights['locations']),
            'financial_metrics': combined_insights['financial_metrics'],
            'overall_risk_level': most_common_risk,
            'recommendations': self._combine_recommendations(risk_assessments)
        }
    
    def _combine_recommendations(self, risk_assessments: List[Dict[str, Any]]) -> List[str]:
        all_recommendations = []
        for assessment in risk_assessments:
            all_recommendations.extend(assessment['recommendations'])
            
        return list(dict.fromkeys(all_recommendations))

def main():
    system = InvestmentAnalysisSystem("data/")
    system.initialize()
    
    query = "Analyze investment opportunities in technology sector"
    weights = {
        'tfidf': 0.3,    # Slightly lower weight for TF-IDF
        'bm25': 0.3,     # Equal weight for BM25
        'semantic': 0.4  # Higher weight for semantic search
    }
    
    results = system.analyze_investment(query, weights=weights)
    
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Analysis completed. Results saved to analysis_results.json")

if __name__ == "__main__":
    main() 