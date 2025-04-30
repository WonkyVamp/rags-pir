from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Any
import numpy as np

class RiskAssessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=3  # Low, Medium, High risk
        )
        
    def assess_risk(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
        risk_levels = ['Low', 'Medium', 'High']
        risk_scores = probabilities[0].numpy()
        
        return {
            'risk_level': risk_levels[np.argmax(risk_scores)],
            'confidence': float(np.max(risk_scores)),
            'risk_scores': {
                level: float(score)
                for level, score in zip(risk_levels, risk_scores)
            }
        }
    
    def analyze_risk_factors(self, text: str) -> List[Dict[str, Any]]:
        sentences = text.split('.')
        risk_factors = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            risk_assessment = self.assess_risk(sentence)
            
            if risk_assessment['risk_level'] != 'Low':
                risk_factors.append({
                    'text': sentence,
                    'risk_level': risk_assessment['risk_level'],
                    'confidence': risk_assessment['confidence']
                })
                
        return risk_factors
    
    def generate_risk_report(self, text: str) -> Dict[str, Any]:
        overall_risk = self.assess_risk(text)
        
        risk_factors = self.analyze_risk_factors(text)
        
        risk_metrics = {
            'high_risk_factors': len([f for f in risk_factors if f['risk_level'] == 'High']),
            'medium_risk_factors': len([f for f in risk_factors if f['risk_level'] == 'Medium']),
            'total_risk_factors': len(risk_factors)
        }
        
        return {
            'overall_risk': overall_risk,
            'risk_factors': risk_factors,
            'risk_metrics': risk_metrics,
            'recommendations': self._generate_recommendations(risk_metrics)
        }
    
    def _generate_recommendations(self, risk_metrics: Dict[str, int]) -> List[str]:
        recommendations = []
        
        if risk_metrics['high_risk_factors'] > 0:
            recommendations.append(
                "Consider implementing additional risk mitigation strategies for high-risk factors."
            )
            
        if risk_metrics['medium_risk_factors'] > 2:
            recommendations.append(
                "Monitor medium-risk factors closely and develop contingency plans."
            )
            
        if risk_metrics['total_risk_factors'] > 5:
            recommendations.append(
                "Consider diversifying investment portfolio to spread risk."
            )
            
        return recommendations 