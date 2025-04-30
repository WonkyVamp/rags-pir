import spacy
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re

class CompanyInsightExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        
    def extract_company_entities(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE']:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
                
        return entities
    
    def extract_financial_entities(self, text: str) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        
        entities = []
        current_entity = None
        
        for i, (token, pred) in enumerate(zip(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), predictions[0])):
            if pred != 0: 
                if current_entity is None:
                    current_entity = {
                        'text': token,
                        'label': self.model.config.id2label[pred.item()],
                        'start': i,
                        'end': i
                    }
                else:
                    current_entity['text'] += ' ' + token
                    current_entity['end'] = i
            elif current_entity is not None:
                entities.append(current_entity)
                current_entity = None
                
        if current_entity is not None:
            entities.append(current_entity)
            
        return entities
    
    def extract_company_insights(self, text: str) -> Dict[str, Any]:
        company_entities = self.extract_company_entities(text)
        financial_entities = self.extract_financial_entities(text)
        
        insights = {
            'companies': [],
            'people': [],
            'locations': [],
            'financial_metrics': []
        }
        
        for entity in company_entities:
            if entity['label'] == 'ORG':
                insights['companies'].append(entity['text'])
            elif entity['label'] == 'PERSON':
                insights['people'].append(entity['text'])
            elif entity['label'] == 'GPE':
                insights['locations'].append(entity['text'])
                
        for entity in financial_entities:
            if 'MONEY' in entity['label'] or 'PERCENT' in entity['label']:
                insights['financial_metrics'].append({
                    'text': entity['text'],
                    'type': entity['label']
                })
                
        return insights
    
    def extract_company_relationships(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        relationships = []
        
        for sent in doc.sents:
            companies = [ent for ent in sent.ents if ent.label_ == 'ORG']
            if len(companies) >= 2:
                relationships.append({
                    'companies': [comp.text for comp in companies],
                    'context': sent.text
                })
                
        return relationships 