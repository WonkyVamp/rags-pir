from typing import Dict, List, Optional, Any
from datetime import datetime
import openai
import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
import tiktoken
import backoff
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class ModelType(Enum):
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT35_TURBO = "gpt-3.5-turbo"

@dataclass
class PromptTemplate:
    template: str
    required_variables: List[str]
    max_tokens: int
    model: ModelType

class OpenAIService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=config['openai_api_key'])
        self.logger = self._setup_logger()
        self.prompt_templates = self._initialize_prompt_templates()
        self.token_encoder = tiktoken.encoding_for_model(ModelType.GPT4.value)
        self.request_cache = {}
        self.max_retries = config.get('max_retries', 3)
        self.timeout = config.get('timeout', 30)
        
        # Initialize rate limiting parameters
        self.rate_limit_tokens = config.get('rate_limit_tokens', 100000)
        self.token_counter = 0
        self.last_reset = datetime.utcnow()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('openai_service')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def _initialize_prompt_templates(self) -> Dict[str, PromptTemplate]:
        return {
            'transaction_analysis': PromptTemplate(
                template="""
                Analyze the following transaction for potential fraud:
                Transaction Details:
                - Amount: ${amount}
                - Time: {timestamp}
                - Location: {location}
                - Customer History: {history}
                
                Consider the following aspects:
                1. Transaction characteristics
                2. Historical patterns
                3. Location analysis
                4. Behavioral consistency
                
                Provide a detailed analysis with:
                - Risk assessment
                - Supporting evidence
                - Confidence level
                - Recommended actions
                """,
                required_variables=['amount', 'timestamp', 'location', 'history'],
                max_tokens=500,
                model=ModelType.GPT4
            ),
            'pattern_explanation': PromptTemplate(
                template="""
                Explain the following fraud pattern:
                Pattern Type: {pattern_type}
                Pattern Details:
                {pattern_details}
                
                Detected Behaviors:
                {behaviors}
                
                Provide:
                1. Pattern explanation
                2. Risk implications
                3. Recommended monitoring approach
                4. Similar patterns to watch
                """,
                required_variables=['pattern_type', 'pattern_details', 'behaviors'],
                max_tokens=600,
                model=ModelType.GPT4
            ),
            'risk_assessment': PromptTemplate(
                template="""
                Assess the risk level for:
                Customer Profile:
                {customer_profile}
                
                Recent Activity:
                {recent_activity}
                
                Risk Indicators:
                {risk_indicators}
                
                Provide:
                1. Risk level assessment
                2. Key risk factors
                3. Mitigation recommendations
                4. Monitoring strategy
                """,
                required_variables=['customer_profile', 'recent_activity', 'risk_indicators'],
                max_tokens=700,
                model=ModelType.GPT4
            )
        }

    def _validate_prompt_variables(
        self,
        template_name: str,
        variables: Dict[str, Any]
    ) -> bool:
        template = self.prompt_templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")
            
        missing_vars = [
            var for var in template.required_variables
            if var not in variables
        ]
        
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
            
        return True

    def _count_tokens(self, text: str) -> int:
        return len(self.token_encoder.encode(text))

    async def _check_rate_limit(self):
        current_time = datetime.utcnow()
        if (current_time - self.last_reset).total_seconds() >= 3600:
            self.token_counter = 0
            self.last_reset = current_time
            
        if self.token_counter >= self.rate_limit_tokens:
            wait_time = 3600 - (current_time - self.last_reset).total_seconds()
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached. Waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
                self.token_counter = 0
                self.last_reset = datetime.utcnow()

    @retry(
        retry=retry_if_exception_type(openai.APIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_openai_request(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        try:
            await self._check_rate_limit()
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=self.timeout
            )
            
            token_usage = response.usage.total_tokens
            self.token_counter += token_usage
            
            return {
                'content': response.choices[0].message.content,
                'token_usage': token_usage,
                'model': model
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API request failed: {str(e)}")
            raise

    async def generate_analysis(
        self,
        template_name: str,
        variables: Dict[str, Any],
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            self._validate_prompt_variables(template_name, variables)
            template = self.prompt_templates[template_name]
            
            if cache_key and cache_key in self.request_cache:
                cached_response = self.request_cache[cache_key]
                if (datetime.utcnow() - cached_response['timestamp']).total_seconds() < 300:
                    return cached_response['response']
            
            prompt = template.template.format(**variables)
            messages = [
                {"role": "system", "content": "You are a fraud detection expert."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._make_openai_request(
                messages,
                template.model.value,
                template.max_tokens
            )
            
            if cache_key:
                self.request_cache[cache_key] = {
                    'response': response,
                    'timestamp': datetime.utcnow()
                }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Analysis generation failed: {str(e)}")
            raise

    async def analyze_transaction(
        self,
        transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        variables = {
            'amount': transaction_data['amount'],
            'timestamp': transaction_data['timestamp'],
            'location': f"{transaction_data.get('latitude', 0)}, {transaction_data.get('longitude', 0)}",
            'history': json.dumps(transaction_data.get('history', {}), indent=2)
        }
        
        return await self.generate_analysis(
            'transaction_analysis',
            variables,
            cache_key=f"txn_{transaction_data['transaction_id']}"
        )

    async def explain_pattern(
        self,
        pattern_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        variables = {
            'pattern_type': pattern_data['type'],
            'pattern_details': json.dumps(pattern_data['details'], indent=2),
            'behaviors': json.dumps(pattern_data['behaviors'], indent=2)
        }
        
        return await self.generate_analysis(
            'pattern_explanation',
            variables,
            cache_key=f"pattern_{hash(json.dumps(pattern_data))}"
        )

    async def assess_risk(
        self,
        customer_data: Dict[str, Any],
        activity_data: Dict[str, Any],
        risk_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        variables = {
            'customer_profile': json.dumps(customer_data, indent=2),
            'recent_activity': json.dumps(activity_data, indent=2),
            'risk_indicators': json.dumps(risk_data, indent=2)
        }
        
        return await self.generate_analysis(
            'risk_assessment',
            variables,
            cache_key=f"risk_{customer_data['customer_id']}_{datetime.utcnow().strftime('%Y%m%d')}"
        )

    def add_template(
        self,
        name: str,
        template: str,
        required_variables: List[str],
        max_tokens: int,
        model: ModelType
    ):
        self.prompt_templates[name] = PromptTemplate(
            template=template,
            required_variables=required_variables,
            max_tokens=max_tokens,
            model=model
        )

    def clear_cache(self):
        self.request_cache.clear()
        self.logger.info("Request cache cleared")

    def get_token_usage(self) -> Dict[str, int]:
        return {
            'current_tokens': self.token_counter,
            'limit': self.rate_limit_tokens,
            'remaining': self.rate_limit_tokens - self.token_counter
        }
