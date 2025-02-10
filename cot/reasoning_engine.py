from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import openai
import json
from collections import defaultdict
import asyncio


@dataclass
class ReasoningStep:
    step_id: str
    description: str
    evidence: Dict[str, Any]
    confidence: float
    dependencies: List[str]
    conclusion: str
    timestamp: datetime


@dataclass
class ReasoningChain:
    chain_id: str
    steps: List[ReasoningStep]
    final_conclusion: str
    confidence_score: float
    supporting_evidence: Dict[str, Any]
    metadata: Dict[str, Any]


class ReasoningEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.openai_client = openai.AsyncOpenAI(api_key=config.get("openai_api_key"))
        self.reasoning_cache = {}
        self.evidence_weights = {
            "transaction_pattern": 0.3,
            "historical_behavior": 0.25,
            "network_analysis": 0.25,
            "anomaly_detection": 0.2,
        }
        self.confidence_thresholds = {"high": 0.8, "medium": 0.6, "low": 0.4}

    async def _generate_cot_prompt(self, evidence: Dict[str, Any]) -> str:
        transaction = evidence.get("transaction", {})
        patterns = evidence.get("patterns", {})
        history = evidence.get("history", {})

        prompt = f"""
        Analyze the following transaction for potential fraud:
        
        Transaction Details:
        - Amount: ${transaction.get('amount', 0)}
        - Time: {transaction.get('timestamp', '')}
        - Location: {transaction.get('location', '')}
        - Customer History: {len(history.get('previous_transactions', []))} previous transactions
        
        Detected Patterns:
        {json.dumps(patterns, indent=2)}
        
        Risk Indicators:
        - Transaction Risk: {transaction.get('risk_score', 0)}
        - Pattern Risk: {patterns.get('risk_score', 0)}
        - Historical Risk: {history.get('risk_score', 0)}
        
        Please analyze this transaction step by step, considering:
        1. Individual transaction characteristics
        2. Pattern matching and anomalies
        3. Historical behavior comparison
        4. Network relationships
        5. Final risk assessment
        
        For each step, provide:
        - Detailed analysis
        - Supporting evidence
        - Confidence level
        - Impact on final conclusion
        """
        return prompt

    async def _analyze_with_llm(self, prompt: str) -> Dict[str, Any]:
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fraud detection expert analyzing transactions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            analysis = response.choices[0].message.content
            return self._parse_llm_response(analysis)

        except Exception as e:
            raise RuntimeError(f"LLM analysis failed: {str(e)}")

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        steps = []
        current_step = {}
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("Step "):
                if current_step:
                    steps.append(current_step)
                current_step = {"description": line}
            elif line.startswith("Evidence:"):
                current_step["evidence"] = line[9:].strip()
            elif line.startswith("Confidence:"):
                current_step["confidence"] = line[11:].strip()
            elif line.startswith("Conclusion:"):
                current_step["conclusion"] = line[11:].strip()

        if current_step:
            steps.append(current_step)

        return {"steps": steps}

    def _evaluate_evidence(
        self, evidence: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        scores = {}

        for category, weight in self.evidence_weights.items():
            if category in evidence:
                category_score = self._calculate_category_score(evidence[category])
                scores[category] = category_score * weight

        total_score = sum(scores.values()) / sum(self.evidence_weights.values())
        return total_score, scores

    def _calculate_category_score(self, category_evidence: Dict[str, Any]) -> float:
        if isinstance(category_evidence, dict):
            if "risk_score" in category_evidence:
                return float(category_evidence["risk_score"])
            elif "score" in category_evidence:
                return float(category_evidence["score"])

        return 0.0

    def _build_reasoning_chain(
        self, llm_analysis: Dict[str, Any], evidence: Dict[str, Any]
    ) -> ReasoningChain:
        steps = []
        evidence_score, category_scores = self._evaluate_evidence(evidence)

        for idx, step_data in enumerate(llm_analysis["steps"]):
            step = ReasoningStep(
                step_id=f"step_{idx}",
                description=step_data["description"],
                evidence={"category_scores": category_scores},
                confidence=self._parse_confidence(step_data.get("confidence", "")),
                dependencies=[f"step_{i}" for i in range(idx)],
                conclusion=step_data.get("conclusion", ""),
                timestamp=datetime.utcnow(),
            )
            steps.append(step)

        final_conclusion = steps[-1].conclusion if steps else "No conclusion"

        chain = ReasoningChain(
            chain_id=str(hash(str(evidence))),
            steps=steps,
            final_conclusion=final_conclusion,
            confidence_score=evidence_score,
            supporting_evidence=evidence,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "num_steps": len(steps),
                "evidence_weights": self.evidence_weights,
            },
        )

        return chain

    def _parse_confidence(self, confidence_str: str) -> float:
        try:
            if "%" in confidence_str:
                return float(confidence_str.strip("%")) / 100
            return float(confidence_str)
        except:
            return 0.5

    def _validate_reasoning_chain(self, chain: ReasoningChain) -> bool:
        if not chain.steps:
            return False

        step_ids = {step.step_id for step in chain.steps}

        for step in chain.steps:
            if not all(dep in step_ids for dep in step.dependencies):
                return False

            if step.confidence < 0 or step.confidence > 1:
                return False

        return True

    async def analyze_transaction(
        self, transaction_data: Dict[str, Any], evidence: Dict[str, Any]
    ) -> ReasoningChain:
        cache_key = str(hash(str(transaction_data)))

        if cache_key in self.reasoning_cache:
            cached_result = self.reasoning_cache[cache_key]
            if datetime.utcnow() - cached_result.steps[-1].timestamp < timedelta(
                hours=1
            ):
                return cached_result

        prompt = await self._generate_cot_prompt(
            {"transaction": transaction_data, **evidence}
        )

        llm_analysis = await self._analyze_with_llm(prompt)
        reasoning_chain = self._build_reasoning_chain(llm_analysis, evidence)

        if self._validate_reasoning_chain(reasoning_chain):
            self.reasoning_cache[cache_key] = reasoning_chain
            return reasoning_chain
        else:
            raise ValueError("Invalid reasoning chain generated")

    async def explain_reasoning(self, chain: ReasoningChain) -> str:
        explanation = ["Fraud Analysis Reasoning:"]

        for step in chain.steps:
            explanation.append(f"\nStep {step.step_id}:")
            explanation.append(f"Analysis: {step.description}")
            explanation.append(f"Evidence: {json.dumps(step.evidence, indent=2)}")
            explanation.append(f"Confidence: {step.confidence:.2%}")
            explanation.append(f"Conclusion: {step.conclusion}")

        explanation.append(f"\nFinal Conclusion: {chain.final_conclusion}")
        explanation.append(f"Overall Confidence: {chain.confidence_score:.2%}")

        return "\n".join(explanation)

    def compare_reasoning_chains(
        self, chain1: ReasoningChain, chain2: ReasoningChain
    ) -> Dict[str, Any]:
        comparison = {
            "confidence_diff": abs(chain1.confidence_score - chain2.confidence_score),
            "step_count_diff": abs(len(chain1.steps) - len(chain2.steps)),
            "conclusion_match": chain1.final_conclusion == chain2.final_conclusion,
            "evidence_overlap": self._calculate_evidence_overlap(
                chain1.supporting_evidence, chain2.supporting_evidence
            ),
        }
        return comparison

    def _calculate_evidence_overlap(
        self, evidence1: Dict[str, Any], evidence2: Dict[str, Any]
    ) -> float:
        keys1 = set(evidence1.keys())
        keys2 = set(evidence2.keys())

        intersection = len(keys1 & keys2)
        union = len(keys1 | keys2)

        return intersection / union if union > 0 else 0.0

    def get_chain_summary(self, chain: ReasoningChain) -> Dict[str, Any]:
        return {
            "chain_id": chain.chain_id,
            "num_steps": len(chain.steps),
            "confidence": chain.confidence_score,
            "conclusion": chain.final_conclusion,
            "timestamp": chain.metadata["timestamp"],
            "key_evidence": sorted(
                chain.supporting_evidence.keys(),
                key=lambda k: chain.supporting_evidence.get(k, {}).get("risk_score", 0),
                reverse=True,
            )[:3],
        }
