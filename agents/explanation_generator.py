from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import asyncio
import openai
import json
from string import Template
from collections import defaultdict
import numpy as np
from .base_agent import BaseAgent, AgentMessage


class ExplanationGenerator(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(
            agent_id=agent_id, agent_type="explanation_generator", config=config
        )
        self.openai_client = openai.AsyncOpenAI(api_key=config.get("openai_api_key"))
        self.explanation_cache = {}
        self.templates = self._initialize_templates()
        self.reasoning_chain = []
        self.explanation_types = {
            "transaction": self._explain_transaction,
            "pattern": self._explain_pattern,
            "anomaly": self._explain_anomaly,
            "risk": self._explain_risk,
        }

    def _initialize_templates(self) -> Dict[str, Template]:
        return {
            "transaction": Template(
                "Transaction Analysis:\n"
                "Amount: $$${amount}\n"
                "Time: ${time}\n"
                "Location: ${location}\n"
                "Key Factors: ${factors}\n"
                "Conclusion: ${conclusion}"
            ),
            "pattern": Template(
                "Pattern Detection:\n"
                "Type: ${pattern_type}\n"
                "Involved Transactions: ${transaction_count}\n"
                "Time Period: ${time_period}\n"
                "Key Observations: ${observations}\n"
                "Impact: ${impact}"
            ),
            "anomaly": Template(
                "Anomaly Detection:\n"
                "Detection Methods: ${methods}\n"
                "Confidence: ${confidence}%\n"
                "Primary Indicators: ${indicators}\n"
                "Supporting Evidence: ${evidence}"
            ),
            "risk": Template(
                "Risk Assessment:\n"
                "Overall Risk Level: ${risk_level}\n"
                "Risk Score: ${risk_score}\n"
                "Contributing Factors: ${factors}\n"
                "Recommended Actions: ${actions}"
            ),
        }

    async def initialize(self) -> bool:
        try:
            self.register_message_handler(
                "generate_explanation", self._handle_explanation_request
            )
            self.register_message_handler(
                "update_templates", self._handle_template_update
            )
            return True
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            return False

    async def _generate_cot_prompt(self, data: Dict[str, Any]) -> str:
        base_prompt = f"""
        Transaction Details:
        - Amount: ${data.get('amount', 0)}
        - Time: {data.get('timestamp', '')}
        - Location: {data.get('location', '')}
        
        Risk Factors:
        - Risk Score: {data.get('risk_score', 0)}
        - Anomaly Score: {data.get('anomaly_score', 0)}
        - Pattern Match: {data.get('pattern_match', False)}
        
        Analysis Steps:
        1. Evaluate transaction characteristics
        2. Consider historical patterns
        3. Assess anomaly indicators
        4. Determine risk implications
        5. Generate human-readable explanation
        
        Please provide a detailed explanation of why this transaction is flagged as suspicious:
        """
        return base_prompt

    async def _generate_openai_explanation(self, prompt: str) -> str:
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fraud detection expert explaining suspicious transactions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return None

    async def _explain_transaction(self, data: Dict[str, Any]) -> str:
        amount = data.get("amount", 0)
        timestamp = datetime.fromisoformat(
            data.get("timestamp", datetime.utcnow().isoformat())
        )
        location = f"{data.get('latitude', 0)}, {data.get('longitude', 0)}"

        factors = []
        if float(amount) > 1000:
            factors.append("High transaction amount")
        if timestamp.hour < 6 or timestamp.hour > 22:
            factors.append("Unusual transaction time")
        if data.get("velocity_24h", 0) > 5000:
            factors.append("High 24-hour velocity")

        return self.templates["transaction"].substitute(
            amount=amount,
            time=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            location=location,
            factors=", ".join(factors),
            conclusion=(
                "Transaction shows multiple risk indicators"
                if factors
                else "Transaction appears normal"
            ),
        )

    async def _explain_pattern(self, data: Dict[str, Any]) -> str:
        patterns = data.get("patterns", {})
        if not patterns:
            return "No significant patterns detected."

        pattern_types = list(patterns.keys())
        total_transactions = sum(
            len(p.get("transactions", [])) for p in patterns.values()
        )

        time_ranges = []
        for pattern in patterns.values():
            timestamps = [
                datetime.fromisoformat(t) for t in pattern.get("timestamps", [])
            ]
            if timestamps:
                time_range = max(timestamps) - min(timestamps)
                time_ranges.append(time_range)

        avg_time_range = (
            sum((r.total_seconds() for r in time_ranges), 0) / len(time_ranges)
            if time_ranges
            else 0
        )

        return self.templates["pattern"].substitute(
            pattern_type=", ".join(pattern_types),
            transaction_count=total_transactions,
            time_period=f"{avg_time_range/3600:.1f} hours",
            observations=self._summarize_patterns(patterns),
            impact=(
                "High risk patterns detected"
                if total_transactions > 3
                else "Moderate risk patterns observed"
            ),
        )

    async def _explain_anomaly(self, data: Dict[str, Any]) -> str:
        detection_results = data.get("detection_results", {})
        anomaly_score = data.get("anomaly_score", 0)

        methods = []
        evidence = []

        for method, result in detection_results.items():
            if result.get("is_anomaly"):
                methods.append(method)
                score = result.get("score", 0)
                evidence.append(f"{method}: {score:.3f}")

        confidence = int(anomaly_score * 100)
        indicators = self._extract_anomaly_indicators(detection_results)

        return self.templates["anomaly"].substitute(
            methods=", ".join(methods),
            confidence=confidence,
            indicators=", ".join(indicators),
            evidence="; ".join(evidence),
        )

    async def _explain_risk(self, data: Dict[str, Any]) -> str:
        risk_score = data.get("risk_score", 0)
        component_scores = data.get("component_scores", {})

        risk_level = (
            "High" if risk_score > 0.7 else "Medium" if risk_score > 0.4 else "Low"
        )

        factors = [f"{k}: {v:.2f}" for k, v in component_scores.items()]

        actions = self._generate_recommendations(risk_score, component_scores)

        return self.templates["risk"].substitute(
            risk_level=risk_level,
            risk_score=f"{risk_score:.2f}",
            factors=", ".join(factors),
            actions="; ".join(actions),
        )

    def _summarize_patterns(self, patterns: Dict[str, Any]) -> str:
        summaries = []
        for pattern_type, pattern_data in patterns.items():
            if pattern_type == "velocity":
                avg_time_diff = np.mean([p["time_difference"] for p in pattern_data])
                summaries.append(
                    f"Average time between transactions: {avg_time_diff:.1f}s"
                )
            elif pattern_type == "location":
                locations = len(pattern_data)
                summaries.append(f"Suspicious activity in {locations} locations")

        return "; ".join(summaries)

    def _extract_anomaly_indicators(
        self, detection_results: Dict[str, Any]
    ) -> List[str]:
        indicators = []
        for method, result in detection_results.items():
            if result.get("is_anomaly"):
                if method == "zscore":
                    max_zscore = max(abs(score) for score in result.get("scores", []))
                    indicators.append(f"Statistical outlier (z={max_zscore:.2f})")
                elif method == "isolation_forest":
                    indicators.append("Isolated transaction pattern")
                elif method == "lof":
                    indicators.append("Local density anomaly")

        return indicators

    def _generate_recommendations(
        self, risk_score: float, component_scores: Dict[str, float]
    ) -> List[str]:
        actions = []
        if risk_score > 0.7:
            actions.append("Immediate manual review required")
            actions.append("Temporarily restrict account activity")
        elif risk_score > 0.4:
            actions.append("Flag for supervisor review")
            actions.append("Enhanced monitoring for 24 hours")
        else:
            actions.append("Standard monitoring")

        if component_scores.get("pattern_score", 0) > 0.6:
            actions.append("Investigate linked transactions")

        return actions

    async def _handle_explanation_request(
        self, message: AgentMessage
    ) -> Dict[str, Any]:
        data = message.content
        success, result = await self.process(data)

        if success:
            return {"status": "success", "explanation": result}
        return {"status": "error", "message": "Explanation generation failed"}

    async def _handle_template_update(self, message: AgentMessage) -> Dict[str, Any]:
        new_templates = message.content.get("templates", {})
        for template_type, template_str in new_templates.items():
            self.templates[template_type] = Template(template_str)
        return {"status": "success", "message": "Templates updated"}

    async def process(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        try:
            explanation_type = data.get("type", "transaction")
            explanation_data = data.get("data", {})

            handler = self.explanation_types.get(explanation_type)
            if not handler:
                return False, {"error": f"Unknown explanation type: {explanation_type}"}

            base_explanation = await handler(explanation_data)

            if data.get("use_cot", False):
                prompt = await self._generate_cot_prompt(explanation_data)
                cot_explanation = await self._generate_openai_explanation(prompt)
            else:
                cot_explanation = None

            result = {
                "base_explanation": base_explanation,
                "cot_explanation": cot_explanation,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "explanation_type": explanation_type,
                    "template_used": str(self.templates[explanation_type]),
                    "cot_used": bool(cot_explanation),
                },
            }

            return True, result

        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return False, {"error": str(e)}

    async def cleanup(self) -> bool:
        try:
            self.explanation_cache.clear()
            self.reasoning_chain.clear()
            return True
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
            return False
