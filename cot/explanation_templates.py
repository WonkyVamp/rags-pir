from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import json
from string import Template
import re
from enum import Enum


class ExplanationType(Enum):
    TECHNICAL = "technical"
    BUSINESS = "business"
    CUSTOMER = "customer"
    REGULATORY = "regulatory"


@dataclass
class TemplateContext:
    transaction_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    evidence: Dict[str, Any]
    risk_factors: List[Dict[str, Any]]
    confidence_level: float


class ExplanationTemplates:
    def __init__(self, config: Dict):
        self.config = config
        self.templates = self._initialize_templates()
        self.formatters = self._initialize_formatters()
        self.risk_thresholds = {"high": 0.8, "medium": 0.5, "low": 0.2}

    def _initialize_templates(self) -> Dict[str, Dict[str, Template]]:
        return {
            ExplanationType.TECHNICAL.value: {
                "header": Template(
                    "Technical Fraud Analysis Report\n"
                    "Transaction ID: ${transaction_id}\n"
                    "Analysis Timestamp: ${timestamp}\n"
                    "Risk Score: ${risk_score}\n"
                ),
                "evidence": Template(
                    "Evidence Analysis:\n"
                    "- Pattern Match Score: ${pattern_score}\n"
                    "- Network Analysis Score: ${network_score}\n"
                    "- Behavioral Score: ${behavioral_score}\n"
                    "- Anomaly Score: ${anomaly_score}\n"
                ),
                "reasoning": Template(
                    "Reasoning Chain:\n"
                    "${reasoning_steps}\n"
                    "Confidence Level: ${confidence}%\n"
                ),
                "recommendation": Template(
                    "Technical Recommendation:\n"
                    "Action: ${action}\n"
                    "Priority: ${priority}\n"
                    "Additional Monitoring: ${monitoring}\n"
                ),
            },
            ExplanationType.BUSINESS.value: {
                "header": Template(
                    "Fraud Review Summary\n"
                    "Review Date: ${date}\n"
                    "Alert Level: ${alert_level}\n"
                ),
                "summary": Template(
                    "Business Impact Summary:\n"
                    "Risk Level: ${risk_level}\n"
                    "Potential Impact: ${impact}\n"
                    "Key Concerns: ${concerns}\n"
                ),
                "action": Template(
                    "Recommended Business Actions:\n"
                    "1. ${primary_action}\n"
                    "2. ${secondary_action}\n"
                    "Expected Outcome: ${outcome}\n"
                ),
            },
            ExplanationType.CUSTOMER.value: {
                "header": Template("Transaction Security Notice\n" "Date: ${date}\n"),
                "explanation": Template(
                    "Dear valued customer,\n\n"
                    "We noticed unusual activity regarding your recent transaction:\n"
                    "${transaction_details}\n\n"
                    "Our security system identified the following concerns:\n"
                    "${security_concerns}\n"
                ),
                "action": Template(
                    "Recommended Steps:\n"
                    "${recommended_steps}\n\n"
                    "If you recognize this activity, no action is needed.\n"
                    "If you don't recognize this activity, please contact us immediately.\n"
                ),
            },
            ExplanationType.REGULATORY.value: {
                "header": Template(
                    "Regulatory Compliance Report\n"
                    "Report ID: ${report_id}\n"
                    "Filing Date: ${filing_date}\n"
                    "Institution: ${institution}\n"
                ),
                "analysis": Template(
                    "Compliance Analysis:\n"
                    "Regulation: ${regulation}\n"
                    "Risk Category: ${risk_category}\n"
                    "Severity Level: ${severity}\n"
                    "Supporting Evidence: ${evidence}\n"
                ),
                "filing": Template(
                    "Filing Details:\n"
                    "Filing Type: ${filing_type}\n"
                    "Priority: ${priority}\n"
                    "Required Action: ${required_action}\n"
                    "Documentation: ${documentation}\n"
                ),
            },
        }

    def _initialize_formatters(self) -> Dict[str, callable]:
        return {
            "amount": lambda x: f"${x:,.2f}",
            "timestamp": lambda x: datetime.fromisoformat(x).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "risk_score": lambda x: f"{x*100:.1f}%",
            "confidence": lambda x: f"{x*100:.1f}%",
            "location": lambda x: (
                f"({x['latitude']:.4f}, {x['longitude']:.4f})"
                if isinstance(x, dict)
                else str(x)
            ),
        }

    def _format_reasoning_steps(self, steps: List[Dict[str, Any]]) -> str:
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            formatted_step = (
                f"Step {i}:\n"
                f"Analysis: {step['description']}\n"
                f"Evidence: {step['evidence']}\n"
                f"Confidence: {self._format_value('confidence', step['confidence'])}\n"
                f"Conclusion: {step['conclusion']}\n"
            )
            formatted_steps.append(formatted_step)
        return "\n".join(formatted_steps)

    def _format_value(self, key: str, value: Any) -> str:
        formatter = self.formatters.get(key)
        if formatter:
            try:
                return formatter(value)
            except:
                return str(value)
        return str(value)

    def _extract_risk_factors(self, context: TemplateContext) -> List[str]:
        risk_factors = []

        if context.transaction_data.get("amount", 0) > self.config.get(
            "high_amount_threshold", 10000
        ):
            risk_factors.append("High transaction amount")

        if (
            context.analysis_results.get("velocity_score", 0)
            > self.risk_thresholds["high"]
        ):
            risk_factors.append("Unusual transaction velocity")

        if (
            context.analysis_results.get("location_score", 0)
            > self.risk_thresholds["high"]
        ):
            risk_factors.append("Suspicious location pattern")

        if (
            context.analysis_results.get("network_score", 0)
            > self.risk_thresholds["high"]
        ):
            risk_factors.append("Concerning network connections")

        return risk_factors

    def _generate_technical_explanation(self, context: TemplateContext) -> str:
        template_dict = self.templates[ExplanationType.TECHNICAL.value]

        header = template_dict["header"].substitute(
            transaction_id=context.transaction_data.get("transaction_id", "Unknown"),
            timestamp=self._format_value("timestamp", datetime.utcnow().isoformat()),
            risk_score=self._format_value("risk_score", context.confidence_level),
        )

        evidence = template_dict["evidence"].substitute(
            pattern_score=self._format_value(
                "risk_score", context.analysis_results.get("pattern_score", 0)
            ),
            network_score=self._format_value(
                "risk_score", context.analysis_results.get("network_score", 0)
            ),
            behavioral_score=self._format_value(
                "risk_score", context.analysis_results.get("behavioral_score", 0)
            ),
            anomaly_score=self._format_value(
                "risk_score", context.analysis_results.get("anomaly_score", 0)
            ),
        )

        reasoning = template_dict["reasoning"].substitute(
            reasoning_steps=self._format_reasoning_steps(
                context.evidence.get("steps", [])
            ),
            confidence=self._format_value("confidence", context.confidence_level),
        )

        risk_level = (
            "High"
            if context.confidence_level > self.risk_thresholds["high"]
            else (
                "Medium"
                if context.confidence_level > self.risk_thresholds["medium"]
                else "Low"
            )
        )

        recommendation = template_dict["recommendation"].substitute(
            action=f"{'Block' if risk_level == 'High' else 'Flag'} transaction",
            priority=risk_level,
            monitoring=f"Enhanced monitoring for {7 if risk_level == 'High' else 3} days",
        )

        return f"{header}\n{evidence}\n{reasoning}\n{recommendation}"

    def _generate_business_explanation(self, context: TemplateContext) -> str:
        template_dict = self.templates[ExplanationType.BUSINESS.value]
        risk_factors = self._extract_risk_factors(context)

        header = template_dict["header"].substitute(
            date=self._format_value("timestamp", datetime.utcnow().isoformat()),
            alert_level=(
                "High"
                if context.confidence_level > self.risk_thresholds["high"]
                else (
                    "Medium"
                    if context.confidence_level > self.risk_thresholds["medium"]
                    else "Low"
                )
            ),
        )

        impact = (
            "Significant"
            if context.confidence_level > self.risk_thresholds["high"]
            else (
                "Moderate"
                if context.confidence_level > self.risk_thresholds["medium"]
                else "Minor"
            )
        )

        summary = template_dict["summary"].substitute(
            risk_level=impact,
            impact=f"Potential financial impact: {self._format_value('amount', context.transaction_data.get('amount', 0))}",
            concerns=(
                ", ".join(risk_factors)
                if risk_factors
                else "No major concerns identified"
            ),
        )

        action = template_dict["action"].substitute(
            primary_action=(
                "Immediate review and escalation"
                if impact == "Significant"
                else "Standard review process"
            ),
            secondary_action=(
                "Enhanced monitoring"
                if impact == "Significant"
                else "Regular monitoring"
            ),
            outcome=(
                "Prevent potential fraud"
                if impact == "Significant"
                else "Ensure transaction legitimacy"
            ),
        )

        return f"{header}\n{summary}\n{action}"

    def _generate_customer_explanation(self, context: TemplateContext) -> str:
        template_dict = self.templates[ExplanationType.CUSTOMER.value]
        risk_factors = self._extract_risk_factors(context)

        header = template_dict["header"].substitute(
            date=self._format_value("timestamp", datetime.utcnow().isoformat())
        )

        transaction_details = (
            f"Amount: {self._format_value('amount', context.transaction_data.get('amount', 0))}\n"
            f"Date: {self._format_value('timestamp', context.transaction_data.get('timestamp', ''))}\n"
            f"Location: {self._format_value('location', context.transaction_data.get('location', ''))}"
        )

        explanation = template_dict["explanation"].substitute(
            transaction_details=transaction_details,
            security_concerns="\n".join(f"- {factor}" for factor in risk_factors),
        )

        steps = [
            "Review the transaction details",
            "Verify the location and amount",
            "Check your recent account activity",
        ]

        if context.confidence_level > self.risk_thresholds["high"]:
            steps.append("Contact our security team immediately")

        action = template_dict["action"].substitute(
            recommended_steps="\n".join(
                f"{i+1}. {step}" for i, step in enumerate(steps)
            )
        )

        return f"{header}\n{explanation}\n{action}"

    def _generate_regulatory_explanation(self, context: TemplateContext) -> str:
        template_dict = self.templates[ExplanationType.REGULATORY.value]

        header = template_dict["header"].substitute(
            report_id=f"REG-{hash(str(context.transaction_data))}",
            filing_date=self._format_value("timestamp", datetime.utcnow().isoformat()),
            institution=self.config.get("institution_name", "Unknown"),
        )

        severity = (
            "High"
            if context.confidence_level > self.risk_thresholds["high"]
            else (
                "Medium"
                if context.confidence_level > self.risk_thresholds["medium"]
                else "Low"
            )
        )

        analysis = template_dict["analysis"].substitute(
            regulation="Anti-Money Laundering (AML)",
            risk_category="Suspicious Transaction",
            severity=severity,
            evidence=json.dumps(context.evidence, indent=2),
        )

        filing = template_dict["filing"].substitute(
            filing_type="SAR" if severity == "High" else "Internal Review",
            priority=severity,
            required_action=(
                "Immediate Filing" if severity == "High" else "Standard Review"
            ),
            documentation="Full transaction history and analysis report",
        )

        return f"{header}\n{analysis}\n{filing}"

    def generate_explanation(
        self, context: TemplateContext, explanation_type: ExplanationType
    ) -> str:
        if explanation_type == ExplanationType.TECHNICAL:
            return self._generate_technical_explanation(context)
        elif explanation_type == ExplanationType.BUSINESS:
            return self._generate_business_explanation(context)
        elif explanation_type == ExplanationType.CUSTOMER:
            return self._generate_customer_explanation(context)
        elif explanation_type == ExplanationType.REGULATORY:
            return self._generate_regulatory_explanation(context)
        else:
            raise ValueError(f"Unsupported explanation type: {explanation_type}")

    def update_template(
        self, explanation_type: ExplanationType, template_key: str, new_template: str
    ):
        if explanation_type.value not in self.templates:
            raise ValueError(f"Invalid explanation type: {explanation_type}")

        if template_key not in self.templates[explanation_type.value]:
            raise ValueError(f"Invalid template key: {template_key}")

        self.templates[explanation_type.value][template_key] = Template(new_template)

    def add_formatter(self, key: str, formatter: callable):
        self.formatters[key] = formatter

    def get_available_templates(self) -> Dict[str, List[str]]:
        return {
            exp_type: list(templates.keys())
            for exp_type, templates in self.templates.items()
        }
