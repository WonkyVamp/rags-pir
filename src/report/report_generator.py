from typing import Dict, List, Any
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader

class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(template_dir))
        
    def generate_report(self, 
                       analysis_results: Dict[str, Any],
                       backtest_results: Dict[str, Any],
                       risk_assessment: Dict[str, Any],
                       report_type: str = "full") -> str:
  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"investment_report_{timestamp}.html"
        
        # Prepare report data
        report_data = {
            "timestamp": timestamp,
            "analysis_results": analysis_results,
            "backtest_results": backtest_results,
            "risk_assessment": risk_assessment,
            "report_type": report_type
        }
        
        self._generate_visualizations(report_data)
        
        template = self.env.get_template(f"{report_type}_report.html")
        
        report_content = template.render(**report_data)
        
        with open(report_path, "w") as f:
            f.write(report_content)
            
        return str(report_path)
    
    def _generate_visualizations(self, report_data: Dict[str, Any]) -> None:
 
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Generate performance charts
        if "backtest_results" in report_data:
            self._plot_performance_metrics(report_data["backtest_results"], viz_dir)
            
        # Generate risk metrics charts
        if "risk_assessment" in report_data:
            self._plot_risk_metrics(report_data["risk_assessment"], viz_dir)
            
    def _plot_performance_metrics(self, backtest_results: Dict[str, Any], viz_dir: Path) -> None:
        
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_results.get("cumulative_returns", []))
        plt.title("Cumulative Returns")
        plt.xlabel("Time")
        plt.ylabel("Returns")
        plt.savefig(viz_dir / "cumulative_returns.png")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_results.get("drawdown", []))
        plt.title("Portfolio Drawdown")
        plt.xlabel("Time")
        plt.ylabel("Drawdown %")
        plt.savefig(viz_dir / "drawdown.png")
        plt.close()
        
    def _plot_risk_metrics(self, risk_assessment: Dict[str, Any], viz_dir: Path) -> None:
        metrics = ["volatility", "var_95", "var_99"]
        plt.figure(figsize=(12, 6))
        
        for metric in metrics:
            if metric in risk_assessment:
                plt.plot(risk_assessment[metric], label=metric)
                
        plt.title("Risk Metrics Over Time")
        plt.xlabel("Time")
        plt.ylabel("Risk Value")
        plt.legend()
        plt.savefig(viz_dir / "risk_metrics.png")
        plt.close() 