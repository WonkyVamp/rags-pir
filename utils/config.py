import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("CONFIG_PATH", "config.json")
        self.config: Dict[str, Any] = {}
        self.load_config()
        
    def load_config(self) -> None:
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = self._get_default_config()
            self.save_config()
            
    def save_config(self) -> None:
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        self.config[key] = value
        self.save_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "retrieval": {
                "model_name": "all-MiniLM-L6-v2",
                "index_type": "L2",
                "dimension": 384,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "min_chunk_size": 100
            },
            "analysis": {
                "lookback_period": 252,
                "confidence_threshold": 0.7,
                "max_positions": 10
            },
            "reporting": {
                "output_dir": "reports",
                "template_dir": "templates",
                "visualization_dir": "visualizations"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "investment_system.log"
            }
        } 