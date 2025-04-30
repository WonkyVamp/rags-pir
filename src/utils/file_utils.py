import os
import json
import csv
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

def ensure_dir(directory: Union[str, Path]) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return json.load(f)

def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 4) -> None:
    path = Path(file_path)
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def write_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    path = Path(file_path)
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)

def read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    return pd.read_csv(file_path, **kwargs)

def write_csv(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    path = Path(file_path)
    ensure_dir(path.parent)
    df.to_csv(path, **kwargs)

def get_file_extension(file_path: Union[str, Path]) -> str:
    return Path(file_path).suffix.lower()

def is_valid_file(file_path: Union[str, Path], allowed_extensions: Optional[List[str]] = None) -> bool:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return False
        
    if allowed_extensions:
        return path.suffix.lower() in allowed_extensions
        
    return True

def list_files(directory: Union[str, Path], 
               pattern: str = "*", 
               recursive: bool = False) -> List[Path]:
    path = Path(directory)
    if recursive:
        return list(path.rglob(pattern))
    return list(path.glob(pattern)) 