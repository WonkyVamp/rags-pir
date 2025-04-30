from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

class ValidationError(Exception):
    pass

def validate_numeric(value: Any, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        num_value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"Value must be numeric, got {type(value)}")
        
    if min_value is not None and num_value < min_value:
        raise ValidationError(f"Value must be >= {min_value}, got {num_value}")
        
    if max_value is not None and num_value > max_value:
        raise ValidationError(f"Value must be <= {max_value}, got {num_value}")
        
    return num_value

def validate_date(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
        
    try:
        if isinstance(value, str):
            return pd.to_datetime(value)
        elif isinstance(value, (int, float)):
            return pd.to_datetime(value, unit='s')
        else:
            raise ValidationError(f"Invalid date format: {value}")
    except Exception as e:
        raise ValidationError(f"Failed to parse date: {str(e)}")

def validate_dataframe(df: Any, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected DataFrame, got {type(df)}")
        
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
            
    return df

def validate_dict(data: Any, required_keys: Optional[List[str]] = None) -> Dict:
    if not isinstance(data, dict):
        raise ValidationError(f"Expected dictionary, got {type(data)}")
        
    if required_keys:
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValidationError(f"Missing required keys: {missing_keys}")
            
    return data

def validate_list(data: Any, min_length: Optional[int] = None, max_length: Optional[int] = None) -> List:
    if not isinstance(data, list):
        raise ValidationError(f"Expected list, got {type(data)}")
        
    if min_length is not None and len(data) < min_length:
        raise ValidationError(f"List must have at least {min_length} items, got {len(data)}")
        
    if max_length is not None and len(data) > max_length:
        raise ValidationError(f"List must have at most {max_length} items, got {len(data)}")
        
    return data

def validate_string(value: Any, min_length: Optional[int] = None, max_length: Optional[int] = None) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"Expected string, got {type(value)}")
        
    if min_length is not None and len(value) < min_length:
        raise ValidationError(f"String must have at least {min_length} characters, got {len(value)}")
        
    if max_length is not None and len(value) > max_length:
        raise ValidationError(f"String must have at most {max_length} characters, got {len(value)}")
        
    return value 