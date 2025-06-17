"""
General utility functions for the semantic book recommender project.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Any, Optional
import json


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def create_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def save_dataframe(df: pd.DataFrame, filepath: str, format: str = 'csv') -> None:
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        format: File format ('csv', 'parquet', 'json')
    """
    create_directory(os.path.dirname(filepath))
    
    if format.lower() == 'csv':
        df.to_csv(filepath, index=False)
    elif format.lower() == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format.lower() == 'json':
        df.to_json(filepath, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    create_directory(os.path.dirname(config_path))
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def clean_text(text: str) -> str:
    """
    Clean text data for processing.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters if needed
    # text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()


def get_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get detailed memory usage information for DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    memory_usage = df.memory_usage(deep=True)
    
    return {
        'total_mb': memory_usage.sum() / 1024**2,
        'per_column_mb': (memory_usage / 1024**2).to_dict(),
        'dtypes': df.dtypes.to_dict()
    }


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print comprehensive information about a DataFrame.
    
    Args:
        df: DataFrame to analyze
        name: Name for the DataFrame
    """
    print(f"\n{'='*50}")
    print(f"{name.upper()} INFORMATION")
    print(f"{'='*50}")
    
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nColumn Information:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"  {col:20s} | {str(dtype):10s} | {null_count:5d} nulls ({null_pct:5.1f}%)")
    
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    
    return True


def sample_dataframe(df: pd.DataFrame, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Sample DataFrame for faster processing during development.
    
    Args:
        df: DataFrame to sample
        n_samples: Number of samples to take
        random_state: Random seed
        
    Returns:
        Sampled DataFrame
    """
    if len(df) <= n_samples:
        return df
    
    return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)


def calculate_text_stats(text_series: pd.Series) -> Dict[str, float]:
    """
    Calculate statistics for text data.
    
    Args:
        text_series: Series containing text data
        
    Returns:
        Dictionary with text statistics
    """
    # Remove null values
    clean_text = text_series.dropna().astype(str)
    
    if len(clean_text) == 0:
        return {}
    
    # Calculate word counts
    word_counts = clean_text.str.split().str.len()
    
    # Calculate character counts
    char_counts = clean_text.str.len()
    
    stats = {
        'total_entries': len(clean_text),
        'avg_words': word_counts.mean(),
        'median_words': word_counts.median(),
        'max_words': word_counts.max(),
        'min_words': word_counts.min(),
        'avg_chars': char_counts.mean(),
        'median_chars': char_counts.median(),
        'max_chars': char_counts.max(),
        'min_chars': char_counts.min()
    }
    
    return stats


class DataFrameProfiler:
    """Class for comprehensive DataFrame profiling."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profile = {}
    
    def generate_profile(self) -> Dict[str, Any]:
        """Generate comprehensive profile of the DataFrame."""
        self.profile = {
            'shape': self.df.shape,
            'memory_usage': get_memory_usage(self.df),
            'missing_data': self._analyze_missing_data(),
            'data_types': self._analyze_data_types(),
            'numeric_summary': self._analyze_numeric_data(),
            'text_summary': self._analyze_text_data(),
            'duplicates': self._analyze_duplicates()
        }
        
        return self.profile
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        return {
            'total_missing': missing_counts.sum(),
            'missing_by_column': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'complete_rows': len(self.df) - self.df.isnull().any(axis=1).sum()
        }
    
    def _analyze_data_types(self) -> Dict[str, Any]:
        """Analyze data types distribution."""
        dtype_counts = self.df.dtypes.value_counts()
        
        return {
            'type_distribution': dtype_counts.to_dict(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'text_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': self.df.select_dtypes(include=['datetime']).columns.tolist()
        }
    
    def _analyze_numeric_data(self) -> Dict[str, Any]:
        """Analyze numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return {}
        
        return {
            'summary_stats': numeric_df.describe().to_dict(),
            'correlations': numeric_df.corr().to_dict() if len(numeric_df.columns) > 1 else {}
        }
    
    def _analyze_text_data(self) -> Dict[str, Any]:
        """Analyze text columns."""
        text_columns = self.df.select_dtypes(include=['object']).columns
        text_analysis = {}
        
        for col in text_columns:
            text_analysis[col] = calculate_text_stats(self.df[col])
        
        return text_analysis
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate records."""
        return {
            'total_duplicates': self.df.duplicated().sum(),
            'duplicate_percentage': (self.df.duplicated().sum() / len(self.df)) * 100,
            'unique_rows': len(self.df.drop_duplicates())
        } 