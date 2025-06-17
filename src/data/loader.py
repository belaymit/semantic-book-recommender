"""
Data loading and preprocessing functions for the books dataset.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_books_data(filepath: str = "books.csv") -> pd.DataFrame:
    """
    Load the books dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file containing books data
        
    Returns:
        DataFrame containing the books data
    """
    try:
        logger.info(f"Loading books data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded {len(df)} books")
        return df
    except FileNotFoundError:
        logger.error(f"File {filepath} not found")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_books_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the books dataset.
    
    Args:
        df: Raw books DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning")
    
    # Create a copy to avoid modifying original data
    cleaned_df = df.copy()
    
    # Handle missing values
    logger.info("Handling missing values")
    
    # Fill missing numeric values with appropriate defaults
    numeric_columns = ['published_year', 'average_rating', 'num_pages', 'ratings_count']
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Fill missing categorical values
    categorical_columns = ['authors', 'categories', 'subtitle']
    for col in categorical_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    # Remove rows with missing essential information
    essential_columns = ['title', 'authors']
    for col in essential_columns:
        if col in cleaned_df.columns:
            cleaned_df = cleaned_df.dropna(subset=[col])
    
    # Remove duplicates
    logger.info("Removing duplicates")
    initial_len = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=['title', 'authors'])
    final_len = len(cleaned_df)
    logger.info(f"Removed {initial_len - final_len} duplicate records")
    
    # Clean text data
    logger.info("Cleaning text data")
    text_columns = ['title', 'authors', 'categories', 'description']
    for col in text_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
    
    logger.info(f"Data cleaning completed. Final dataset size: {len(cleaned_df)}")
    return cleaned_df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about the dataset.
    
    Args:
        df: Books DataFrame
        
    Returns:
        Dictionary containing dataset information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }
    
    return info


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.
    
    Args:
        df: Books DataFrame
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # Can be modified based on specific requirements
    )
    
    logger.info(f"Data split - Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df 