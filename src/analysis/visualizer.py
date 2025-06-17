"""
Visualization functions for books dataset analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def setup_plot_style():
    """Setup consistent plotting style."""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def plot_data_overview(df: pd.DataFrame) -> None:
    """
    Create an overview visualization of the dataset.
    
    Args:
        df: Books DataFrame
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dataset shape info
    axes[0, 0].text(0.1, 0.7, f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns", 
                    fontsize=16, transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.1, 0.5, f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", 
                    fontsize=14, transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.1, 0.3, f"Columns: {', '.join(df.columns[:5])}...", 
                    fontsize=12, transform=axes[0, 0].transAxes)
    axes[0, 0].set_title("Dataset Overview")
    axes[0, 0].axis('off')
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
    if len(missing_data) > 0:
        missing_data.plot(kind='barh', ax=axes[0, 1], color='coral')
        axes[0, 1].set_title("Missing Values by Column")
        axes[0, 1].set_xlabel("Number of Missing Values")
    else:
        axes[0, 1].text(0.5, 0.5, "No Missing Values", ha='center', va='center', 
                        transform=axes[0, 1].transAxes, fontsize=16)
        axes[0, 1].set_title("Missing Values")
        axes[0, 1].axis('off')
    
    # Data types
    dtype_counts = df.dtypes.value_counts()
    axes[1, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title("Data Types Distribution")
    
    # Sample of data
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table_data = df.head(3)[['title', 'authors', 'categories']].fillna('N/A')
    table = axes[1, 1].table(cellText=table_data.values, 
                            colLabels=table_data.columns,
                            cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 1].set_title("Sample Data")
    
    plt.tight_layout()
    plt.show()


def plot_rating_analysis(df: pd.DataFrame) -> None:
    """
    Analyze and visualize book ratings.
    
    Args:
        df: Books DataFrame
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Rating distribution
    if 'average_rating' in df.columns:
        df['average_rating'].hist(bins=30, ax=axes[0, 0], alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title("Distribution of Average Ratings")
        axes[0, 0].set_xlabel("Average Rating")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(df['average_rating'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["average_rating"].mean():.2f}')
        axes[0, 0].legend()
    
    # Ratings count distribution
    if 'ratings_count' in df.columns:
        # Log scale for better visualization
        ratings_count_log = np.log1p(df['ratings_count'].fillna(0))
        ratings_count_log.hist(bins=30, ax=axes[0, 1], alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title("Distribution of Ratings Count (Log Scale)")
        axes[0, 1].set_xlabel("Log(Ratings Count + 1)")
        axes[0, 1].set_ylabel("Frequency")
    
    # Rating vs Ratings Count
    if 'average_rating' in df.columns and 'ratings_count' in df.columns:
        sample_df = df.sample(n=min(1000, len(df)))  # Sample for better performance
        axes[1, 0].scatter(sample_df['ratings_count'], sample_df['average_rating'], 
                          alpha=0.6, color='purple')
        axes[1, 0].set_xlabel("Ratings Count")
        axes[1, 0].set_ylabel("Average Rating")
        axes[1, 0].set_title("Average Rating vs Ratings Count")
        axes[1, 0].set_xscale('log')
    
    # Top rated books
    if 'average_rating' in df.columns and 'title' in df.columns:
        top_rated = df.nlargest(10, 'average_rating')[['title', 'average_rating', 'authors']]
        y_pos = np.arange(len(top_rated))
        axes[1, 1].barh(y_pos, top_rated['average_rating'], color='gold')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([title[:30] + '...' if len(title) > 30 else title 
                                   for title in top_rated['title']], fontsize=8)
        axes[1, 1].set_xlabel("Average Rating")
        axes[1, 1].set_title("Top 10 Highest Rated Books")
    
    plt.tight_layout()
    plt.show()


def plot_publication_analysis(df: pd.DataFrame) -> None:
    """
    Analyze publication years and trends.
    
    Args:
        df: Books DataFrame
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    if 'published_year' in df.columns:
        # Clean publication year data
        pub_years = df['published_year'].dropna()
        pub_years = pub_years[(pub_years >= 1800) & (pub_years <= 2024)]  # Reasonable range
        
        # Publication year distribution
        pub_years.hist(bins=50, ax=axes[0, 0], alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 0].set_title("Distribution of Publication Years")
        axes[0, 0].set_xlabel("Publication Year")
        axes[0, 0].set_ylabel("Number of Books")
        
        # Books per decade
        decades = (pub_years // 10) * 10
        decade_counts = decades.value_counts().sort_index()
        decade_counts.plot(kind='bar', ax=axes[0, 1], color='teal')
        axes[0, 1].set_title("Books Published by Decade")
        axes[0, 1].set_xlabel("Decade")
        axes[0, 1].set_ylabel("Number of Books")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Publication trend over time
        yearly_counts = pub_years.value_counts().sort_index()
        yearly_counts.plot(ax=axes[1, 0], color='green', linewidth=2)
        axes[1, 0].set_title("Publication Trend Over Time")
        axes[1, 0].set_xlabel("Year")
        axes[1, 0].set_ylabel("Number of Books")
        
        # Recent publications (last 20 years)
        recent_years = pub_years[pub_years >= 2004]
        if len(recent_years) > 0:
            recent_counts = recent_years.value_counts().sort_index()
            recent_counts.plot(kind='bar', ax=axes[1, 1], color='orange')
            axes[1, 1].set_title("Publications in Recent Years (2004+)")
            axes[1, 1].set_xlabel("Year")
            axes[1, 1].set_ylabel("Number of Books")
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_category_analysis(df: pd.DataFrame) -> None:
    """
    Analyze book categories and genres.
    
    Args:
        df: Books DataFrame
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    if 'categories' in df.columns:
        # Clean and extract categories
        categories = df['categories'].dropna()
        # Split multiple categories and flatten
        all_categories = []
        for cat_string in categories:
            if isinstance(cat_string, str) and cat_string.lower() != 'unknown':
                # Split by common delimiters
                cats = cat_string.replace(',', ';').replace('&', ';').split(';')
                all_categories.extend([cat.strip() for cat in cats if cat.strip()])
        
        if all_categories:
            # Top categories
            category_counts = pd.Series(all_categories).value_counts().head(15)
            category_counts.plot(kind='barh', ax=axes[0, 0], color='lightblue')
            axes[0, 0].set_title("Top 15 Book Categories")
            axes[0, 0].set_xlabel("Number of Books")
            
            # Category distribution (pie chart for top 8)
            top_8_categories = category_counts.head(8)
            other_count = category_counts.iloc[8:].sum()
            if other_count > 0:
                pie_data = top_8_categories.copy()
                pie_data['Others'] = other_count
            else:
                pie_data = top_8_categories
                
            axes[0, 1].pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title("Category Distribution (Top Categories)")
    
    # Page count analysis
    if 'num_pages' in df.columns:
        pages = df['num_pages'].dropna()
        pages = pages[(pages > 0) & (pages < 2000)]  # Reasonable range
        
        if len(pages) > 0:
            pages.hist(bins=30, ax=axes[1, 0], alpha=0.7, color='lightsalmon', edgecolor='black')
            axes[1, 0].set_title("Distribution of Book Page Counts")
            axes[1, 0].set_xlabel("Number of Pages")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].axvline(pages.mean(), color='red', linestyle='--', 
                              label=f'Mean: {pages.mean():.0f}')
            axes[1, 0].legend()
    
    # Authors analysis
    if 'authors' in df.columns:
        authors = df['authors'].dropna()
        # Extract individual authors
        all_authors = []
        for author_string in authors:
            if isinstance(author_string, str) and author_string.lower() != 'unknown':
                # Split by common delimiters
                auth_list = author_string.replace(',', ';').replace('&', ';').split(';')
                all_authors.extend([auth.strip() for auth in auth_list if auth.strip()])
        
        if all_authors:
            # Top authors by book count
            author_counts = pd.Series(all_authors).value_counts().head(10)
            author_counts.plot(kind='barh', ax=axes[1, 1], color='lightgreen')
            axes[1, 1].set_title("Top 10 Most Prolific Authors")
            axes[1, 1].set_xlabel("Number of Books")
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plot correlation matrix for numeric variables.
    
    Args:
        df: Books DataFrame
    """
    setup_plot_style()
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title("Correlation Matrix of Numeric Variables")
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numeric variables for correlation analysis.")


def create_summary_report(df: pd.DataFrame) -> str:
    """
    Create a comprehensive summary report of the dataset.
    
    Args:
        df: Books DataFrame
        
    Returns:
        Summary report as string
    """
    report = []
    report.append("="*60)
    report.append("BOOKS DATASET ANALYSIS SUMMARY")
    report.append("="*60)
    
    # Basic info
    report.append(f"\n📊 DATASET OVERVIEW:")
    report.append(f"   • Total Books: {len(df):,}")
    report.append(f"   • Total Columns: {len(df.columns)}")
    report.append(f"   • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing data
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        report.append(f"\n❗ MISSING DATA:")
        for col, missing_count in missing_data[missing_data > 0].items():
            pct = (missing_count / len(df)) * 100
            report.append(f"   • {col}: {missing_count:,} ({pct:.1f}%)")
    
    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report.append(f"\n📈 NUMERIC VARIABLES SUMMARY:")
        for col in numeric_cols:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    report.append(f"   • {col}:")
                    report.append(f"     - Range: {data.min():.2f} to {data.max():.2f}")
                    report.append(f"     - Mean: {data.mean():.2f}")
                    report.append(f"     - Median: {data.median():.2f}")
    
    # Category analysis
    if 'categories' in df.columns:
        categories = df['categories'].dropna()
        unique_categories = set()
        for cat_string in categories:
            if isinstance(cat_string, str):
                cats = cat_string.replace(',', ';').replace('&', ';').split(';')
                unique_categories.update([cat.strip() for cat in cats if cat.strip()])
        
        report.append(f"\n📚 CATEGORIES:")
        report.append(f"   • Unique Categories: {len(unique_categories)}")
        report.append(f"   • Books with Categories: {len(categories):,}")
    
    # Publication years
    if 'published_year' in df.columns:
        pub_years = df['published_year'].dropna()
        pub_years = pub_years[(pub_years >= 1800) & (pub_years <= 2024)]
        if len(pub_years) > 0:
            report.append(f"\n📅 PUBLICATION YEARS:")
            report.append(f"   • Earliest: {pub_years.min():.0f}")
            report.append(f"   • Latest: {pub_years.max():.0f}")
            report.append(f"   • Most Common Decade: {(pub_years.mode().iloc[0] // 10) * 10:.0f}s")
    
    report.append("\n" + "="*60)
    
    return "\n".join(report) 