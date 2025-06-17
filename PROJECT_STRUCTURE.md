# Semantic Book Recommender - Project Structure

This document outlines the professional project structure for the semantic book recommender system.

## 📁 Directory Structure

```
semantic-book-recommender/
├── 📊 Data & Configuration
│   ├── books.csv                          # Original dataset (7k books from Kaggle)
│   └── config/
│       └── project_config.json            # Project configuration and parameters
│
├── 🔧 Source Code (Modular Functions)
│   └── src/
│       ├── __init__.py                     # Package initialization
│       ├── data/                           # Data handling modules
│       │   ├── __init__.py
│       │   └── loader.py                   # Data loading & cleaning functions
│       ├── analysis/                       # Analysis modules
│       │   ├── __init__.py
│       │   └── visualizer.py               # Visualization functions
│       └── utils/                          # Utility modules
│           ├── __init__.py
│           └── helpers.py                  # Helper functions & utilities
│
├── 📓 Analysis Notebooks
│   └── notebooks/
│       └── 01_exploratory_data_analysis.ipynb  # Main EDA notebook
│
├── 📋 Project Management
│   ├── requirements.txt                    # Python dependencies
│   ├── PROJECT_STRUCTURE.md               # This documentation
│   ├── README.md                          # Project overview
│   └── MIT.md                             # License information
│
└── 📦 Assets
    ├── murple_logo.png                     # Project branding
    └── app_screenshot.png                  # Application preview
```

## 🎯 Key Design Principles

### 1. **Modular Architecture**
- Each function has its own dedicated file
- Clear separation of concerns (data, analysis, utilities)
- Easy to test, maintain, and extend

### 2. **Professional Organization**
- Standard Python package structure
- Comprehensive documentation
- Configuration-driven approach

### 3. **Scalable Design**
- Modular functions can be easily imported and reused
- Configuration file for easy parameter management
- Clean separation between code and analysis

## 📊 Data Pipeline

1. **Raw Data** → `books.csv` (7k books from Kaggle)
2. **Loading** → `src/data/loader.py` functions
3. **Cleaning** → Preprocessing and validation
4. **Analysis** → `src/analysis/visualizer.py` functions
5. **Visualization** → Comprehensive charts in notebook

## 🔧 Core Modules

### `src/data/loader.py`
- `load_books_data()` - Load CSV with error handling
- `clean_books_data()` - Comprehensive data cleaning
- `get_data_info()` - Dataset information summary
- `split_data()` - Train/test splitting

### `src/analysis/visualizer.py`
- `plot_data_overview()` - Dataset overview visualization
- `plot_rating_analysis()` - Rating distribution and analysis
- `plot_publication_analysis()` - Publication trends over time
- `plot_category_analysis()` - Genre and category insights
- `plot_correlation_matrix()` - Correlation heatmap
- `create_summary_report()` - Comprehensive text summary

### `src/utils/helpers.py`
- `DataFrameProfiler` class - Comprehensive data profiling
- `print_dataframe_info()` - Detailed DataFrame inspection
- `setup_logging()` - Logging configuration
- `save_dataframe()` - Data export utilities
- Text processing and validation functions

## 📓 Analysis Notebook

### `01_exploratory_data_analysis.ipynb`
1. **Data Loading** - Import and initial inspection
2. **Data Overview** - Comprehensive profiling
3. **Data Cleaning** - Preprocessing pipeline
4. **Statistical Analysis** - Detailed statistical insights
5. **Visualizations** - Multiple chart types for different aspects
6. **Insights & Conclusions** - Key findings for recommendation system

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```bash
   jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
   ```

3. **Use Modular Functions**:
   ```python
   from src.data.loader import load_books_data, clean_books_data
   from src.analysis.visualizer import plot_rating_analysis
   
   df = load_books_data("books.csv")
   df_clean = clean_books_data(df)
   plot_rating_analysis(df_clean)
   ```

## 📈 Next Steps

This foundation provides:
- ✅ Clean, professional project structure
- ✅ Comprehensive data analysis
- ✅ Modular, reusable code
- ✅ Rich visualizations
- ✅ Documentation and configuration

Ready for semantic model development and recommendation system implementation! 