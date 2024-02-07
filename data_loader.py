import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import datetime
import mimetypes  # Add for MIME-based inference
import filetype  # Add for content-based inference



def validate_file_path(file_path):
    """
    Validates the user-provided file path.

    Args:
        file_path (str): The file path to validate.

    Returns:
        bool: True if the file path is valid, False otherwise.
    """

    # Check if the file path is empty
    if not file_path:
        print("File path cannot be empty.")
        return False

    # Check if the file path is absolute
    if not os.path.isabs(file_path):
        print("File path must be absolute.")
        return False

    # Check if the file path exists
    if not os.path.exists(file_path):
        print("File path does not exist.")
        return False

    # Check if the file path is a file
    if not os.path.isfile(file_path):
        print("File path is not a file.")
        return False

    # If all checks pass, return True
    return True

def load_data(file_path):
    """
    Loads the dfset from the user's input file path, supporting CSV, Excel, JSON, and potentially more.

    Args:
        file_path (str): The file path of the dfset.

    Returns:
        pd.dataFrame: The loaded dfFrame.

    Raises:
        ValueError: If the file format is not supported or the file path is invalid.
    """
 # Attempt file format inference using multiple methods:
    file_format = try_infer_format(file_path)

    if file_format in ('csv', 'xlsx', 'json'):
        try:
            if file_format == 'csv':
                df = pd.read_csv(file_path)
            elif file_format == 'xlsx':
                df = pd.read_excel(file_path)
            elif file_format == 'json':
                df = pd.read_json(file_path)
        except pd.errors.EmptydfError:
            raise ValueError(f"File '{file_path}' is empty.")
    else:
        raise ValueError(f'Unsupported file format: {file_format}. Please use CSV, Excel, JSON, or provide the format manually.')

    if df.empty:
        raise ValueError(f"File '{file_path}' contains no data.")  # Check for empty data

    return df

def try_infer_format(file_path):
    """
    Attempts to infer the file format using multiple methods.
    """

    # 1. Use MIME-based inference:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        file_format = mime_type.split('/')[1]
        return file_format

    # 2. Use content-based inference:
    kind = filetype.guess(file_path)
    if kind:
        return kind.extension

    # 3. Fallback to extension-based method:
    return file_path.split('.')[-1].lower()

def inspect_data(df, columns_for_corr=None, columns_for_skewness=None):
    """
    This function inspects a pandas dfFrame and prints various summary statistics,
    data quality checks, and visualizations. Users can optionally specify columns
    for correlation analysis and skewness checks.

    Args:
        data (pd.dfFrame): The dfFrame to inspect.
        columns_for_corr (list, optional): List of column names for correlation analysis.
        columns_for_skewness (list, optional): List of column names for skewness checks.

    Output:
        None. The function only prints the results to the console.
    """

    # Step 1: Print basic information
    print_basic_info(df)

    # Step 2: Check for missing values
    check_missing_values(df)

    # Step 3: Print summary statistics for numerical columns
    print_summary_stats(df)

    # Step 4: Check for duplicate rows
    check_duplicate_rows(df)

    # Step 5: Check for outliers using z-scores and box plots
    check_outliers(df)

    # Step 6: Check for df ranges or patterns (customizable)
    check_df_ranges(df)

    # Step 7: Perform correlation analysis (optional)
    if columns_for_corr:
        perform_corr_analysis(df, columns_for_corr)

    # Step 8: Check for skewness (optional)
    if columns_for_skewness:
        check_skewness(df, columns_for_skewness)

    # Step 9: Perform visual exploration
    perform_visual_exploration(df)

    # Step 10: Perform time series analysis (placeholder)
    perform_time_series_analysis(df)

def print_basic_info(df):
    """
    Prints basic information about the df.

    Args:
        df (pd.dfFrame): The dfFrame to inspect.
    """

    print(f"The data has {df.shape[0]} rows and {df.shape[1]} columns.")
    print("data types of the columns:")
    print(df.dtypes)

def check_missing_values(df):
    """
    Checks for missing values in the data.

    Args:
        df (pd.dfFrame): The dfFrame to inspect.
    """

    print("Number of missing values in each column:")
    print(df.isnull().sum())

def print_summary_stats(df):
    """
    Prints summary statistics for numerical columns in the df.

    Args:
        df (pd.dfFrame): The dfFrame to inspect.
    """

    numerical_cols = df.select_dtypes(include=np.number).columns
    print("Summary statistics for numerical columns:")
    print(df[numerical_cols].describe())

def check_duplicate_rows(df):
    """
    Checks for duplicate rows in the df.

    Args:
        df (pd.dfFrame): The dfFrame to inspect.
    """

    print("Number of duplicate rows:")
    print(df.duplicated().sum())

def check_outliers(df):
    """
    Checks for outliers in the df using z-scores and box plots.

    Args:
        df (pd.dfFrame): The dfFrame to inspect.
    """

    print("Potential outliers based on z-scores:")
    z_scores = df.apply(stats.zscore)
    outliers = df[(z_scores.abs() > 3).any(axis=1)]
    print(outliers)

    print("Box plots for numerical columns:")
    numerical_cols = df.select_dtypes(include=np.number).columns
    if numerical_cols:
        df[numerical_cols].plot(kind="box", subplots=True, layout=(1, -1), figsize=(10, 5))
    else:
        print("No numerical columns found.")
    plt.show()

def check_df_ranges(df):
    """
    Checks for df ranges or patterns in the df (customizable).

    Args:
        df (pd.dfFrame): The dfFrame to inspect.
    """

    print("df ranges or patterns:")
    # Replace placeholders with your checks based on domain knowledge
    # Example: check if heights are within reasonable ranges
    if "Height" in df.columns:
        print(df[(df["Height"] < 100) | (df["Height"] > 250)])
    # Example: check if weights are between 20 and 200 kg
    if "Weight" in df.columns:
        print(df[(df["Weight"] < 20) | (df["Weight"] > 200)])

def perform_corr_analysis(df, columns_for_corr):
    """
    Performs correlation analysis for specified columns in the df.

    Args:
        df (pd.dfFrame): The dfFrame to inspect.
        columns_for_corr (list): List of column names for correlation analysis.
    """

    print("Correlation matrix for specified columns:")
    corr_matrix = df[columns_for_corr].corr()
    print(corr_matrix)

def check_skewness(df, columns_for_skewness):
    """
    Checks for skewness in specified columns in the df.

    Args:
        df (pd.dfFrame): The dfFrame to inspect.
        columns_for_skewness (list): List of column names for skewness checks.
    """

    print("Columns with skewed distributions:")
    skewed_columns = df[columns_for_skewness].select_dtypes(include=np.number).apply(lambda x: x.skew())
    skewed_columns = skewed_columns[abs(skewed_columns) > 0.5]
    print(skewed_columns)

def perform_visual_exploration(df):
    """
    Performs visual exploration of the df.

    Args:
        df (pd.dfFrame): The dfFrame to inspect.
    """

    print("Visual exploration of the df:")
    # Example: histograms for numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns
    if numerical_cols:
        df[numerical_cols].hist(bins=10, figsize=(10, 10))
        plt.show()
    # Add more visualizations as needed (scatter plots, line plots, etc.)

def perform_time_series_analysis(df):
    """
    Performs time series analysis of the df (placeholder).

    Args:
        df (pd.dfFrame): The dfFrame to inspect.
    """

    print("Visualizing time-related patterns:")