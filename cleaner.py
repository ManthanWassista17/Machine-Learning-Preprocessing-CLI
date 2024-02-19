from data_loader import load_data  # Assuming load_data is defined in data_loader.py
import pandas as pd
import numpy as np


def clean_data(df, options={"method": "dropna", "threshold": 3.0}):
    """
    Cleans a DataFrame by handling missing values, duplicates, outliers, and other errors.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        options (dict, optional): A dictionary specifying parameters for cleaning.
            - method (str, optional): The method to handle missing values.
                - "dropna": Drop rows with missing values.
                - "fillna": Fill missing values with a strategy (e.g., mean, median).
            - threshold (float, optional): The threshold for identifying outliers using z-scores.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    print("Initial DataFrame:")
    print(df)

    # Check for missing values
    missing_counts = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_counts)

    # Handle missing values based on options
    if missing_counts.any():
        if options["method"] == "dropna":
            cleaned_df = df.dropna()
            print("\nDropped rows with missing values:", df.shape[0] - cleaned_df.shape[0])
        elif options["method"] == "fillna":
            # Implement appropriate filling strategy based on column types and distributions
            cleaned_df = df.fillna(...)  # Replace `...` with specific imputation techniques
            print("\nFilled missing values using specified strategy")
        else:
            raise ValueError("Invalid method for handling missing values.")
    else:
        cleaned_df = df

    # Check for duplicate rows
    duplicate_count = cleaned_df.duplicated().sum()
    print("\nNumber of duplicate rows:", duplicate_count)

    # Handle duplicate rows (optional)
    if duplicate_count > 0:
        # Decide on a strategy based on criteria (e.g., keep first/last occurrence)
        # `cleaned_df = cleaned_df.drop_duplicates(...)`  # Example using drop_duplicates
        print("\nDuplicate rows not handled in this example.")
    else:
        print("\nNo duplicate rows found.")

    # Identify potential outliers using z-scores
    z_scores = cleaned_df.apply(pd.Series.zscore)  # Calculate z-scores for each column
    abs_z_scores = z_scores.abs()
    potential_outliers = abs_z_scores[abs_z_scores > options["threshold"]]
    outlier_indices = potential_outliers.any(axis=1)
    outlier_count = outlier_indices.sum()
    print("\nPotential outliers based on z-scores (absolute value > {}):".format(options["threshold"]))
    print(outlier_count)

    # Handle outliers (optional)
    if outlier_count > 0:
        # Decide on a strategy based on criteria (e.g., remove, winsorize, robust methods)
        # `cleaned_df = cleaned_df[~outlier_indices]`  # Example: remove outliers
        print("\nOutliers not handled in this example.")
    else:
        print("\nNo outliers identified based on specified threshold.")

    # (Optional) Perform additional cleaning steps such as transformations or visualizations

    print("\nCleaned DataFrame:")
    print(cleaned_df)

    return cleaned_df

# # Example usage
# data = ...  # Load your data here
# options = {"method": "fillna", "threshold": 2.5}  # Customize cleaning options
# cleaned_df = clean_data(data, options)
