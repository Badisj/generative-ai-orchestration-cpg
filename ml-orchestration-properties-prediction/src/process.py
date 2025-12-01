import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

def parse_arguments():
    """
    Parse command-line arguments for input and output CSV file paths.
    Returns:
        argparse.Namespace: Parsed arguments containing input and output file paths.
    """
    parser = argparse.ArgumentParser(description="Process yogurt data and compute correlations.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the processed CSV file.')
    return parser.parse_args()

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def process_data(df):
    """
    Process the DataFrame to compute a correlation matrix and filter columns.
    
    Steps:
      1. Compute correlation matrix for all numeric columns.
      2. Drop rows/columns with all NaN values.
      3. Save and display correlation matrix as a heatmap.
      4. Remove columns not included in the correlation matrix from the original DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        tuple: (Processed DataFrame, List of retained columns)
    """
    
    # Identify columns with all NaN correlations (if any)
    cols_nan = ['Recipe Name', 'RO_Water', 'Past_Milk_2_4_Fat', 'Solid_Milk_Conc_100', 'Heat_Time_s', 'concentrate_ratio']
    if cols_nan:
        print(f"Dropping {len(cols_nan)} columns with NaN correlations: {cols_nan}")

    # Retain only columns present in the correlation matrix
    df = df.drop(columns=cols_nan)

    # Print columns retained for correlation analysis
    print(f"Retained {len(df.columns)} columns for correlation analysis: {list(df.columns)}")

    # Compute correlation matrix for numeric columns only
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()

    # Plot and save the correlation heatmap
    print("Generating correlation matrix heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='Blues', cbar=True)
    plt.title("Correlation Matrix of Estimated Yogurt Targets")
    plt.savefig('./data/processed/yogurt_targets_correlation_matrix.png', dpi=1200, bbox_inches='tight')
    plt.close()
    print("Correlation matrix heatmap saved to ./data/processed/yogurt_targets_correlation_matrix.png")

    return df

def main():
    """
    Main function to:
      1. Parse input arguments
      2. Load data
      3. Process data (compute correlations and filter)
      4. Save processed data to output CSV
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Load input dataset
    print(f"Loading data from {args.input_csv}...")
    df = load_data(args.input_csv)

    # Copy data for processing
    df_with_targets = df.copy()

    # Process data to compute correlations and filter valid columns
    df_with_targets = process_data(df_with_targets)

    # Save processed DataFrame to output CSV
    df_with_targets.to_csv(args.output_csv, index=False)
    print(f"Processed data saved to {args.output_csv}")

if __name__ == "__main__":
    # Entry point for script execution
    main()