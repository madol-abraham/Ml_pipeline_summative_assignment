import pandas as pd
import numpy as np

def load_and_align_datasets(primary_csv_path, secondary_csv_path):
    """Load and align two datasets by repeating the smaller dataset."""
    df1 = pd.read_csv(primary_csv_path)
    df2 = pd.read_csv(secondary_csv_path)
    
    repeats = len(df1) // len(df2) + 1
    df2_repeated = pd.concat([df2] * repeats, ignore_index=True).iloc[:len(df1)]
    
    df1.reset_index(drop=True, inplace=True)
    df2_repeated.reset_index(drop=True, inplace=True)
    
    return df1, df2_repeated

def concatenate_selected_columns(df1, df2_repeated, columns_to_select):
    """Concatenate selected columns from the second dataset to the first dataset."""
    selected_columns = df2_repeated[columns_to_select]
    df_combined = pd.concat([df1, selected_columns], axis=1)
    return df_combined

def remove_redundant_columns(df, columns_to_remove):
    """Remove unnecessary columns from the dataset."""
    df.drop(columns=columns_to_remove, inplace=True)
    return df

def fill_missing_values(df, columns):
    """Fill missing values in specified columns with their median."""
    for col in columns:
        df[col].fillna(df[col].median(), inplace=True)
    return df

def encode_categorical_columns(df, categorical_columns):
    """Convert categorical variables to numerical format."""
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes
    return df

def remove_duplicates(df):
    """Remove duplicate rows from the dataset."""
    return df.drop_duplicates()

def determine_irrigation(row):
    """Custom function to determine if irrigation is needed based on conditions."""
    if row['Soil Moisture'] < 12 and row['Temperature'] > 35:
        return 1  # High temp and low soil moisture
    elif row['Soil Moisture'] < 15 and row['Temperature'] > 25 and row['Crop Type'] in [1, 2]:
        return 1  # Water-intensive crops (encoded categories)
    elif row['Soil Moisture'] < 40 and row['Temperature'] > 20:
        return 1  # Moderate conditions
    else:
        return 0  # No irrigation needed

def add_irrigation_column(df):
    """Apply irrigation logic to create a new target column."""
    df['Irrigation_Needed'] = df.apply(determine_irrigation, axis=1)
    return df

def preprocess_data(primary_csv, secondary_csv, columns_to_select, columns_to_remove, fillna_columns, categorical_columns):
    """Complete preprocessing pipeline."""
    df1, df2_repeated = load_and_align_datasets(primary_csv, secondary_csv)
    df = concatenate_selected_columns(df1, df2_repeated, columns_to_select)
    df = remove_redundant_columns(df, columns_to_remove)
    df = fill_missing_values(df, fillna_columns)
    df = encode_categorical_columns(df, categorical_columns)
    df = remove_duplicates(df)
    df = add_irrigation_column(df)
    return df

if __name__ == "__main__":
    processed_df = preprocess_data(
        'TARP.csv',
        'smart_irrigation_dataset.csv',
        columns_to_select=['Soil Type', 'Crop Type'],
        columns_to_remove=[' Soil Humidity', 'Air temperature (C)', 'N', 'P', 'K', 'Wind gust (Km/h)', 'Pressure (KPa)', 'ph', 'Status'],
        fillna_columns=['Wind speed (Km/h)', 'Air humidity (%)', 'rainfall'],
        categorical_columns=['Soil Type', 'Crop Type']
    )
    processed_df.to_csv('processed_data.csv', index=False)
