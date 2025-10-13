import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import shutil

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Delete folder and its contents
        print(f"Folder '{folder_path}' has been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

def create_folder_if_not_exists(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")

folder_path = '../data/intermediary_steps/temp'
create_folder_if_not_exists(folder_path)

def remove_columns_from_csv(input_csv, output_csv, columns_to_remove):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Print available columns
    print("Available columns in the CSV:")
    print(df.columns.tolist())
    
    # Find and handle missing columns
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    if missing_columns:
        print(f"The following columns are missing and will be ignored: {missing_columns}")

    # Remove specified columns if they exist
    df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

    print(f"Specified columns have been removed and saved to {output_csv}.")

columns_to_remove = [
    'LAYER', 'SONEASTS_','SONEASTS_I', 'SCID', 'CLAF', 'PRID', 
    'BOTDEP', 'AREA', 'PERIMETER', 'ISO', 'SOVEUR_ID', 'DEGRAD_ID', 
    'SOVID_NEW', 'ISOC', 'SUID', 'NEWSUID', 'TCID', 'PROP', 'PRID', 
    'TOPDEP', 'BOTDEP', 'MISCUNITS', 'SOILMAPUNI', 'PRID1', 'PRID2',
    'PRID3', 'PRID4', 'PRID5', 'PRID6', 'PRID7', 'PRID8', 'PRID9', 
    'PRID10', 'SONWESTS_', 'SONWESTS_I', 'ISO_', 'DEGRAD_ID_', 
    'FNODE_', 'TNODE_', 'LPOLY_', 'RPOLY_', 'LENGTH', 'SONEAST_', 
    'SONEAST_ID', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'IDTIC', 'XTIC', 
    'YTIC', 'MISC', 'CLIP', 'SONWEST_', 'SONWEST_ID', 'SOIL1',
    'PROP1', 'SOIL2', 'PROP2', 'SOIL3', 'PROP3', 'SOIL4', 'PROP4',
    'SOIL5', 'PROP5', 'SOIL6', 'PROP6', 'SOIL7', 'PROP7', 'SOIL8', 
    'PROP8', 'SOIL9', 'PROP9'
    ]

input_csv = "../data/intermediary_steps/complete_sotwis_data.csv"
output_csv = "../data/intermediary_steps/temp/complete_sotwis_data_deleted_irrelevant_cols.csv"

remove_columns_from_csv(input_csv, output_csv, columns_to_remove)

def remove_exclusive_entries(input_csv, output_csv, exclusive_fields):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Identify rows where all fields except the exclusive_fields are empty
    other_fields = [col for col in df.columns if col not in exclusive_fields]
    rows_to_remove = df[other_fields].isnull().all(axis=1)

    # Remove the identified rows
    df_cleaned = df[~rows_to_remove]

    # Save the modified DataFrame to a new CSV file
    df_cleaned.to_csv(output_csv, index=False)

    print(f"Rows with information exclusively in {exclusive_fields} have been removed and saved to {output_csv}.")

exclusive_fields = ['BOTTOM_LEFT_LAT', 'BOTTOM_LEFT_LON', 'UPPER_RIGHT_LAT', 'UPPER_RIGHT_LON']

input_csv = "../data/intermediary_steps/temp/complete_sotwis_data_deleted_irrelevant_cols.csv"
output_csv = "../data/intermediary_steps/temp/complete_sotwis_data_no_empty_entries.csv"

remove_exclusive_entries(input_csv, output_csv, exclusive_fields)

def remove_empty_columns(input_csv, output_csv):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Drop columns where all values are NaN
    df_cleaned = df.dropna(axis=1, how='all')

    # Save the updated DataFrame to a new CSV file
    df_cleaned.to_csv(output_csv, index=False)

    print(f"Empty columns have been removed and saved to {output_csv}.")

input_csv = '../data/intermediary_steps/temp/complete_sotwis_data_no_empty_entries.csv' 
output_csv = '../data/intermediary_steps/temp/complete_sotwis_data_cleaned.csv' 

remove_empty_columns(input_csv, output_csv)

def categorical_to_numerical(input_csv, output_csv):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Define the mapping of textural classes to normalized values
    drain_mapping = {
        'C': 1.0,  # Coarse
        'M': 0.75,  # Medium
        'Z': 0.5,  # Medium fine
        'F': 0.25,  # Fine
        'V': 0.0   # Very fine
    }

    # Replace the DRAIN field values using the mapping
    if 'DRAIN' in df.columns:
        df['DRAIN'] = df['DRAIN'].map(drain_mapping)

    # Normalize texture class (PSCL)
    if 'PSCL' in df.columns:
        df['PSCL'] = df['PSCL'].map(drain_mapping)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

    print(f"Dataset has been normalized and saved to {output_csv}.")

input_csv = '../data/intermediary_steps/temp/complete_sotwis_data_cleaned.csv'  # Input CSV file
output_csv = '../data/intermediary_steps/complete_sotwis_preprocessed.csv'  # Output CSV file after normalization

categorical_to_numerical(input_csv, output_csv)

delete_folder(folder_path=folder_path)