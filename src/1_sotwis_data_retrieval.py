import geopandas as gpd
import pandas as pd
import os
from pathlib import Path
import dbfread
import warnings
warnings.filterwarnings('ignore')

def get_project_root():
    """
    Get the root directory of the project (parent of src folder)
    """
    current_file = Path(__file__)
    return current_file.parent.parent

def process_shapefile(shp_file):
    """
    Process a single shapefile and return geometry bounds and attribute data
    """
    bounds_data = []
    try:
        print(f"Reading shapefile: {shp_file.name}")
        gdf = gpd.read_file(str(shp_file))
        
        # Get the bounds for each geometry
        for idx, row in gdf.iterrows():
            bounds = row.geometry.bounds
            bound_dict = {
                'Bottom_Left_Lat': bounds[1],
                'Bottom_Left_Lon': bounds[0],
                'Upper_Right_Lat': bounds[3],
                'Upper_Right_Lon': bounds[2]
            }
            
            # Add all non-geometry columns to the bounds dictionary
            for col in gdf.columns:
                if col != 'geometry':
                    bound_dict[col] = row[col]
            
            bounds_data.append(bound_dict)
        
        return bounds_data
    except Exception as e:
        print(f"Error processing {shp_file}: {str(e)}")
        return []

def process_dbf(dbf_file):
    """
    Process a single DBF file and return its data
    """
    try:
        print(f"Reading DBF file: {dbf_file.name}")
        table = dbfread.DBF(str(dbf_file))
        return [{k: v for k, v in record.items()} for record in table]
    except Exception as e:
        print(f"Error processing {dbf_file}: {str(e)}")
        return []

def extract_sotwis_data():
    """
    Extract data from SOTWIS SOTER files and combine into a single CSV.
    """
    # Setup paths
    project_root = get_project_root()
    data_dir = project_root / "data" / "SOTWIS_SOVEUR_ver1.0"
    shapefiles_dir = data_dir / "ShapeFiles"
    
    all_data = []
    
    # Process shapefiles
    print("\nProcessing shapefiles...")
    if shapefiles_dir.exists():
        shp_files = list(shapefiles_dir.rglob("*.shp"))
        for shp_file in shp_files:
            bounds_data = process_shapefile(shp_file)
            if bounds_data:
                all_data.extend(bounds_data)
    else:
        print(f"Warning: ShapeFiles directory not found at {shapefiles_dir}")
    
    # Process standalone DBF files
    print("\nProcessing DBF files...")
    dbf_files = []
    for dir_name in ['ShapeFiles', 'Layers', 'LegendFiles']:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            dbf_files.extend([
                f for f in dir_path.rglob("*.dbf") 
                if not os.path.exists(str(f)[:-4] + '.shp')
            ])
    
    for dbf_file in dbf_files:
        dbf_data = process_dbf(dbf_file)
        if dbf_data:
            all_data.extend(dbf_data)
    
    # Check if we have any data
    if not all_data:
        raise ValueError("No data was successfully extracted from the files")
    
    # Convert to DataFrame
    print("\nCombining data...")
    df = pd.DataFrame(all_data)
    
    # Clean up column names and remove duplicates
    df.columns = [str(col).strip().upper() for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Fill NaN values with appropriate placeholders
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna('')
    df = df.fillna('')
    
    return df

def get_region_bounds(df):
    """
    Calculate the overall region bounds from the dataset
    """
    try:
        bounds_columns = ['BOTTOM_LEFT_LAT', 'BOTTOM_LEFT_LON', 'UPPER_RIGHT_LAT', 'UPPER_RIGHT_LON']
        if all(col in df.columns for col in bounds_columns):
            # Convert to numeric and handle invalid entries
            df_numeric = df[bounds_columns].apply(pd.to_numeric, errors='coerce')
            
            # Calculate region bounds using the cleaned numeric data
            region_bounds = {
                'Region_Bottom_Left_Lat': df_numeric['BOTTOM_LEFT_LAT'].min(skipna=True),
                'Region_Bottom_Left_Lon': df_numeric['BOTTOM_LEFT_LON'].min(skipna=True),
                'Region_Upper_Right_Lat': df_numeric['UPPER_RIGHT_LAT'].max(skipna=True),
                'Region_Upper_Right_Lon': df_numeric['UPPER_RIGHT_LON'].max(skipna=True)
            }
            return region_bounds
    except Exception as e:
        print(f"Error calculating region bounds: {str(e)}")
    
    return None

def main():
    try:
        # Create output directory if it doesn't exist
        output_dir = get_project_root() / "data" / "intermediary_steps"
        output_dir.mkdir(exist_ok=True)
        
        # Extract all data
        print("Starting data extraction...")
        combined_data = extract_sotwis_data()
        
        # Save to CSV
        output_file = output_dir / "complete_sotwis_data.csv"
        combined_data.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        
        # Calculate and save region bounds if possible
        region_bounds = get_region_bounds(combined_data)
        if region_bounds:
            print("\nOverall region boundaries:")
            for key, value in region_bounds.items():
                print(f"{key}: {value}")
            
            bounds_file = output_dir / "region_bounds.csv"
            pd.DataFrame([region_bounds]).to_csv(bounds_file, index=False)
            print(f"Region bounds saved to {bounds_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()