import pandas as pd
import os
import shutil
import glob
from datetime import datetime

# File paths
plant_data_file = "/media/bigdata/plant_station/plant_data.csv"
all_plant_data_file = "/media/bigdata/plant_station/all_plant_data.csv"
temp_file = "/media/bigdata/plant_station/all_plant_data.temp.csv"
images_dir = "images"
latest_image_path = os.path.join(images_dir, "latest.jpg")

def save_latest_image():
    """
    Finds the most recent image in the images directory and
    copies it to images/latest.jpg.
    """
    # Get all jpg files in the images directory
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    
    # Filter out 'latest.jpg' itself to avoid copying it to itself
    image_files = [f for f in image_files if os.path.basename(f) != "latest.jpg"]
    
    if not image_files:
        print("No image files found in the images directory.")
        return False
    
    # Find the most recent file based on modification time
    latest_file = max(image_files, key=os.path.getmtime)
    
    # Copy the file
    shutil.copy2(latest_file, latest_image_path)
    print(f"Copied {latest_file} to {latest_image_path}")
    return True

def append_new_data():
    # Ensure the plant data file exists
    if not os.path.exists(plant_data_file):
        print(f"Error: Source file {plant_data_file} not found.")
        return
    
    # Load the new plant data
    df_new = pd.read_csv(plant_data_file)

    # Convert timestamps to datetime format
    df_new["Timestamp"] = pd.to_datetime(df_new["Timestamp"])

    # Check if all_plant_data.csv exists
    if os.path.exists(all_plant_data_file):
        # Load existing all-time data
        df_all = pd.read_csv(all_plant_data_file)
        df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"])

        # Find new entries by comparing timestamps
        df_merged = pd.concat([df_all, df_new]).drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp")
    else:
        # If no all-time data exists, create it from the new data
        df_merged = df_new

    # Write to temporary file first
    df_merged.to_csv(temp_file, index=False)
    
    # Atomically replace the original file
    shutil.move(temp_file, all_plant_data_file)

    print(f"Updated {all_plant_data_file} with new data.")

# Run the functions
if __name__ == "__main__":
    append_new_data()
    save_latest_image()
