import os
import shutil
from src.utils.paths import *


# Pull the results for the different complexity datasets to a specified directory (or current working directory) for
# a given timestamp (if no timestamp specified, use latest)
def pull_results(source_dir=OUTPUT_DIR, target_dir=PAPERDATA_DIR, timestamp=None, file_prefix="noisy_simdata_compl"):
    # Browse source_dir for folder names starting with "file_name"
    relevant_folders = sorted([item for item in os.listdir(source_dir) if
                               os.path.isdir(os.path.join(source_dir, item)) and item.startswith(file_prefix)])

    # Iterate through each relevant folder
    for folder in relevant_folders:
        folder_full_path = os.path.join(source_dir, folder)

        # If no timestamp is given, take the latest
        if timestamp is None:
            timestamps = os.listdir(folder_full_path)
            # Remove underscore and take latest value
            numerical_timestamps = [int(timestamp.replace("_", "")) for timestamp in timestamps]
            numerical_timestamp = max(numerical_timestamps)
            timestamp = str(numerical_timestamp)[0:-4] + "_" + str(numerical_timestamp)[-4:]
            print(timestamp)

        # Select relevant sub-folder given timestamp
        current_folder = os.path.join(folder_full_path, timestamp)

        # Loop over pipe result files: rename and move every file to target directory
        for old_filename in os.listdir(current_folder):
            # Construct the new filename
            new_filename = f"{folder}_{old_filename}"

            # Move the file to the target directory with the modified name
            source_file_path = os.path.join(current_folder, old_filename)
            target_file_path = os.path.join(target_dir, new_filename)
            shutil.copy(source_file_path, target_file_path)

    print("Finished pulling files!")

