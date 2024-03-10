import os
import shutil
from src.utils.paths import *


# Pull the results for the different complexity datasets to a specified directory (or current working directory) for
# a given timestamp (if no timestamp specified, use latest)
def pull_results(file_prefix, source_dir=PAPER_PIPE_OUTPUT_DIR, target_dir=PAPERDATA_DIR, timestamp="latest"):

    # If only one item provided convert to list since function loops over list
    if isinstance(file_prefix, str):
        file_prefix = [file_prefix]

    # Get latest timestamp if none is provided:
    if timestamp == "latest":
        timestamps = os.listdir(PAPER_PIPE_OUTPUT_DIR)
        # Remove underscore and take latest value
        numerical_timestamps = [int(timestamp.replace("_", "")) for timestamp in timestamps]
        numerical_timestamp = max(numerical_timestamps)
        timestamp = str(numerical_timestamp)[0:-4] + "_" + str(numerical_timestamp)[-4:]


    # # Browse source_dir for folder names starting with "file_name"
    # relevant_folders = sorted([item for item in os.listdir(source_dir) if
    #                            os.path.isdir(os.path.join(source_dir, item)) and item.startswith(file_prefix)])

    relevant_folder = os.path.join(source_dir, timestamp)

    # Iterate through each relevant folder
    for folder in os.listdir(relevant_folder):
        current_folder = os.path.join(source_dir, timestamp, folder)

        if folder in file_prefix:
            # Loop over pipe result files: rename and move every file to target directory
            for old_filename in os.listdir(current_folder):
                # Construct the new filename
                new_filename = f"{folder}_{old_filename}"

                # Move the file to the target directory with the modified name
                source_file_path = os.path.join(current_folder, old_filename)

                # Move logs to log folder
                if new_filename.endswith(".log"):
                    target_file_path = os.path.join(target_dir, "logs", new_filename)
                else:
                    target_file_path = os.path.join(target_dir, new_filename)

                # Copy files
                shutil.copy(source_file_path, target_file_path)

    print("Finished pulling files!")

