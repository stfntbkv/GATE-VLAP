# Import necessary libraries
import csv

# Define the source and target file paths
source_path = '../data/classes/classes.csv'
target_path = '../data/classes/random_classes.csv'

# Define the line indices to extract
line_indices = [11, 33, 22, 70, 59, 48, 64, 40, 19, 75, 53, 21, 20, 51, 7, 49, 68, 44, 56, 28, 36, 1, 65, 17, 27, 72,
                35, 61, 16, 71]

# Open the source CSV file and create a new target CSV file
with open(source_path, 'r') as csv_file, open(target_path, 'w', newline='') as target_file:
    # Create a CSV reader and writer object
    csv_reader = csv.reader(csv_file)
    csv_writer = csv.writer(target_file)

    # Loop through each line in the source CSV file
    for i, row in enumerate(csv_reader):
        # Check if the current line index is in the list of indices to extract
        if i in line_indices:
            # Write the current row to the target CSV file
            csv_writer.writerow(row)
