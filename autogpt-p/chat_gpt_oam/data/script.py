import csv

# Step 1: Read affordances from the first file
affordances_set = set()

with open('affordances/basic_affordances.csv', 'r') as file1:
    reader = csv.reader(file1)
    for row in reader:
        affordances_set.add(row[1])  # assuming affordance is in the second column

# Step 2 and 3: Read second file and keep only the lines with affordances present in the set
kept_lines = []

with open('../ground_truth/simulation_classes_ground_truth_basic.csv', 'r') as file2:
    reader = csv.reader(file2)
    header = next(reader)  # save the header for later
    for row in reader:
        if row[1] in affordances_set:  # assuming affordance is in the second column
            kept_lines.append(row)

# Step 4: Write kept lines back to the second file
with open('../ground_truth/simulation_classes_ground_truth_basic.csv', 'w', newline='') as file2:
    writer = csv.writer(file2)
    writer.writerow(header)  # write the header
    writer.writerows(kept_lines)