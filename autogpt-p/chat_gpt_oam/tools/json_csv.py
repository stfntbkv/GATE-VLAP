import argparse
import csv
import json


def convert_objects(objects_file: str):
    with open(objects_file + '.csv') as f:
        # Read the CSV file using the csv.reader() method
        csv_reader = csv.reader(f)

        # Create an empty list to hold the object classes
        object_classes = []

        # Loop through each row in the CSV file
        for row in csv_reader:
            # Get the values of the "id" and "name" fields
            object_id = row[0]
            object_name = row[1]

            # Create a dictionary for the object class
            object_class = {"name": object_name, "id": object_id}

            # Append the object class to the list
            object_classes.append(object_class)

    # Create a dictionary for the JSON data
    json_data = {"object_classes": object_classes}

    # Open a file for writing
    with open(objects_file + '.json', 'w') as f:
        # Write the JSON data to the file using json.dump()
        json.dump(json_data, f, indent=4)


def convert_affordances(affordances_file: str):
    with open(affordances_file + '.csv') as f:
        # Read the CSV file using the csv.reader() method
        csv_reader = csv.reader(f)

        # Create an empty list to hold the object classes
        affordances = []

        # Loop through each row in the CSV file
        for row in csv_reader:
            # skip header if it exists
            if row[0] == "id":
                continue
            # Get the values of the "id" and "name" fields
            affordance_id = row[0]
            affordance_name = row[1]
            affordance_prompt = row[2]
            affordance_description = row[3]

            # Create a dictionary for the object class
            affordance = {"name": affordance_name, "id": affordance_id, "prompt": affordance_prompt,
                          "description": affordance_description}

            # Append the object class to the list
            affordances.append(affordance)

    # Create a dictionary for the JSON data
    json_data = {"affordance_types": affordances}

    # Open a file for writing
    with open(affordances_file + '.json', 'w') as f:
        # Write the JSON data to the file using json.dump()
        json.dump(json_data, f, indent=4)


def convert_affordances_to_csv(affordances_file: str):
    with open(affordances_file + '.json') as f:
        # Read the CSV file using the csv.reader() method
        data = json.load(f)
        rows = []
        for affordance in data["affordance_types"]:
            rows.append((affordance["name"], affordance["id"], affordance["prompt"], affordance["description"]))

        with open(affordances_file + '.csv', "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            headers = ["name", "id", "prompt", "description"]
            csvwriter.writerow(headers)
            csvwriter.writerows(rows)


def convert_results(results_file: str):
    # Sample JSON data
    with open(results_file + '.json') as f:
        data_dict = json.load(f)

        # Load JSON data into a Python dictionary
        # data_dict = json.loads(json_data)

        # Extract "affordances" list from the dictionary
        rows = []
        for i in range(len(data_dict["oams"])):
            affordances_list = data_dict["oams"][i]["affordances"]

            # Extract "object" value from the dictionary
            object_value = data_dict["oams"][i]["object"]

            # Define the headers for the CSV file
            headers = ["object", "affordances"]

            # Loop through the affordances list and create a row for each affordance
            for affordance in affordances_list:
                rows.append([object_value, affordance["name"]])

        # Write the data to a CSV file
        with open(results_file + '.csv', "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerows(rows)

def convert_results_to_json(results_file: str):
    with open(results_file + '.csv') as f:
        # Read the CSV file using the csv.reader() method
        csv_reader = csv.reader(f)

        # Create an empty list to hold the object classes
        oams = {}

        # Loop through each row in the CSV file
        for row in csv_reader:
            # skip header if it exists
            if row[0] in oams.keys():
                oams[row[0]].append(row[1])
            else:
                oams[row[0]] = [row[1]]

            # Create a dictionary for the object class
                # Create a dictionary for the JSON data

    oams_json = [{"object": k, "affordances" : [{"name": n} for n in v]} for k,v in oams.items()]
    json_data = {"oams": oams_json}

    # Open a file for writing
    with open(results_file + '.json', 'w') as f:
        # Write the JSON data to the file using json.dump()
        json.dump(json_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--objects', type=str, default='data/classes/random_classes', help='Path to objects file')
    parser.add_argument('--affordances', type=str, default='data/affordances/proposed_affordances_alternative',
                        help='Path to affordances file')
    parser.add_argument('--result', type=str, default='results/combination7', help='Path to results file')
    args = parser.parse_args()
    convert_objects(args.objects)
    convert_affordances(args.affordances)
    convert_results(args.result)
