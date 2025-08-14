import argparse
import csv
import json
from typing import List

HELPER_SUFFIXES = ["_can", "_bottle"]


def convert_class_name(name: str):
    return name.lower().replace(" ", "_").replace("-", "_")


def remove_helper_suffix(name: str, helper_suffix: List[str]):
    ignore_list = ["trash_can"]
    if name in ignore_list:
        return name
    class_name = name
    for suffix in helper_suffix:
        class_name = class_name.replace(suffix, "")
    return class_name


def remove_helper_suffixes_class(source_file_path: str, helper_suffix: List[str]):
    file = open(source_file_path)
    data = json.load(file)
    for o in data["object_classes"]:
        o["name"] = remove_helper_suffix(o["name"], helper_suffix)
    with open(source_file_path, 'w') as file:
        # Write the data to the file
        json.dump(data, file)

    source_file_path = source_file_path.replace(".json", ".csv")

    with open(source_file_path) as f:
        # Read the CSV file using the csv.reader() method
        csv_reader = csv.reader(f)
        result = []
        for row in csv_reader:
            result.append([row[0], remove_helper_suffix(row[1], helper_suffix)])
    with open(source_file_path, 'w') as f:
        csv_writer = csv.writer(f)
        for row in result:
            csv_writer.writerow(row)


def remove_helper_suffixes_oam(source_file_path: str, helper_suffix: List[str]):
    file = open(source_file_path)
    data = json.load(file)
    for oam in data["oams"]:
        oam["object"] = remove_helper_suffix(oam["object"], helper_suffix)
    with open(source_file_path, 'w') as file:
        # Write the data to the file
        json.dump(data, file)

    source_file_path = source_file_path.replace(".json", ".csv")
    with open(source_file_path) as f:
        # Read the CSV file using the csv.reader() method
        csv_reader = csv.reader(f)
        result = []
        for row in csv_reader:
            result.append([remove_helper_suffix(row[0], helper_suffix), row[1]])
    with open(source_file_path, 'w') as f:
        csv_writer = csv.writer(f)
        for row in result:
            csv_writer.writerow(row)


def convert_classes(source_file_path: str, target_file_path: str):
    file = open(source_file_path)
    data = json.load(file)
    for o in data["object_classes"]:
        o["name"] = convert_class_name(o["name"])
    with open(target_file_path, 'w') as file:
        # Write the data to the file
        json.dump(data, file)

    source_file_path = source_file_path.replace(".json", ".csv")
    target_file_path = target_file_path.replace(".json", ".csv")
    with open(source_file_path) as f:
        # Read the CSV file using the csv.reader() method
        csv_reader = csv.reader(f)
        result = []
        for row in csv_reader:
            result.append([row[0], convert_class_name(row[1])])
    with open(target_file_path, 'w') as f:
        csv_writer = csv.writer(f)
        for row in result:
            csv_writer.writerow(row)


def convert_oam(source_file_path: str, target_file_path: str):
    file = open(source_file_path)
    data = json.load(file)
    for oam in data["oams"]:
        oam["object"] = convert_class_name(oam["object"])
    with open(target_file_path, 'w') as file:
        # Write the data to the file
        json.dump(data, file,indent=4)

    source_file_path = source_file_path.replace(".json", ".csv")
    target_file_path = target_file_path.replace(".json", ".csv")
    with open(source_file_path) as f:
        # Read the CSV file using the csv.reader() method
        csv_reader = csv.reader(f)
        result = []
        for row in csv_reader:
            result.append([convert_class_name(row[0]), row[1]])
    with open(target_file_path, 'w') as f:
        csv_writer = csv.writer(f)
        for row in result:
            csv_writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--method', type=str, default='suffix', help='What to convert')
    parser.add_argument('--function', type=str, default='classes', help='Object to convert')
    parser.add_argument('--source', type=str, default='data/affordances/proposed_affordances_alternative',
                        help='Path to source file')
    parser.add_argument('--target', type=str, default='results/combination7', help='Path to target file')
    args = parser.parse_args()
    if args.method == 'convert':
        if args.function == "classes":
            convert_classes(args.source, args.target)
        else:
            convert_oam(args.source, args.target)
    else:
        if args.function == "classes":
            remove_helper_suffixes_class(args.source, HELPER_SUFFIXES)
        else:
            remove_helper_suffixes_oam(args.source, HELPER_SUFFIXES)
