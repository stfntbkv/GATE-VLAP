import json


def create_subgroup_of_capabilities(source_file, target_file, mappings):

    with open(source_file, "r") as f:
        json_data = json.load(f)

    output = []

    # Convert JSON data back to list of tuples
    for capability, skill in mappings.items():
        for item in json_data:
            if capability == item["name"]:
                item["skill"] = skill if len(skill) != 0 else item["skill"]
                output.append(item)

    with open(target_file, "w") as f:
        json.dump(output, f, indent=4)
