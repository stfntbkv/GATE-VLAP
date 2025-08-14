from object_affordance_mapping.object_affordance_mapping import *
from os.path import join as opj
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the root directory of the package
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

data_dir = "data"
affordance_dir = "affordances"
classes_dir = "classes"
oam_dir = "oam"

affordance_file = opj(opj(opj(root_dir, data_dir), affordance_dir), "proposed_affordances_alternative.json")
classes_file = opj(opj(opj(root_dir, data_dir), classes_dir), "classes.json")
oam_file = opj(opj(opj(root_dir, data_dir), oam_dir), "preliminary_results.json")


def basic_usage_example():

    # load database for objects and affordances
    oa_database = ObjectAffordanceDatabase(classes_file, affordance_file)

    # initialize oam_database and load oams from json file
    oam_database = ObjectAffordanceMappingDatabase(oa_database)
    oam_database.read_from_file(oam_file)

    # get affordances of different objects

    # alternative loading
    oam_database = ObjectAffordanceMappingDatabase.load_from_data("simulation_classes_.json",
                                                                  "proposed_affordances_alternative.json",
                                                                  "gpt-4_.json")

    knife_affordances = oam_database.get_affordance_by_object_name("coffee_cup")

    for aff in knife_affordances:
        print(aff.name)


if __name__ == "__main__":
    print(OAM_PACKAGE_DIR)
    basic_usage_example()
