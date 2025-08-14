import csv
import json
import os
from typing import Dict

from pddl.core import LogicOp, Not

from autogpt_p.evaluation.autogpt_p_evaluation import correct_goal, collapse_goal
from autogpt_p.helpers.dnf import goal_to_dnf
from autogpt_p.helpers.scene_read_write import read_scene


def transform_environment(file: str):
    dict_relations = []
    objects, relations, locations = read_scene(file)
    for rel in relations:
        if rel.relation_name == "on" or rel.relation_name == "in":
            dict_relations.append([str(o) for o in rel.related_objects])
        elif rel.relation_name == "at" and rel.related_objects[0].class_name == "robot":
            dict_relations.append([str(o) for o in rel.related_objects])
        elif rel.relation_name == "at" and rel.related_objects[0].class_name == "human":
            dict_relations.append([None, "person"])
            dict_relations.append(["person", str(rel.related_objects[1])])
        elif rel.relation_name == "liquid_in":
            dict_relations.append(["liquid"] + [str(o) for o in rel.related_objects])
        elif rel.relation_name == "opened":
            dict_relations.append(["opened"] + [str(o) for o in rel.related_objects])
        elif rel.relation_name == "closed":
            dict_relations.append(["closed"] + [str(o) for o in rel.related_objects])
        elif rel.relation_name == "chopped":
            dict_relations.append(["chopped"] + [str(o) for o in rel.related_objects])
        else:
            print("Unsupported relation:" + rel.relation_name)
    for loc in locations:
        dict_relations.append([loc, loc])

    return dict_relations


def transform_scenario(file: str):
    result = {}
    result["lmp_cfg"] = "saycan/repl"
    result["environments"] = {}
    result["experiments"] = []
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        scenarios = []
        # skip the header line
        for row in reader:
            experiment = {}
            task = row['task']
            experiment["initial_event"] = task
            scene_file = row['scene_file']
            scene_name = scene_file.replace(".txt","")
            experiment["object_locations"] = "@" + scene_name
            if scene_name not in result["environments"].keys():
                relative_path_to_scene_dir = "../scenes"
                scenes_dir = os.path.join(".", relative_path_to_scene_dir)
                result["environments"][scene_name] = transform_environment(os.path.join(scenes_dir, scene_file))
            desired_goal = row['desired_goal']
            goal = LogicOp.from_string(desired_goal)
            goal = correct_goal(collapse_goal(goal))
            experiment["goal_state"] = transform_goal_state(goal)
            correction = row["correction_event"]
            experiment["goal_state"]["correction_event"] = correction
            result["experiments"].append(experiment)

    with open(file.replace("csv","json"), 'w') as json_file:
        # Step 4: Write the data to the file
        json.dump(result, json_file, indent=4)


def transform_goal_state(goal) -> Dict:
    goal_state = {"object_locations": [], "liquid_locations": {}, "open_closed": {}, "chopped":[], "forbidden_locations":[]}
    dnf = goal_to_dnf(goal)
    for subgoal in dnf:
        for pred in subgoal:
            if isinstance(pred, Not):
                goal_state["forbidden_locations"].append([str(v.name) for v in pred.logic_elements[0].variables])
                continue
            if pred.name == "on" or pred.name == "in":
                goal_state["object_locations"].append([str(v.name) for v in pred.variables])
            if pred.name == "at" and "robot" in pred.variables[0].name:
                goal_state["agent_location"] = pred.variables[1].name
            if pred.name == "liquid_in":
                goal_state["liquid_locations"][pred.variables[0].name] = pred.variables[1].name
            if pred.name == "opened":
                goal_state["open_closed"][pred.variables[0].name] = "opened"
            if pred.name == "closed":
                goal_state["open_closed"][pred.variables[0].name] = "closed"
            if pred.name == "chopped":
                goal_state["chopped"].append(pred.variables[0].name)
            if pred.name == "inhand":
                if "human" in pred.variables[1].name:
                    goal_state["object_locations"].append([pred.variables[0].name, "person"])
                else:
                    goal_state["object_locations"].append([pred.variables[0].name, "righthand"])
    return goal_state


if __name__ == "__main__":
    transform_scenario("evaluation_memory.csv")
