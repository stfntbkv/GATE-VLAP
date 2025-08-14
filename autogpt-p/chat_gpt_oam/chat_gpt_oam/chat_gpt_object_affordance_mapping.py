import argparse
import concurrent
import json
import logging
import os
import time

import openai
from openai import OpenAI

from chat_gpt_oam.tools.rate_manager import RateManager
from chat_gpt_oam.logical_expressions.expression_parser import parse_logic_expression
from object_affordance_mapping.object_affordance_mapping import ObjectAffordanceDatabase, ObjectAffordanceMapping, \
    AffordanceType

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

specific_mode = "specific"
combination_mode = "combination"
list_affordances_mode = "list_affordances"
list_objects_mode = "list_objects"
preparation_specific = {"role": "system", "content": "You answer the following questions only with either yes or no"}
preparation_list_affordances = {"role": "system", "content": "You answer the following questions only "
                                                             "with a list of terms separated by commas. Only list "
                                                             "affordances "
                                                             "within the following list:\n"}
preparation_list_objects = {"role": "system", "content": "You answer the following questions only "
                                                         "with a list of objects separated by commas. Only list objects "
                                                         "within the following list:\n "}
prompt_list_affordances = "List the affordances of a typical {}"
prompt_list_objects = "List all objects that have the affordance {} which means: {}"


def invert_map(d):
    inverted = {}
    for key, values in d.items():
        for value in values:
            inverted.setdefault(value, []).append(key)
    return inverted

def print_affordance_with_description(affordance: AffordanceType):
    return "{}: {}".format(affordance.name, affordance.description)


class ChatGPTOAM:

    def __init__(self, oa_database: ObjectAffordanceDatabase, model: str, logging_file_path: str = ""):
        self.oa_database = oa_database
        self.oams = []
        self.query_counter = 0
        self.start_time = time.time()
        self.model = model
        self.logging = len(logging_file_path) > 0
        if self.logging:
            logging.basicConfig(filename=logging_file_path,
                                level=logging.INFO,
                                format='%(message)s')
        if model == "gpt-4":
            RateManager(max_rate=500)
        else:
            RateManager(max_rate=500)

    def generate_oams(self, start_index=0, mode="combination", two_stage_mode=False):
        """
        Generates the object affordance mappings using chat gpt api.
        Saves the result in self.oams
        :param two_stage_mode: whether to first ask the question without specifying to answer with yes or no and then
        summarize
        :param mode:
        :param start_index: in case a prior run went wrong pick the index of the first class with no output to start
        generating oams from there
        """
        self.start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=70) as executor:
            # submit 10 tasks to the pool
            print("running parallel")
            if mode != list_objects_mode:
                results = [executor.submit(self.generate_oam_single, obj, mode) for obj in
                           self.oa_database.objects[start_index:]]
                # print the results as they become available
                concurrent.futures.wait(results)
                print("result ready")
                for future in results:
                    self.oams.append(future.result())
            else:
                self.generate_oams_from_affordances()

    def generate_oams_from_affordances(self):
        affordance_object_mapping = {}
        obj_list = [o.name for o in self.oa_database.objects]

        for affordance in self.oa_database.affordances:
            prompt = prompt_list_objects.format(affordance.name, affordance.description)
            aff_message = {"role": "user", "content": prompt}
            rate_manager = RateManager(0)
            preparation = preparation_list_objects.copy()
            preparation["content"] = preparation["content"] + str(obj_list)

            # ChatGPT Interaction
            history = []
            r = ""
            while len(r) < 1:
                rate_manager.wait_until_rate()
                try:
                    response = client.chat.completions.create(model=self.model,
                                                              messages=[preparation, aff_message],
                                                              temperature=0,
                                                              timeout=50)
                    r = response.choices[0].message.content
                    # when a network error occurs write the preliminary results in json file
                except Exception as e:
                    print(e)
            if self.logging:
                logging.info("Q:{}\nA: {}".format(prompt,r))
            objects = [o.strip().replace(".", "") for o in r.split(",")]
            # the filtered list should only contain objects that actually exists
            filtered = list(filter(lambda obj: obj in obj_list, objects))
            affordance_object_mapping[affordance] = [self.oa_database.get_object_by_name(name) for name in filtered]

        object_affordance_mapping = invert_map(affordance_object_mapping)
        self.oams = [ObjectAffordanceMapping(object, affordances) for object, affordances
                     in object_affordance_mapping.items()]

    def generate_oam_single(self, obj, mode):
        oam = ObjectAffordanceMapping(obj)
        history = []
        if mode == specific_mode:
            self.generate_oam_specific(oam, history)
        elif mode == list_affordances_mode:
            self.generate_oam_list(oam, history)
        elif mode == combination_mode:
            self.generate_oam_combination(oam, history)
        self._log_history(oam.object_class.name, history)
        return oam

    def generate_oam_list(self, oam, history=None):
        # prompt design
        prompt = prompt_list_affordances.format(oam.object_class.name)
        aff_message = {"role": "user", "content": prompt}
        aff_list = [i.name for i in self.oa_database.affordances]
        rate_manager = RateManager(0)
        preparation = preparation_list_affordances.copy()
        preparation["content"] = preparation["content"] \
                                                  + "\n".join([print_affordance_with_description(a)
                                                               for a in self.oa_database.affordances])
        r = ""
        while len(r) < 1:
            rate_manager.wait_until_rate()
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[preparation, aff_message],
                    temperature=0
                )
                r = response["choices"][0]["message"]["content"]
                # when a network error occurs write the preliminary results in json file
            except Exception as e:
                print(e)

        history.append((prompt, r))
        affordances = [aff.strip().replace(".", "") for aff in r.split(",")]
        filtered = list(filter(lambda aff: aff in aff_list, affordances))
        oam.affordances = [self.oa_database.get_affordance_by_name(f) for f in filtered]

    def generate_oam_specific(self, oam, history=None):
        rate_manager = RateManager(50)
        for affordance in self.oa_database.affordances:
            prompt = affordance.chat_gpt_prompt.format(oam.object_class.name)
            aff_message = {"role": "user", "content": prompt}

            rate_manager.wait_until_rate()
            # ChatGPT Interaction
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[preparation_specific, aff_message],
                    temperature=0,
                    timeout=5
                )
                # when a network error occurs write the preliminary results in json file
            except:
                self.write_json("results/test.json")
                return
            self.query_counter = self.query_counter + 1
            r = response["choices"][0]["message"]["content"]
            history.append((prompt, r))
            print(oam.object_class.name + "-" + affordance.name + "-" + r)
            if "yes" in r.lower():
                oam.add_affordance(affordance)

    def generate_oam_combination(self, oam, history):
        for affordance in self.oa_database.affordances:
            expression = parse_logic_expression(affordance.chat_gpt_prompt.replace("{}", oam.object_class.name),
                                                self.model,
                                                history)
            if expression.evaluate():
                print(affordance.name + "-" + oam.object_class.name)
                oam.add_affordance(affordance)
            else:
                print("not({})".format(affordance.name + "-" + oam.object_class.name))

    def read_json(self, file):
        """
        Reads from json file and stores data in self.oams.
        Can be used to easily extend runs
        :param file: the file to read from
        :return:
        """
        f = open(file)
        data = json.load(f)
        db = self.oa_database
        for o in data["oams"]:
            oam = ObjectAffordanceMapping(db.get_object_by_name(o["object"]))
            for a in o["affordances"]:
                oam.add_affordance(db.get_affordance_by_name(a["name"]))
            self.oams.append(oam)

    def write_json(self, file):
        """
        Writes object affordance mappings saved in self.oams to a json file
        :param file: path to the json file
        """

        data = {"oams": [{"object": oam.object_class.name,
                          "affordances": [{"name": a.name} for a in oam.affordances]} for oam in self.oams]}
        with open(file, 'w') as f:
            # Write the data to the file
            json.dump(data, f, indent=4)

    def _log_history(self, object, history):
        if self.logging:
            logging.info("Object {} -------------------".format(object))
            for pair in history:
                logging.info("Q: {}\nA: {}".format(pair[0], pair[1]))

    def write_csv(self):
        pass

    def read_scv(self):
        pass  # TODO


def main():
    parser = argparse.ArgumentParser(description='Parse command line arguments for object and affordance data')

    # Required arguments
    parser.add_argument('--objects', type=str, required=False, help='Path to objects.json',
                        default="../data/classes/new.json")
    parser.add_argument('--affordances', type=str, required=False, help='Path to affordances.json',
                        default="../data/affordances/proposed_affordances_alternative.json")
    parser.add_argument('--result', type=str, required=False, help='Path where result should be saved',
                        default="results/new.json")
    parser.add_argument('--model', type=str, required=False, help='Used model', default='gpt-4')

    # Optional arguments
    parser.add_argument('--extend', type=str, help='If set this preloads a json file')
    parser.add_argument('--start', type=int, help='start index of the object list from where to start asking chatgpt')

    args = parser.parse_args()

    if args.start:
        start_index = args.start
    else:
        start_index = 0

    args = parser.parse_args()
    db = ObjectAffordanceDatabase(args.objects, args.affordances)
    cgpt_oam = ChatGPTOAM(db, args.model)
    if args.extend:
        cgpt_oam.read_json(args.extend)
    cgpt_oam.generate_oams(start_index)
    print("Finished")
    cgpt_oam.write_json(args.result)


if __name__ == "__main__":
    main()
