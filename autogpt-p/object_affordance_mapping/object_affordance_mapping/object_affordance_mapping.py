from __future__ import annotations
import os
from typing import List
import json

DATA_DIR = "data"
AFFORDANCES_DIR = "affordances"
CLASSES_DIR = "classes"
OAM_DIR = "oam"
PYTHON_DIR = "python"
PACKAGE_DIR = "object_affordance_mapping"

OAM_VAR_NAME = "OAM_ROOT"
PACKAGE_ROOT_DIR = os.environ[OAM_VAR_NAME]


class ObjectAffordanceDatabase:

    def __init__(self, object_file_path, affordances_file_path):
        self.objects = read_objects(object_file_path)
        self.affordances = read_affordances(affordances_file_path)

    def get_affordance_by_name(self, name):
        result = [a for a in self.affordances if a.name == name]
        return result[0] if len(result) != 0 else None

    def get_object_by_name(self, name):
        result = [o for o in self.objects if o.name == name]
        return result[0] if len(result) != 0 else None

    def __str__(self):
        return "Objects: " + str([str(o) for o in self.objects]) + "\n" \
            + "Affordances: " + str([str(a) for a in self.affordances])


class ObjectClass:

    def __init__(self, name: str, object_id: str):
        self.name = name
        self.id = object_id

    def __eq__(self, other):
        if isinstance(other, ObjectClass):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


class AffordanceType:

    def __init__(self, name: str, affordance_id: str, prompt: str, description: str):
        self.name = name
        self.id = affordance_id
        self.chat_gpt_prompt = prompt
        self.description = description

    def __eq__(self, other):
        if isinstance(other, AffordanceType):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


class ObjectAffordanceMapping:

    def __init__(self, object_class: ObjectClass, affordances: List[AffordanceType] = None):
        if affordances is None:
            affordances = []
        self.object_class = object_class
        self.affordances = affordances

    def add_affordance(self, affordance: AffordanceType):
        self.affordances.append(affordance)

    def __str__(self):
        return str(self.object_class) + str([str(a) for a in self.affordances])


class ObjectAffordanceMappingDatabase:
    def __init__(self, oa_database: ObjectAffordanceDatabase):
        self.oams = []
        self.oa_database = oa_database

    def __str__(self):
        return "\n".join([str(oam) for oam in self.oams])

    def add_oam(self, oam: ObjectAffordanceMapping):
        self.oams.append(oam)

    def add_oams(self, oams: List[ObjectAffordanceMapping]):
        self.oams.append(oams)

    def read_from_file(self, filepath: str, clear=False):
        """
        These methods assume the file ist formatted as Object,Affordance and that all affordances of objects are
        listed consecutive
        :param clear: whether to delete all previously added oams
        :param filepath: the path were the previously generated object affordance mapping is from
        :return:
        """
        if clear:
            self.oams.clear()

        f = open(filepath)
        data = json.load(f)
        db = self.oa_database
        for o in data["oams"]:
            oam = ObjectAffordanceMapping(db.get_object_by_name(o["object"]))
            for a in o["affordances"]:
                oam.add_affordance(db.get_affordance_by_name(a["name"]))
            self.oams.append(oam)

    def get_affordances_by_object_class(self, object_class: ObjectClass) -> List[AffordanceType]:
        oam = [oam for oam in self.oams if oam.object_class == object_class]
        return oam[0].affordances if len(oam) > 0 else []

    def get_affordance_by_object_name(self, name: str) -> List[AffordanceType]:
        oam = [oam for oam in self.oams if oam.object_class.name == name]
        return oam[0].affordances if len(oam) > 0 else []

    def get_objects_with_affordance(self, affordance: AffordanceType) -> List[ObjectClass]:
        return [oam.object_class for oam in self.oams if affordance in oam.affordances]

    @classmethod
    def load_from_data(cls, classes_file: str, affordances_file: str, oam_file: str) -> ObjectAffordanceMappingDatabase:
        """
        Loads OAMDatabase relative to the data dir of object_affordance_mapping so only the file names have to be given
        :param classes_file:
        :param affordances_file:
        :param oam_file:
        :return: OAMDatabase loaded from the given files
        """
        data_path = os.path.join(PACKAGE_ROOT_DIR, DATA_DIR)
        return cls.load(os.path.join(data_path, CLASSES_DIR, classes_file),
                        os.path.join(data_path, AFFORDANCES_DIR, affordances_file),
                        os.path.join(data_path, OAM_DIR, oam_file))

    @classmethod
    def load(cls,  classes_path: str, affordances_path: str, oam_path: str) -> ObjectAffordanceMappingDatabase:
        """
        Loads OAMDatabase relative to the current working directory
        :param classes_path:
        :param affordances_path:
        :param oam_path:
        :return: OAMDatabase loaded from the given files
        """
        oa_db = ObjectAffordanceDatabase(classes_path, affordances_path)
        oam_db = ObjectAffordanceMappingDatabase(oa_db)
        oam_db.read_from_file(oam_path)
        return oam_db


def read_objects(objects_file_path: str):
    with open(objects_file_path, 'r') as f:
        # load the JSON data into a list
        data = json.load(f)
    # create a list of ObjectClass objects
    objects = []
    for obj in data["object_classes"]:
        # create a new ObjectClass object for each item in the list
        object_class_obj = ObjectClass(obj['name'], obj['id'])
        objects.append(object_class_obj)
    return objects


def read_affordances(affordances_file_path: str):
    print(affordances_file_path)
    with open(affordances_file_path, 'r') as f:
        # load the JSON data into a list
        data = json.load(f)
    # create a list of ObjectClass objects
    affordances = []
    for obj in data["affordance_types"]:
        # create a new ObjectClass object for each item in the list
        print(obj)
        affordance = AffordanceType(obj['name'], obj['id'], obj['prompt'], obj['description'])
        affordances.append(affordance)
    return affordances
