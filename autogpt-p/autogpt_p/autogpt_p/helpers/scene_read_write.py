"""
This file contains methods to read and write scenes into simple files that are optimized for a fast creation.
So the syntax is very limited and does not include things like confidence and the bounding box.
I want to be able to quickly define new scenes using the file format.
It does not check for consistency between relations and objects
The format is:

SCENE-DESCRIPTION
OBJECTS
class:id
class:id
...
END-OBJECTS
RELATIONS
relation_name class:id class:id ...
relation_name class:id class:id ...
...
END-RELATIONS

"""
from typing import List, Tuple, Dict

from object_detection.detection_memory_segment import DetectedObject, BoundingBox3D
from object_detection.detection_memory_segment import ObjectRelation

HEADER = "SCENE-DESCRIPTION\n"
START_OBJECTS = "OBJECTS\n"
END_OBJECTS = "END-OBJECTS\n"
START_RELATIONS = "RELATIONS\n"
END_RELATIONS = "END-RELATIONS\n"
START_LOCATIONS = "LOCATIONS\n"
END_LOCATIONS = "END-LOCATIONS"

ID_SEP = ":"
REL_SEP = " "

DEFAULT_BBX = [0, 0, 1, 1]
DEFAULT_CONFIDENCE = 1.0


def read_scene(filename) -> Tuple[List[DetectedObject], List[ObjectRelation],
Dict[str, List[DetectedObject]]]:
    """
    Reads in a file with the format specified here
    :param filename: the path to the file to read the scene from
    :return: a list of the detected objects and a list of the objects relations
    """
    with open(filename, 'r') as file:
        string = file.read()
        objects_string = _extract_substring(string, START_OBJECTS, END_OBJECTS).split("\n")
        detected_objects = [_read_object(s) for s in objects_string]
        relations_string = _extract_substring(string, START_RELATIONS, END_RELATIONS).split("\n")
        detected_relations = [_read_relation(s) for s in relations_string]
        locations_string = _extract_substring(string, START_LOCATIONS, END_LOCATIONS).split("\n")
        locations = {_read_location(s)[0]: _read_location(s)[1] for s in locations_string}
    return detected_objects, detected_relations, locations


def write_scene(filename, objects: List[DetectedObject], relations: List[ObjectRelation],
                locations: Dict[str, List[DetectedObject]]):
    """
    Writes the scene in the format specified here
    :param filename: the path to the file where to write the scene to
    :param objects: the detected objects in the scene
    :param relations: the object relations
    :param locations:
    """
    with open(filename, 'w') as file:
        file.write(HEADER)
        file.write(START_OBJECTS)
        for o in objects:
            file.write(_write_object(o) + "\n")
        file.write(END_OBJECTS)
        file.write(START_RELATIONS)
        for r in relations:
            file.write(_write_relation(r) + "\n")
        file.write(END_RELATIONS)
        file.write(START_LOCATIONS)
        for l, objects in locations.items():
            file.write(_write_location(l, objects) + "\n")
        file.write(END_LOCATIONS)

def print_scene(objects: List[DetectedObject], relations: List[ObjectRelation],
                locations: Dict[str, List[DetectedObject]]):
    """
    Writes the scene in the format specified here
    :param filename: the path to the file where to write the scene to
    :param objects: the detected objects in the scene
    :param relations: the object relations
    :param locations:
    """
    file = ""
    file += HEADER + "\n"
    file += START_OBJECTS + "\n"
    for o in objects:
        file += (_write_object(o) + "\n")
    file += END_OBJECTS + "\n"
    file += START_RELATIONS + "\n"
    for r in relations:
        file += (_write_relation(r) + "\n")
    file += END_RELATIONS + "\n"
    file += START_LOCATIONS + "\n"
    for l, objects in locations.items():
        file += (_write_location(l, objects) + "\n")
    file += END_LOCATIONS + "\n"
    print(file)


def _write_object(object: DetectedObject) -> str:
    return object.class_name + ID_SEP + str(object.id)


def _write_relation(relation: ObjectRelation) -> str:
    return relation.relation_name + REL_SEP + REL_SEP.join([_write_object(o) for o in relation.related_objects])


def _write_location(location: str, objects: List[DetectedObject]) -> str:
    return location + REL_SEP + REL_SEP.join([_write_object(o) for o in objects])


def _read_object(string: str) -> DetectedObject:
    split = string.split(ID_SEP)
    detected_object = DetectedObject.default_bbx(split[0], int(split[1]), DEFAULT_CONFIDENCE)
    if len(split) < 3:
        return detected_object
    else:
        bbx_string = split[2].split(",")
        assert len(bbx_string) == 6
        bbx = [float(s) for s in bbx_string]
        detected_object.bbx3d = BoundingBox3D(bbx[0], bbx[1], bbx[2], bbx[3], bbx[4], bbx[5])
        return detected_object

def _read_relation(string: str) -> ObjectRelation:
    split = string.split(REL_SEP)
    return ObjectRelation(split[0], [_read_object(s) for s in split[1:]])


def _read_location(string: str) -> Tuple[str, List[DetectedObject]]:
    split = string.split(REL_SEP)
    return split[0], [_read_object(s) for s in split[1:]]


def _extract_substring(input_string, start, end):
    start_index = input_string.find(start) + len(start)
    end_index = input_string.find(end)
    if 0 <= start_index < end_index and end_index >= 0:
        substring = input_string[start_index:end_index].strip()
        return substring
    else:
        return ""


def main():
    # simple test code whether object relations work
    table0 = DetectedObject.default_bbx("table", 0, DEFAULT_CONFIDENCE)
    table1 = DetectedObject.default_bbx("table", 1, DEFAULT_CONFIDENCE)
    counter0 = DetectedObject.default_bbx("counter", 0, DEFAULT_CONFIDENCE)
    fork0 = DetectedObject.default_bbx("fork", 0, DEFAULT_CONFIDENCE)
    fork1 = DetectedObject.default_bbx("fork", 1, DEFAULT_CONFIDENCE)
    dinner_knife0 = DetectedObject.default_bbx("dinner_knife", 0, DEFAULT_CONFIDENCE)
    dinner_knife1 = DetectedObject.default_bbx("dinner_knife", 1, DEFAULT_CONFIDENCE)
    bowl0 = DetectedObject.default_bbx("bowl", 1, DEFAULT_CONFIDENCE)
    human0 = DetectedObject.default_bbx("human", 0, DEFAULT_CONFIDENCE)
    on0 = ObjectRelation("on", [bowl0, table0])
    on1 = ObjectRelation("on", [fork0, counter0])
    on2 = ObjectRelation("on", [fork1, counter0])
    on3 = ObjectRelation("on", [dinner_knife1, counter0])
    on4 = ObjectRelation("on", [dinner_knife0, counter0])
    at0 = ObjectRelation("at", [human0, table1])
    detected_objects = [table0, table1, bowl0, counter0, fork1, fork0, dinner_knife0, dinner_knife1]
    detected_relations = [on0, on1, on2, on3, on4, at0]
    detected_locations = {"table0": [table0, bowl0], "counter0": [fork1, fork0, dinner_knife0, dinner_knife1]}
    write_scene("evaluation_memory_new.txt", detected_objects, detected_relations, detected_locations)
    o, r, l = read_scene("evaluation_memory_new.txt")


if __name__ == "__main__":
    main()
