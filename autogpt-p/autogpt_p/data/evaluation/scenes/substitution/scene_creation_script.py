import random
from typing import List

from object_affordance_mapping.object_affordance_mapping import ObjectAffordanceDatabase, \
    ObjectAffordanceMappingDatabase
from object_detection.detection_memory_segment import DetectedObject

from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.helpers.scene_read_write import write_scene


def _load_oa_db_objects() -> List[str]:
    oam_db = ObjectAffordanceMappingDatabase.load_from_data("simulation_classes_.json",
                                                            "proposed_affordances_alternative.json",
                                                            "gpt-4_.json")
    return [o.name for o in oam_db.oa_database.objects]


def create_scene(name: str, objects: List[str]):
    detected_objects = [DetectedObject.default_bbx(o, 0, 1.0) for o in objects]
    write_scene(name + ".txt", detected_objects, [], {})


def create_scene_without(name: str, objects: List[str]):
    all_objects = _load_oa_db_objects()
    [all_objects.remove(o) for o in objects]
    detected_objects = [DetectedObject.default_bbx(o, 0, 1.0) for o in all_objects]
    write_scene(name + ".txt", detected_objects, [], {})


def create_scene_random(name: str, objects: List[str], target_number: int):
    all_objects = _load_oa_db_objects()
    [all_objects.remove(o) for o in objects]
    random_objects = random.sample(all_objects, target_number - len(objects))
    detected_objects = [DetectedObject.default_bbx(o, 0, 1.0) for o in objects + random_objects]
    write_scene(name + ".txt", detected_objects, [], {})
