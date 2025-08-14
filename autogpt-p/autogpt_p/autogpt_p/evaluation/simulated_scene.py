from typing import List, Dict

from object_detection.detection_memory_segment import DetectedObject, ObjectRelation


class SimulatedScene:

    def __init__(self, objects: List[DetectedObject], relations: List[ObjectRelation],
                 object_locations: Dict[str, List[DetectedObject]]):
        self.objects = objects
        self.relations = [ObjectRelation(r.relation_name, [objects[objects.index(o)] if o in objects else o
                                                           for o in r.related_objects]) for r in relations]
        self.object_locations = {key: [objects[objects.index(o)] for o in os] for key, os in object_locations.items()}

    def get_locations(self) -> List[str]:
        return list(self.object_locations.keys())

    def get_objects_at(self, location: str) -> List[DetectedObject]:
        if location in self.object_locations.keys():
            return self.object_locations[location]
        else:
            return []

    def get_all_relations_of(self, detected_object: DetectedObject):
        return list(set([r for r in self.relations if detected_object in r.related_objects]))
