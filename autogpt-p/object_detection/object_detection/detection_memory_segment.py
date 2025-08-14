from typing import List


class BoundingBox:
    def __init__(self, xMin: float, yMin: float, xMax: float, yMax: float):
        self.xmin = xMin
        self.ymin = yMin
        self.xmax = xMax
        self.ymax = yMax

    @classmethod
    def default(cls):
        return BoundingBox(0.0, 0.0, 0.0, 0.0)


class BoundingBox3D:
    def __init__(self, xMin: float, yMin: float, zMin: float, xMax: float, yMax: float, zMax: float):
        self.xmin = xMin
        self.ymin = yMin
        self.zmin = zMin
        self.xmax = xMax
        self.ymax = yMax
        self.zmax = zMax

    @classmethod
    def default(cls):
        return BoundingBox3D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class DetectedObject:
    def __init__(self, bbx, bbx3d, class_name, id, confidence):
        self.class_name = class_name
        self.id = id
        self.bbx = bbx
        self.bbx3d = bbx3d
        self.confidence = confidence

    @classmethod
    def default_bbx(cls, class_name, id, confidence):
        return DetectedObject(BoundingBox.default(), BoundingBox3D.default(), class_name, id, confidence)


    def __hash__(self):
        return self.class_name.__hash__() + self.id

    def __eq__(self, other):
        if isinstance(other, DetectedObject):
            return self.class_name.__eq__(other.class_name) and self.id == other.id
        return False

    def __str__(self):
        return self.class_name + (str(self.id) if self.id >= 0 else "")


class ObjectDetectionState:
    def __init__(self, frame: str, detectedObjects: List[DetectedObject]):
        self.frame = frame
        self.detected_objects = detectedObjects


class ObjectRelation:
    def __init__(self, relation_name: str, related_objects: List[DetectedObject]):
        for o in related_objects:
            assert o
        self.related_objects = related_objects
        self.relation_name = relation_name

    def __eq__(self, other):
        if isinstance(other, ObjectRelation):
            return self.relation_name.__eq__(other.relation_name) and self.related_objects == other.related_objects
        return False

    def __str__(self):
        return self.relation_name + " " + " ".join(str(o) for o in self.related_objects)

    def __hash__(self):
        return hash(self.relation_name) + int(sum(hash(o) for o in self.related_objects))


class ObjectRelationState:
    def __init__(self, frame: str, object_relations: List[ObjectRelation]):
        self.frame = frame
        self.object_relations = object_relations
