from object_detection.detection_memory_segment import DetectedObject, ObjectRelation

from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.helpers.scene_read_write import read_scene, write_scene, print_scene

O = "on"

table = "table"
coffee_table = "coffee_table"
counter = "counter"
fork = "fork"
dinner_knife = "dinner_knife"
spoon = "spoon"
spatula = "spatula"
kitchen_knife = "kitchen_knife"
scissors = "scissors"
bottle_opener = "bottle_opener"
bowl = "bowl"
plate = "plate"
cutting_board = "cutting_board"
coffee_cup = "coffee_cup"
glass = "glass"
wine_glass = "wine_glass"
plastic_cup = "plastic_cup"
can = "can"
bottle = "bottle"
bucket = "bucket"
basket = "basket"
sink = "sink"
dishwasherc = "dishwasherc"
microwave = "microwave"
oven = "oven"
stove = "stove"
plant = "plant"
watering_can = "watering_can"
apple = "apple"
banana = "banana"
salad = "salad"
orange = "orange"
tomato = "tomato"
cucumber = "cucumber"
milk_cartoon = "milk_cartoon"
cornflakes = "cornflakes"
tape = "tape"
hammer = "hammer"
screw = "screw"
nail = "nail"
stapler = "stapler"
book = "book"
remote = "remote"
chair = "chair"
desk = "desk"
monitor = "monitor"
pc = "pc"
keyboard = "keyboard"
screwdriver = "screwdriver"
sponge = "sponge"
cloth = "cloth"
broom = "broom"
hand_brush = "hand_brush"
towel = "towel"
paper_towel = "paper_towel"
pan = "pan"
pot = "pot"
box = "box"
door = "door"
window = "window"
bench = "bench"
couch = "couch"
stool = "stool"
refrigerator = "refrigerator"
tablet = "tablet"
mop = "mop"
tissue = "tissue"
tea_pot = "tea_pot"
vase = "vase"
milk = "milk"
water = "water"

table0 = "table:0"
coffee_table0 = "coffee_table:0"
counter0 = "counter:0"
fork0 = "fork:0"
dinner_knife0 = "*dinner_knife:0"
spoon0 = "spoon:0"
spatula0 = "spatula:0"
kitchen_knife0 = "kitchen_knife:0"
scissors0 = "scissors:0"
bottle_opener0 = "bottle_opener:0"
bowl0 = "bowl:0"
plate0 = "plate:0"
cutting_board0 = "cutting_board:0"
coffee_cup0 = "coffee_cup:0"
glass0 = "glass:0"
wine_glass0 = "wine_glass:0"
plastic_cup0 = "plastic_cup:0"
can0 = "can:0"
bottle0 = "bottle:0"
bucket0 = "bucket:0"
basket0 = "basket:0"
sink0 = "sink:0"
dishwasherc0 = "dishwasherc:0"
microwave0 = "microwave:0"
oven0 = "oven:0"
stove0 = "stove:0"
plant0 = "plant:0"
watering_can0 = "watering_can:0"
apple0 = "apple:0"
banana0 = "banana:0"
salad0 = "salad:0"
orange0 = "orange:0"
tomato0 = "tomato:0"
cucumber0 = "cucumber:0"
milk_cartoon0 = "milk_cartoon:0"
cornflakes0 = "cornflakes:0"
tape0 = "tape:0"
hammer0 = "hammer:0"
screw0 = "screw:0"
nail0 = "nail:0"
stapler0 = "stapler:0"
book0 = "book:0"
remote0 = "remote:0"
chair0 = "chair:0"
desk0 = "desk:0"
monitor0 = "monitor:0"
pc0 = "pc:0"
keyboard0 = "keyboard:0"
screwdriver0 = "screwdriver:0"
sponge0 = "sponge:0"
cloth0 = "cloth:0"
broom0 = "broom:0"
hand_brush0 = "hand_brush:0"
towel0 = "towel:0"
paper_towel0 = "paper_towel:0"
pan0 = "pan:0"
pot0 = "pot:0"
box0 = "box:0"
door0 = "door:0"
window0 = "window:0"
bench0 = "bench:0"
couch0 = "couch:0"
stool0 = "stool:0"
refrigerator0 = "refrigerator:0"
tablet0 = "tablet:0"
mop0 = "mop:0"
tissue0 = "tissue:0"
tea_pot0 = "tea_pot:0"
vase0 = "vase:0"
milk0 = "milk:0"
water0 = "water:0"

class Creation:

    def __init__(self):
        self.objects = []
        self.relations = []
        self.locations = {}

    def reset(self):
        self.objects = []
        self.relations = []
        self.locations = {}

    def add_relation(self, name, object_list):
        objects = [self._create_and_return(_name(o), _id(o)) for o in object_list]
        self.relations.append(ObjectRelation(name, objects))
        if name == "on":
            self.add_to_location(_obj(objects[1]), _obj(objects[0]))

        self.objects = list(set(self.objects))

    def add_to_location(self, loc, obj):
        print(loc)
        loc_name = _to_loc(loc)
        object = self._create_and_return(_name(obj), _id(obj))
        if loc_name in self.locations.keys():
            self.locations[loc_name].append(object)
        else:
            loc = self._create_and_return(_name(loc), _id(loc))
            self.locations[loc_name] = [loc, object]

    def add_object(self, obj):
        self._create_and_return(_name(obj), _id(obj))

    def write(self, filename):
        write_scene(filename + ".txt", self.objects, self.relations, self.locations)

    def print(self):
        print_scene(self.objects, self.relations, self.locations)

    def _create_and_return(self, obj, id):
        if id == -1:
            id = self._get__id(obj)
        object = DetectedObject.default_bbx(obj, id, 1.0)
        if object not in self.objects:
            self.objects.append(object)
        return object

    def _get__id(self, obj):
        id = -1
        for o in self.objects:
            if obj == o.class_name and o.id > id:
                id = o.id
        return id + 1


def _name(str):
    return str.split(":")[0]


def _id(str):
    if len(str.split(":")) == 2:
        return int(str.split(":")[1])
    else:
        return -1


def _to_loc(string):
    return _name(string) + str(_id(string))


def _loc(obj: DetectedObject):
    return obj.class_name + str(obj.id)


def _obj(obj: DetectedObject):
    return obj.class_name + ":" + str(obj.id)
