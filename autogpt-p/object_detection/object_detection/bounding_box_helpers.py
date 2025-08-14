from typing import List, Tuple

from object_detection.detection_memory_segment import BoundingBox


def x_within(x: float, bbx: BoundingBox) -> bool:
    return bbx.xmin <= x <= bbx.xmax


def y_within(y: float, bbx: BoundingBox) -> bool:
    return bbx.ymin <= y <= bbx.ymax


def on(bbx1: BoundingBox, bbx2: BoundingBox) -> bool:
    # as the top left of the image is 0,0 y_max is lower than y_min
    return y_within(bbx1.ymax, bbx2) and (x_within(bbx1.xmin, bbx2) or x_within(bbx1.xmax, bbx2))


def within(bbx1: BoundingBox, bbx2: BoundingBox) -> bool:
    return x_within(bbx1.xmin, bbx2) and x_within(bbx1.xmax, bbx2) \
        and y_within(bbx1.ymin, bbx2) and y_within(bbx1.ymax, bbx2)


def xywh2bbx(xywh: List[float]) -> BoundingBox:
    return BoundingBox(xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3])


def xyxy2bbx(xyxy: List[float]) -> BoundingBox:
    return BoundingBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3])


def cxcys2bbx(cxcys: List[float]) -> BoundingBox:
    hw = cxcys[2]
    hh = cxcys[3]
    return BoundingBox(cxcys[0] - hw, cxcys[1] - hh, cxcys[2] + hw, cxcys[3] + hh)


def bbx2xyxy(bbx: BoundingBox) -> List[float]:
    return [bbx.xmin, bbx.ymin, bbx.xmax, bbx.ymax]


def bbx2xywh(bbx: BoundingBox) -> List[float]:
    return [bbx.xmin, bbx.ymin, bbx.xmax - bbx.xmin, bbx.ymax - bbx.ymin]


def bbx2cxcys(bbx: BoundingBox) -> List[float]:
    w = bbx.xmax - bbx.xmin
    h = bbx.ymax - bbx.ymin
    return [bbx.xmin + 0.5 * w, bbx.ymin + 0.5 * h, w, h]


def bbx_to_corners(bbx) -> List[Tuple[float, float]]:
    return [(bbx.xmin, bbx.ymin), (bbx.xmax, bbx.ymin), (bbx.xmin, bbx.ymax), (bbx.xmax, bbx.ymax)]
