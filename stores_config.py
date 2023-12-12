import cv2
import numpy as np

stores = {
    "subway-subcentro": {
        "type": "box-restricted",
        "coords": {
            "x1": 0,
            "y1": 150,
            "x2": 650,
            "y2": 415
        },
        "polygon": np.array([
            [0, 150], [650, 150], [650, 415], [0, 415]]).astype(int), # topleft, top-right, bottom-right, bottom-left
        "rotate": cv2.ROTATE_90_CLOCKWISE,
        "crop": None
    },
    "subway-los-militares": {
        "type": "box-restricted",
        "coords":{
            "x1": 0,
            "y1": 260,
            "x2": 480,
            "y2": 490
        },
        "polygon": np.array([
            [0, 260], [480, 260], [480, 490], [0, 490]
        ]),
        "rotate": None,
        "crop": {"height": [0, 500], "width": [0, -1]}
    },
    "subway-mall-del-centro": {
        "type": "box-restricted",
        "coords":{
            "x1": 0,
            "y1": 420,
            "x2": 480,
            "y2": 680
        },
        "polygon": np.array([
            [0, 420], [480, 420], [480, 690], [0, 690]
        ]),
        "rotate": None,
        "crop": None
    },
    "subway-merced": {
        "type": "box-restricted",
        "coords":{
            "x1": 100,
            "y1": 60,
            "x2": 420,
            "y2": 700
        },
        "polygon": np.array([
            [100, 60], [420, 60], [420, 700], [100, 700]
        ]),
        "rotate": None,
        "crop": None
    },
    "subway-tenderini": {
        "type": "box-restricted",
        "coords":{
            "x1": 0,
            "y1": 230,
            "x2": 480,
            "y2": 530
        },
        "polygon": np.array([
            [0, 230], [480, 230], [480, 430], [0, 530]
        ]),
        "rotate": None,
        "crop": {"height": [0, 540], "width": [0, -1]}
    }
}