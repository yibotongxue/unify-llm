from base64 import b64encode

import cv2
import numpy as np


def encode_image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", image)
    return b64encode(buffer).decode("utf-8")


def encode_image_to_base64_with_path(image_path: str) -> str:
    image = cv2.imread(image_path)
    return encode_image_to_base64(image)
