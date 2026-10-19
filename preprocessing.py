"""Image preprocessing functions, starting with eye-based rotation."""

from math import atan2, ceil, degrees, floor

import cv2


def Get_angle(left_eye, right_eye):
    #input: eye coordinate pairs
    # Left/right refer to the picture. Image y coordinates increase downwards.
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return degrees(atan2(dy, dx))


def rotate_face(image, left_eye, right_eye):
    angle = Get_angle(left_eye, right_eye)
    eye_midpoint = ((left_eye[0] + right_eye[0]) / 2,
                    (left_eye[1] + right_eye[1]) / 2)
    rotation = cv2.getRotationMatrix2D(eye_midpoint, angle, 1.0)
    height, width = image.shape[:2]
    rotated_image = cv2.warpAffine(image, rotation, (width, height))
    return rotated_image, angle


def crop_face(image, box_coordinates, margin=0.10):
    # Input: rotated image, (x1, y1, x2, y2), extra margin on each side.
    x1, y1, x2, y2 = box_coordinates
    box_width = x2 - x1
    box_height = y2 - y1
    height, width = image.shape[:2]
    left = max(0, min(width, floor(x1 - box_width * margin)))
    top = max(0, min(height, floor(y1 - box_height * margin)))
    right = max(0, min(width, ceil(x2 + box_width * margin)))
    bottom = max(0, min(height, ceil(y2 + box_height * margin)))
    if right <= left or bottom <= top:
        raise ValueError("Face box does not produce a non-empty crop.")
    cropped_image = image[top:bottom, left:right].copy()
    return cropped_image, (left, top, right, bottom)
