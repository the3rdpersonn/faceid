"""Image preprocessing functions, starting with eye-based rotation."""

from math import atan2, degrees

import cv2


def eye_line_angle(left_eye, right_eye):
    # Left/right refer to the picture. Image y coordinates increase downwards.
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return degrees(atan2(dy, dx))


def rotate_face(image, left_eye, right_eye):
    angle = eye_line_angle(left_eye, right_eye)
    eye_midpoint = ((left_eye[0] + right_eye[0]) / 2,
                    (left_eye[1] + right_eye[1]) / 2)
    rotation = cv2.getRotationMatrix2D(eye_midpoint, angle, 1.0)
    height, width = image.shape[:2]
    rotated_image = cv2.warpAffine(image, rotation, (width, height))
    return rotated_image, angle
