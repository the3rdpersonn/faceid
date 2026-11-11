"""Image geometry and buffalo l embedding functions."""

from math import atan2, ceil, degrees, floor

import cv2
import numpy as np
import onnxruntime


def Get_angle(left_eye, right_eye):
    # left and right refer to the picture. image y coordinates increase downwards.
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_in_radians = atan2(dy, dx)
    angle_in_degrees = degrees(angle_in_radians)
    return angle_in_degrees


def rotate_face(image, left_eye, right_eye):
    angle = Get_angle(left_eye, right_eye)
    midpoint_x = (left_eye[0] + right_eye[0]) / 2
    midpoint_y = (left_eye[1] + right_eye[1]) / 2
    eye_midpoint = (midpoint_x, midpoint_y)

    rotation = cv2.getRotationMatrix2D(eye_midpoint, angle, 1.0)
    height, width = image.shape[:2]
    rotated_image = cv2.warpAffine(image, rotation, (width, height))
    return rotated_image, angle, rotation


def transform_face_box(box, rotation_matrix):
    # move all four corners into the rotated image's coordinate system
    x1, y1, x2, y2 = box
    corners = np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]], dtype=np.float64)
    transformed_boxes = cv2.transform(corners, rotation_matrix)
    rotated_corners = transformed_boxes[0]

    x_coordinates = rotated_corners[:, 0]
    y_coordinates = rotated_corners[:, 1]
    left = float(x_coordinates.min())
    top = float(y_coordinates.min())
    right = float(x_coordinates.max())
    bottom = float(y_coordinates.max())
    return left, top, right, bottom


def crop_face(image, box_coordinates, margin=0.002):
    # expand the face box, then keep the crop inside the image
    x1, y1, x2, y2 = box_coordinates
    box_width = x2 - x1
    box_height = y2 - y1
    height, width = image.shape[:2]
    horizontal_margin = box_width * margin
    vertical_margin = box_height * margin
    left = floor(x1 - horizontal_margin)
    top = floor(y1 - vertical_margin)
    right = ceil(x2 + horizontal_margin)
    bottom = ceil(y2 + vertical_margin)

    left = min(left, width)
    top = min(top, height)
    right = min(right, width)
    bottom = min(bottom, height)
    left = max(0, left)
    top = max(0, top)
    right = max(0, right)
    bottom = max(0, bottom)

    if right <= left or bottom <= top:
        raise ValueError("Face box does not produce a non-empty crop.")
    crop = image[top:bottom, left:right]
    cropped_image = crop.copy()
    return cropped_image, (left, top, right, bottom)


def prepare_embedding_face(original_image, landmarks):
    # fit the five original-image points to insightface's 112-pixel template
    target_points = [(38.2946, 51.6963), (73.5318, 51.5014),
                     (56.0252, 71.7366), (41.5493, 92.3655),
                     (70.7299, 92.2041)]
    if len(landmarks) != len(target_points):
        raise ValueError("Buffalo L needs five landmarks.")

    equations = []
    targets = []
    for index in range(len(target_points)):
        x, y = landmarks[index][:2]
        if not np.isfinite(x) or not np.isfinite(y) or x <= 0 or y <= 0:
            raise ValueError("A facial landmark is missing.")

        target_x, target_y = target_points[index]
        equations.append([x, -y, 1, 0])
        equations.append([y, x, 0, 1])
        targets.append(target_x)
        targets.append(target_y)

    # least squares finds one rotation, uniform scale and translation
    coefficients, _, rank, _ = np.linalg.lstsq(equations, targets, rcond=None)
    a, b, shift_x, shift_y = coefficients
    if rank < 4 or (a == 0 and b == 0):
        raise ValueError("The landmarks do not define an alignment.")

    matrix = np.array([[a, -b, shift_x], [b, a, shift_y]])
    aligned_face = cv2.warpAffine(original_image, matrix, (112, 112))
    return aligned_face


def load_embedder(model_path):
    # load the recognizer once and reuse its cpu session for each image
    filename = str(model_path)
    session = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
    return session


def create_embedding(aligned_face, session):
    # convert bgr pixels to rgb, normalize them, and put channels before height and width
    rgb_image = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    float_image = rgb_image.astype(np.float32)
    centered_image = float_image - 127.5
    scaled_image = centered_image / 127.5
    channels_first = np.transpose(scaled_image, (2, 0, 1))
    model_input = np.expand_dims(channels_first, axis=0)
    model_input = np.ascontiguousarray(model_input)

    # run the network and scale the resulting vector to length one
    inputs = session.get_inputs()
    input_name = inputs[0].name
    input_values = {input_name: model_input}
    outputs = session.run(None, input_values)
    model_output = outputs[0]
    vector = model_output.reshape(-1)
    if vector.size != 512 or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 512-value Buffalo L embedding.")
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length == 0:
        raise ValueError("Cannot normalize the Buffalo L embedding.")
    embedding = vector / length
    return embedding
