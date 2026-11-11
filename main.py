from ultralytics import YOLO
from pathlib import Path
import csv
import cv2
import numpy as np
from preprocessing import crop_face, create_embedding, load_embedder
from preprocessing import prepare_embedding_face, rotate_face, transform_face_box



PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_DIR / "weights"
DETECTOR_PATH = WEIGHTS_DIR / "yolo11s-pose_widerface.pt"
RECOGNIZER_PATH = WEIGHTS_DIR / "w600k_r50.onnx"


def eredmeny_print(box, i):
    coordinates = box.xyxy[0]
    confidence = box.conf[0]
    class_id = box.cls[0]

    print(f"Face {i + 1}")
    print(f"  Coordinates: {coordinates}")
    print(f"  Confidence: {confidence:.5f}")
    print(f"  Class ID: {int(class_id)}")

def draw_BB(box, images):
    #extract bb coordinates
    x1, y1, x2, y2 = box.xyxy[0]
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    confidence = float(box.conf[0]) #detection confidence stored for display
    #bb drawing:
    cv2.rectangle(
        images,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        2
    )
    confidence_percent = 100 * confidence
    label = f"Face {confidence_percent:.2f}%"
    cv2.putText(
        images,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2
    )


def draw_landmarks(keypoints, image, scale=1.0, min_confidence=0.5):
    # five points per face. numbering preserves the model's landmark order.
    # left/right refer to the picture as viewed, not the person's anatomy.
    landmark_names = ("left eye", "right eye", "nose tip",
                      "left mouth corner", "right mouth corner")
    # coordinates are in original-image pixels; scale only changes the drawing.
    cpu_keypoints = keypoints.data.cpu()
    landmark_groups = cpu_keypoints.tolist()
    for face_number, points in enumerate(landmark_groups, start=1):
        print(f"  Landmarks for face {face_number} (original-image pixels):")
        for point_i, point in enumerate(points):
            x, y = point[:2]
            confidence = None
            if len(point) == 3:
                confidence = point[2]

            valid = x > 0 and y > 0
            confidence_text = "n/a"
            if confidence is not None:
                valid = valid and confidence >= min_confidence
                confidence_text = f"{confidence:.3f}"
            print(f"    {landmark_names[point_i]}: x={x:.2f}, y={y:.2f}, "
                  f"confidence={confidence_text}, drawn={valid}")

            if not valid:
                continue

            scaled_x = round(x * scale)
            scaled_y = round(y * scale)
            position = (scaled_x, scaled_y)
            cv2.circle(image, position, 3, (0, 0, 255), -1)
            cv2.putText(image, str(point_i),
                        (position[0] + 4, position[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

def process_aligned_faces(result, session, aligned_dir, cropped_dir, embedding_dir=None):
    processed_faces = [] # return images for the gui as well as saving them.
    if result.keypoints is None:
        return processed_faces
    source_path = Path(result.path)
    cpu_keypoints = result.keypoints.data.cpu()
    landmark_groups = cpu_keypoints.tolist()
    cpu_boxes = result.boxes.xyxy.cpu()
    boxes = cpu_boxes.tolist()

    for face_i, points in enumerate(landmark_groups):
        face_number = face_i + 1
        left_eye = points[0][:2]
        right_eye = points[1][:2]
        aligned_image, _, rotation_matrix = rotate_face(result.orig_img, left_eye, right_eye)
        face_stem = f"{source_path.stem}_face_{face_number}"
        aligned_path = aligned_dir / f"{face_stem}_aligned{source_path.suffix}"
        cv2.imwrite(str(aligned_path), aligned_image)

        # reuse the original face box instead of detecting the rotated image again
        original_box = boxes[face_i]
        rotated_box = transform_face_box(original_box, rotation_matrix)
        cropped_image = None
        try:
            cropped_image, _ = crop_face(aligned_image, rotated_box)
        except ValueError as error:
            print(f"Preview crop unavailable for {face_stem}: {error}")
        if cropped_image is not None:
            cropped_path = cropped_dir / f"{face_stem}_cropped{source_path.suffix}"
            cv2.imwrite(str(cropped_path), cropped_image)

        # the recognizer has its own crop; the box crop is for inspection
        embedding = None
        if session is not None:
            try:
                model_face = prepare_embedding_face(result.orig_img, points)
                embedding = create_embedding(model_face, session)
            except Exception as error:
                print(f"Embedding unavailable for {face_stem}: {error}")

        if embedding is not None and embedding_dir is not None:
            model_face_path = embedding_dir / f"{face_stem}_input.png"
            if not cv2.imwrite(str(model_face_path), model_face):
                raise RuntimeError(f"Could not save embedding input: {model_face_path}")
            embedding_path = embedding_dir / f"{face_stem}_embedding.npy"
            np.save(embedding_path, embedding)

        processed_faces.append((aligned_image, cropped_image, embedding))
    return processed_faces

#------------------------------------


def main():
    # the batch test reads old captures but only writes to its own folder
    input_path = PROJECT_DIR / "kepek/landmarking test2/nyers"
    output_dir = PROJECT_DIR / "kepek/embedding test1"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(DETECTOR_PATH))
    session = load_embedder(RECOGNIZER_PATH)
    results = model(str(input_path), stream=True, verbose=False)
    report_path = output_dir / "results.csv"

    # write one row per face, or an unavailable row if no face was returned
    with report_path.open("w", newline="", encoding="utf-8") as report:
        writer = csv.writer(report)
        writer.writerow(["image", "face", "status", "preview_file", "input_file", "embedding_file"])
        for result in results:
            source_path = Path(result.path)
            faces = process_aligned_faces(result, session, output_dir, output_dir,
                                          embedding_dir=output_dir)
            if not faces:
                writer.writerow([source_path.name, "", "unavailable", "", "", ""])
            for face_number, face in enumerate(faces, start=1):
                _, cropped_image, embedding = face
                face_stem = f"{source_path.stem}_face_{face_number}"
                preview_name = f"{face_stem}_cropped{source_path.suffix}"
                if cropped_image is None:
                    preview_name = ""
                if embedding is None:
                    row = [source_path.name, face_number, "unavailable", preview_name, "", ""]
                else:
                    model_face_name = f"{face_stem}_input.png"
                    vector_name = f"{face_stem}_embedding.npy"
                    row = [source_path.name, face_number, "ok", preview_name, model_face_name, vector_name]

                writer.writerow(row)
    print(f"Embedding test results: {report_path}")


if __name__ == "__main__":
    main()
