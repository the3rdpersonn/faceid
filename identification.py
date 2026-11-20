"""Save, reload, and compare references for the single-person scan."""

import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from preprocessing import create_embedding, prepare_embedding_face


PERSON_ID = "person_001"
REFERENCE_POSES = ("front", "slight_left", "left", "slight_right", "right")
REFERENCE_COLUMNS = ("person_id", "pose", "turn_score", "image_file", "embedding_file")
NAMED_REFERENCE_COLUMNS = (*REFERENCE_COLUMNS, "display_name")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def save_scan(parent, captures, session, progress):
    # each completed capture set gets a new folder under the selected parent
    poses = []
    for pose, _, _, _ in captures:
        poses.append(pose)
    if tuple(poses) != REFERENCE_POSES:
        raise ValueError("The scan must contain one capture of each reference pose.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    scan_dir = Path(parent) / f"scan_{timestamp}"
    scan_dir.mkdir()
    image_dir = scan_dir / "images"
    image_dir.mkdir()
    embedding_dir = scan_dir / "embeddings"
    embedding_dir.mkdir()

    rows = []
    for number, capture in enumerate(captures, start=1):
        pose, score, frame, landmarks = capture
        progress(f"Saving reference {number} of {len(captures)}")
        aligned_face = prepare_embedding_face(frame, landmarks)
        embedding = create_embedding(aligned_face, session)

        stem = f"{number:02d}_{pose}"
        image_path = image_dir / f"{stem}.png"
        embedding_path = embedding_dir / f"{stem}.npy"
        # use pathlib for the actual write so Windows folder names can contain accents
        saved, encoded = cv2.imencode(".png", frame)
        if not saved:
            raise RuntimeError(f"Could not encode {image_path.name}.")
        image_path.write_bytes(encoded.tobytes())
        np.save(embedding_path, embedding)

        relative_image = image_path.relative_to(scan_dir).as_posix()
        relative_embedding = embedding_path.relative_to(scan_dir).as_posix()
        rows.append({
            "person_id": PERSON_ID,
            "pose": pose,
            "turn_score": f"{score:.6f}",
            "image_file": relative_image,
            "embedding_file": relative_embedding,
            "display_name": "",
        })

    # publish the manifest only when every image and embedding is ready
    write_reference_rows(scan_dir, rows)
    return scan_dir


def write_reference_rows(scan_dir, rows):
    # replace the CSV only after the new copy has been written successfully
    temporary_manifest = scan_dir / f"references_{uuid4().hex}.tmp"
    try:
        with temporary_manifest.open("w", newline="", encoding="utf-8") as report:
            writer = csv.DictWriter(report, fieldnames=NAMED_REFERENCE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_manifest.replace(scan_dir / "references.csv")
    finally:
        temporary_manifest.unlink(missing_ok=True)


def reference_file(scan_dir, filename):
    # a manifest can only point to files within its selected scan folder
    relative_path = Path(filename)
    path = (scan_dir / relative_path).resolve()
    if relative_path.is_absolute() or not path.is_relative_to(scan_dir) or not path.is_file():
        raise ValueError(f"Missing or invalid reference file: {filename}")
    return path


def read_reference_rows(folder):
    scan_dir = Path(folder).resolve()
    manifest = scan_dir / "references.csv"
    if not manifest.is_file():
        raise ValueError("Choose a completed scan folder containing references.csv.")

    rows = []
    poses = []
    names = set()
    with manifest.open(newline="", encoding="utf-8") as report:
        reader = csv.DictReader(report)
        columns = tuple(reader.fieldnames or ())
        if columns not in (REFERENCE_COLUMNS, NAMED_REFERENCE_COLUMNS):
            raise ValueError("The reference manifest has unexpected columns.")
        for row in reader:
            if row["person_id"] != PERSON_ID or None in row or None in row.values():
                raise ValueError("Each reference must belong to person_001 and have a complete row.")
            # older scans have no display name column
            name = row.get("display_name", "")
            names.add(name)
            rows.append(row)
            poses.append(row["pose"])
    if tuple(poses) != REFERENCE_POSES:
        raise ValueError("The selected scan must contain all five reference poses.")
    if len(names) != 1:
        raise ValueError("The reference rows have different display names.")
    return scan_dir, rows, names.pop()


def read_scan_name(folder):
    _, _, display_name = read_reference_rows(folder)
    return display_name


def set_scan_name(folder, display_name):
    scan_dir, rows, _ = read_reference_rows(folder)
    for row in rows:
        row["display_name"] = display_name
    write_reference_rows(scan_dir, rows)


def load_reference_scan(folder):
    scan_dir, rows, display_name = read_reference_rows(folder)
    references = []
    for row in rows:
        image_path = reference_file(scan_dir, row["image_file"])
        embedding_path = reference_file(scan_dir, row["embedding_file"])
        if image_path.suffix != ".png" or embedding_path.suffix != ".npy":
            raise ValueError("The reference files must be PNG images and NumPy vectors.")
        if image_path.stem != embedding_path.stem:
            raise ValueError("A reference image and its vector must have the same name.")

        # NumPy loads the saved numbers without allowing Python objects in the file
        embedding = np.load(embedding_path, allow_pickle=False)
        if embedding.shape != (512,) or embedding.dtype != np.float32:
            raise ValueError(f"Expected a 512-value float32 vector in {embedding_path.name}.")
        if not np.isfinite(embedding).all() or not np.isclose(np.linalg.norm(embedding), 1):
            raise ValueError(f"Expected a finite unit vector in {embedding_path.name}.")
        references.append(embedding)
    return references, display_name


def load_references(folder):
    references, _ = load_reference_scan(folder)
    return references


def match_embedding(query, references, cutoff=None):
    if not references:
        raise ValueError("At least one reference embedding is required.")
    if cutoff is not None and (not isfinite(cutoff) or not -1 <= cutoff <= 1):
        raise ValueError("Enter a cutoff between -1 and 1, or leave it blank.")

    # the query and references are unit vectors, so their dot product is cosine similarity
    best_score = -1.0
    for reference in references:
        score = float(np.dot(query, reference))
        best_score = max(best_score, score)
    best_score = min(best_score, 1.0)

    identity = None
    if cutoff is not None:
        if best_score >= cutoff:
            identity = PERSON_ID
        else:
            identity = "unknown"
    return best_score, identity


@dataclass
class PhotoResult:
    path: Path
    face_number: int | None
    box: list | None
    score: float | None
    result: str


def read_photo(path):
    # decode bytes so paths containing Unicode also work on Windows
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def score_photo(path, detect, session, references, cutoff):
    try:
        image = read_photo(path)
    except (OSError, cv2.error):
        image = None
    if image is None:
        return [PhotoResult(path, None, None, None, "Could not read image")]

    try:
        detection = detect(image, quiet=True)
    except Exception:
        return [PhotoResult(path, None, None, None, "Detection unavailable")]
    if detection.boxes is None or len(detection.boxes) == 0:
        return [PhotoResult(path, None, None, None, "No faces detected")]

    boxes = detection.boxes.xyxy.cpu().tolist()
    landmark_groups = []
    if detection.keypoints is not None:
        landmark_groups = detection.keypoints.data.cpu().tolist()

    results = []
    for index, box in enumerate(boxes):
        face_number = index + 1
        try:
            landmarks = landmark_groups[index]
            aligned_face = prepare_embedding_face(image, landmarks)
            query = create_embedding(aligned_face, session)
        except Exception:
            results.append(PhotoResult(path, face_number, box, None, "Embedding unavailable"))
            continue
        score, identity = match_embedding(query, references, cutoff)
        if identity is None:
            identity = "Score only"
        results.append(PhotoResult(path, face_number, box, score, identity))
    return results
