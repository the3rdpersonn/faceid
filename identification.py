"""Save face references and compare photos against enrolled people."""

import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from preprocessing import create_embedding, prepare_embedding_face


REFERENCE_POSES = ("front", "slight_left", "left", "slight_right", "right")
REFERENCE_COLUMNS = ("person_id", "pose", "turn_score", "image_file", "embedding_file")
NAMED_REFERENCE_COLUMNS = (*REFERENCE_COLUMNS, "display_name")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def save_scan(parent, captures, session, progress, person_id=None, display_name=""):
    # each completed capture set gets a new folder under the selected parent
    poses = []
    for pose, _, _, _ in captures:
        poses.append(pose)
    if tuple(poses) != REFERENCE_POSES:
        raise ValueError("The scan must contain one capture of each reference pose.")

    # a new enrollment gets one id, shared by all five reference images
    # generate an identity only for a new person so rescans keep the existing identity
    if person_id is None:
        person_id = f"person_{uuid4().hex}"

    capture_time = datetime.now()
    timestamp = capture_time.strftime("%Y%m%d_%H%M%S_%f")
    scan_dir = Path(parent) / f"scan_{timestamp}"
    scan_dir.mkdir()
    image_dir = scan_dir / "images"
    image_dir.mkdir()
    embedding_dir = scan_dir / "embeddings"
    embedding_dir.mkdir()

    # collect the manifest entries while the reference images and vectors are saved
    rows = []
    for number, capture in enumerate(captures, start=1):
        pose, score, frame, landmarks = capture
        progress(f"Saving reference {number} of {len(captures)}")
        # prepare the face pixels in the format expected by the embedding network
        aligned_face = prepare_embedding_face(frame, landmarks)
        embedding = create_embedding(aligned_face, session)

        stem = f"{number:02d}_{pose}"
        image_path = image_dir / f"{stem}.png"
        embedding_path = embedding_dir / f"{stem}.npy"
        # use pathlib for the actual write so windows folder names can contain accents
        _, encoded = cv2.imencode(".png", frame)
        image_bytes = encoded.tobytes()
        image_path.write_bytes(image_bytes)
        np.save(embedding_path, embedding)

        # relative paths allow the complete scan folder to be moved later
        relative_image_path = image_path.relative_to(scan_dir)
        relative_embedding_path = embedding_path.relative_to(scan_dir)
        relative_image = relative_image_path.as_posix()
        relative_embedding = relative_embedding_path.as_posix()
        rows.append({
            "person_id": person_id,
            "pose": pose,
            "turn_score": f"{score:.6f}",
            "image_file": relative_image,
            "embedding_file": relative_embedding,
            "display_name": display_name,
        })

    # publish the manifest only when every image and embedding is ready
    write_reference_rows(scan_dir, rows)
    return scan_dir


def write_reference_rows(scan_dir, rows):
    # replace the csv only after the new copy has been written successfully
    temporary_manifest = scan_dir / f"references_{uuid4().hex}.tmp"
    try:
        with temporary_manifest.open("w", newline="", encoding="utf-8") as report:
            writer = csv.DictWriter(report, fieldnames=NAMED_REFERENCE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        # replace the manifest in one step after its new contents are fully written
        temporary_manifest.replace(scan_dir / "references.csv")
    finally:
        temporary_manifest.unlink(missing_ok=True)


def reference_file(scan_dir, filename):
    # a manifest can only point to files within its selected scan folder
    relative_path = Path(filename)
    path = scan_dir / relative_path
    path = path.resolve()
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
    # sets collect distinct values so the scan can be checked for conflicting identities
    names = set()
    person_ids = set()
    with manifest.open(newline="", encoding="utf-8") as report:
        reader = csv.DictReader(report)
        for row in reader:
            person_ids.add(row["person_id"])
            # older scans have no display name column
            name = row.get("display_name", "")
            names.add(name)
            rows.append(row)
            poses.append(row["pose"])
    if tuple(poses) != REFERENCE_POSES:
        raise ValueError("The selected scan must contain all five reference poses.")
    if len(person_ids) != 1:
        raise ValueError("All references in one scan must belong to the same person.")
    if len(names) != 1:
        raise ValueError("The reference rows have different display names.")
    return scan_dir, rows, person_ids.pop(), names.pop()


def read_scan_identity(folder):
    _, _, person_id, display_name = read_reference_rows(folder)
    return person_id, display_name


def set_scan_name(folder, display_name):
    scan_dir, rows, _, _ = read_reference_rows(folder)
    # apply this operation to each of the five pose references
    for row in rows:
        row["display_name"] = display_name
    write_reference_rows(scan_dir, rows)


@dataclass
class ReferenceProfile:
    person_id: str
    display_name: str
    embeddings: list


def load_reference_scan(folder):
    scan_dir, rows, person_id, display_name = read_reference_rows(folder)
    references = []
    for row in rows:
        reference_file(scan_dir, row["image_file"])
        embedding_path = reference_file(scan_dir, row["embedding_file"])

        # numpy loads the saved numbers without allowing python objects in the file
        embedding = np.load(embedding_path, allow_pickle=False)
        if embedding.shape != (512,) or embedding.dtype != np.float32:
            raise ValueError(f"Expected a 512-value float32 vector in {embedding_path.name}.")
        finite_values = np.isfinite(embedding)
        length = np.linalg.norm(embedding)
        unit_length = np.isclose(length, 1)
        if not finite_values.all() or not unit_length:
            raise ValueError(f"Expected a finite unit vector in {embedding_path.name}.")
        references.append(embedding)
    return ReferenceProfile(person_id, display_name, references)


def load_references(folder):
    profile = load_reference_scan(folder)
    return profile.embeddings


def load_reference_gallery(folder):
    profiles = {}
    # scan folder names contain the capture time, so the newest scans come first
    gallery_folder = Path(folder)
    manifests = gallery_folder.glob("scan_*/references.csv")
    # sorting these timestamped paths in reverse places the latest scan first
    newest_first = sorted(manifests, reverse=True)
    for manifest in newest_first:
        scan_folder = manifest.parent
        person_id, _ = read_scan_identity(scan_folder)
        # keep only the newest scan encountered for each person
        if person_id not in profiles:
            profiles[person_id] = load_reference_scan(scan_folder)
    if not profiles:
        raise ValueError("Choose a gallery folder containing completed scan folders.")
    return list(profiles.values())


def match_embedding(query, profiles, cutoff=None):
    if not profiles:
        raise ValueError("At least one reference profile is required.")
    if cutoff is not None and (not isfinite(cutoff) or not -1 <= cutoff <= 1):
        raise ValueError("Enter a cutoff between -1 and 1, or leave it blank.")

    # start below every possible cosine score so the first profile can become the best
    best_score = float("-inf")
    winner = None
    for profile in profiles:
        person_score = -1.0
        # each dot product is cosine similarity because both vectors have unit length
        for reference in profile.embeddings:
            dot_product = np.dot(query, reference)
            score = float(dot_product)
            person_score = max(person_score, score)
        person_score = min(person_score, 1.0)
        if person_score > best_score:
            best_score = person_score
            winner = profile
        elif person_score == best_score:
            # an exact tie between two enrolled people has no unique winner
            winner = None

    if cutoff is None or best_score < cutoff:
        winner = None
    return best_score, winner


@dataclass
class PhotoResult:
    path: Path
    face_number: int | None
    box: list | None
    score: float | None
    result: str
    person_id: str | None = None
    display_name: str = ""


def read_photo(path):
    # decode bytes so paths containing unicode also work on windows
    image_bytes = path.read_bytes()
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def score_photo(path, detect, session, profiles, cutoff):
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

    cpu_boxes = detection.boxes.xyxy.cpu()
    boxes = cpu_boxes.tolist()
    landmark_groups = []
    if detection.keypoints is not None:
        cpu_keypoints = detection.keypoints.data.cpu()
        landmark_groups = cpu_keypoints.tolist()

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

        score, winner = match_embedding(query, profiles, cutoff)
        if winner is not None:
            row = PhotoResult(path, face_number, box, score, "Matched",
                              winner.person_id, winner.display_name)
        elif cutoff is None:
            row = PhotoResult(path, face_number, box, score, "Score only")
        else:
            row = PhotoResult(path, face_number, box, score, "unknown")
        results.append(row)
    return results
