"""The live webcam view used to collect enrollment references."""

from pathlib import Path
from queue import Empty, Queue
from statistics import median
from threading import Thread
from time import monotonic
from tkinter import Button, Entry, Frame, Label, filedialog, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from identification import IMAGE_SUFFIXES, PERSON_ID, load_reference_scan, read_photo
from identification import read_scan_name, save_scan, score_photo, set_scan_name
from preprocessing import measure_turn


BACKGROUND = "#101010"
LANDMARK_LABELS = ("Left eye", "Right eye", "Nose tip", "Left mouth corner", "Right mouth corner")
READOUT_INTERVAL = 0.15
OBSERVATIONS_NEEDED = 3
PROMPT_TIMEOUT = 8
CAPTURE_STEPS = (
    ("front", "Look straight at the camera"),
    ("slight_left", "Turn slightly toward the left of the preview"),
    ("left", "Turn a little farther left"),
    ("center", "Come back to the center"),
    ("slight_right", "Turn slightly toward the right of the preview"),
    ("right", "Turn a little farther right"),
)


def read_pose(result, frame):
    # accept one face with five usable points in the original frame
    if result.boxes is None or len(result.boxes) != 1 or result.keypoints is None:
        return None
    groups = result.keypoints.data.cpu().tolist()
    if len(groups) != 1 or len(groups[0]) != 5:
        return None

    landmarks = groups[0]
    height, width = frame.shape[:2]
    for point in landmarks:
        x, y = point[:2]
        if not (0 < x < width and 0 < y < height):
            return None
        if len(point) > 2 and not point[2] >= 0.5:
            return None

    try:
        score = measure_turn(landmarks[0], landmarks[1], landmarks[2])
    except ValueError:
        return None

    left, top, right, bottom = result.boxes.xyxy.cpu().tolist()[0]
    box_width = right - left
    box_height = bottom - top
    return score, landmarks, box_width, box_height


def draw_test_box(image, box, caption):
    x1, y1, x2, y2 = box
    start = (round(x1), round(y1))
    end = (round(x2), round(y2))
    cv2.rectangle(image, start, end, (255, 255, 0), 3)

    # pillow keeps accented names readable when drawing them onto a photo
    height, width = image.shape[:2]
    preview_ratio = max(width / 960, height / 540)
    font_size = max(20, round(18 * preview_ratio))
    try:
        font = ImageFont.truetype("segoeui.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    photo = Image.fromarray(rgb_image)
    drawing = ImageDraw.Draw(photo)
    caption = " ".join(caption.split())
    left, top, right, bottom = drawing.textbbox((0, 0), caption, font=font)
    padding = 4
    label_width = right - left + 2 * padding
    label_height = bottom - top + 2 * padding

    label_x = max(0, min(start[0], width - label_width))
    label_y = max(0, start[1] - label_height - 3)
    drawing.rectangle((label_x, label_y, label_x + label_width, label_y + label_height),
                      fill="black")
    text_origin = (label_x + padding - left, label_y + padding - top)
    drawing.text(text_origin, caption, font=font, fill=(0, 255, 255))
    return cv2.cvtColor(np.asarray(photo), cv2.COLOR_RGB2BGR)


def accepted_turn(pose, score_from_front):
    # use the direction as well as the size of the change
    if pose == "center":
        return abs(score_from_front) <= 0.04
    if pose.endswith("left"):
        if score_from_front >= 0:
            return False
    elif score_from_front <= 0:
        return False

    turn = abs(score_from_front)
    if pose.startswith("slight"):
        return 0.08 <= turn < 0.16
    return 0.16 <= turn <= 0.30


class CaptureSequence:
    def __init__(self):
        self.index = 0
        self.baseline = None
        self.recent_scores = []
        self.valid_count = 0
        self.captures = []

    def observe(self, frame, measurement):
        # return true when this observation completes the current prompt
        if measurement is None:
            self.recent_scores.clear()
            self.valid_count = 0
            return False
        score, landmarks, width, height = measurement
        if width < frame.shape[1] * 0.15 or height < frame.shape[0] * 0.15:
            self.recent_scores.clear()
            self.valid_count = 0
            return False

        pose, _ = CAPTURE_STEPS[self.index]
        if pose == "front":
            self.recent_scores.append(score)
            if max(self.recent_scores) - min(self.recent_scores) > 0.02:
                self.recent_scores = [score]
            if len(self.recent_scores) < OBSERVATIONS_NEEDED:
                return False
            self.baseline = median(self.recent_scores)
        else:
            if not accepted_turn(pose, score - self.baseline):
                self.valid_count = 0
                return False
            self.valid_count += 1
            if self.valid_count < OBSERVATIONS_NEEDED:
                return False

        if pose != "center":
            # keep the source frame paired with the points detected on it
            self.captures.append((pose, score - self.baseline, frame, landmarks))
        self.index += 1
        self.valid_count = 0
        return True


class ScanView(Frame):
    def __init__(self, parent, detect, get_embedder, show_image):
        super().__init__(parent, bg=BACKGROUND)
        self.detect = detect
        self.get_embedder = get_embedder
        self.show_image = show_image
        self.messages = Queue()
        self.visible = False
        self.job = None
        self.sequence = None
        self.save_parent = None
        self.reference_folder = None
        self.test_folder = None
        self.prompt_time = 0
        self.photo_results = []
        self.test_display_name = ""
        self.review_image = None
        self.detection_pending = False
        self.generation = 0
        self.last_detection = 0
        self.has_readout_frame = False

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.prompt = Label(self, text="Face the camera", bg=BACKGROUND, fg="white",
                            font=("Segoe UI", 18), bd=0, padx=16, pady=8,
                            highlightthickness=1, highlightbackground="white",
                            highlightcolor="white")
        self.prompt.grid(row=0, column=0, pady=14)

        self.preview = Label(self, bg="black", fg="white", width=1, height=1)
        self.preview.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.preview.source_image = None
        self.preview.display_limit = (960, 540)
        self.preview.bind("<Configure>", self.redraw_preview)

        self.sidebar = Frame(self, bg=BACKGROUND)
        self.sidebar.grid(row=0, column=1, rowspan=3, sticky="ns", padx=(4, 14), pady=14)
        self.readout = Label(self.sidebar, bg=BACKGROUND, fg="white", width=29,
                             justify="left", anchor="w", font=("Consolas", 10))
        self.readout.pack(fill="x")
        self.update_readout(None)

        self.choose_save_button = self.sidebar_button("Choose save folder", self.choose_save_folder)
        self.save_location = self.sidebar_label("No save folder selected")
        self.start_button = self.sidebar_button("Start scan", self.start_scan)
        self.cancel_button = self.sidebar_button("Cancel", self.cancel_scan)
        self.status = self.sidebar_label("")

        self.choose_reference_button = self.sidebar_button("Select reference scan", self.choose_reference_scan)
        self.reference_location = self.sidebar_label("No reference scan selected")
        self.name_button = self.sidebar_button("Name reference scan", self.name_reference_scan)
        self.choose_test_button = self.sidebar_button("Select test image folder", self.choose_test_folder)
        self.test_location = self.sidebar_label("No test image folder selected")
        self.sidebar_label("Score cutoff (optional)")
        self.cutoff = Entry(self.sidebar)
        self.cutoff.pack(fill="x")
        test_controls = Frame(self.sidebar, bg=BACKGROUND)
        test_controls.pack(fill="x", pady=2)
        self.test_button = Button(test_controls, text="Run test", command=self.run_test)
        self.test_button.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.print_button = Button(test_controls, text="Print results", command=self.print_results)
        self.print_button.pack(side="left", fill="x", expand=True, padx=(2, 0))
        self.sidebar_button("Live preview", self.return_to_live)

        columns = ("image", "face", "score", "result")
        self.table = ttk.Treeview(self, columns=columns, show="headings", height=6)
        self.table.heading("image", text="Image")
        self.table.heading("face", text="Face")
        self.table.heading("score", text="Score")
        self.table.heading("result", text="Result")
        self.table.column("image", width=220, minwidth=70)
        self.table.column("face", width=42, minwidth=36, stretch=False, anchor="center")
        self.table.column("score", width=70, minwidth=55, stretch=False, anchor="center")
        self.table.column("result", width=150, minwidth=80)
        self.table.bind("<<TreeviewSelect>>", self.review_result)
        self.table_scroll = ttk.Scrollbar(self, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=self.table_scroll.set)
        self.refresh_controls()

    @property
    def busy(self):
        return self.sequence is not None or self.job is not None

    def sidebar_button(self, text, action):
        button = Button(self.sidebar, text=text, command=action)
        button.pack(fill="x", pady=2)
        return button

    def sidebar_label(self, text):
        label = Label(self.sidebar, text=text, bg=BACKGROUND, fg="white",
                      justify="left", anchor="w", wraplength=240)
        label.pack(fill="x", pady=1)
        return label

    def refresh_controls(self):
        # the same camera and models are shared, so a scan and a test take turns
        selection_state = "disabled" if self.busy else "normal"
        self.choose_save_button.configure(state=selection_state)
        self.choose_reference_button.configure(state=selection_state)
        self.choose_test_button.configure(state=selection_state)
        self.cutoff.configure(state=selection_state)
        can_name = not self.busy and self.reference_folder is not None
        self.name_button.configure(state="normal" if can_name else "disabled")
        can_start = not self.busy and self.save_parent is not None
        self.start_button.configure(state="normal" if can_start else "disabled")
        self.cancel_button.configure(state="normal" if self.sequence is not None else "disabled")
        can_test = not self.busy and self.reference_folder is not None and self.test_folder is not None
        self.test_button.configure(state="normal" if can_test else "disabled")
        can_print = not self.busy and bool(self.photo_results)
        self.print_button.configure(state="normal" if can_print else "disabled")

    def redraw_preview(self, _event):
        if self.visible and self.preview.source_image is not None:
            self.show_image(self.preview, self.preview.source_image)

    def set_visible(self, visible):
        self.visible = visible
        self.generation += 1
        self.has_readout_frame = False
        self.update_readout(None)
        if visible:
            self.pack(fill="both", expand=True)
            self.return_to_live()
        else:
            self.pack_forget()

    def show_frame(self, frame):
        # send only the newest frame when the detector has finished its previous job
        if self.review_image is not None:
            return
        if not self.has_readout_frame:
            self.show_image(self.preview, frame)
        if self.job is not None or self.detection_pending:
            return
        now = monotonic()
        if now - self.last_detection < READOUT_INTERVAL:
            return
        self.last_detection = now
        self.detection_pending = True
        snapshot = frame.copy()
        worker = Thread(target=self.detect_pose, args=(snapshot, self.generation), daemon=True)
        worker.start()

    def detect_pose(self, frame, generation):
        try:
            result = self.detect(frame, quiet=True)
            measurement = read_pose(result, frame)
            self.messages.put(("pose", generation, (frame, measurement, None)))
        except Exception as error:
            self.messages.put(("pose", generation, (frame, None, str(error))))

    def poll_results(self):
        while True:
            try:
                kind, generation, value = self.messages.get_nowait()
            except Empty:
                break
            if kind == "pose":
                self.detection_pending = False
                if self.visible and generation == self.generation:
                    frame, measurement, error = value
                    self.show_pose(frame, measurement, error)
            elif generation == self.generation:
                self.handle_job_message(kind, value)
        if self.sequence is not None and monotonic() - self.prompt_time >= PROMPT_TIMEOUT:
            self.repeat_prompt()

    def show_pose(self, frame, measurement, error):
        self.update_readout(measurement)
        if error is not None:
            self.readout.configure(text="Landmarks unavailable")
        annotated = frame
        if measurement is not None:
            annotated = frame.copy()
            for point in measurement[1]:
                center = (round(point[0]), round(point[1]))
                cv2.circle(annotated, center, 4, (255, 255, 0), -1)
        self.show_image(self.preview, annotated)
        self.has_readout_frame = True
        if self.sequence is not None:
            self.observe_capture(frame, measurement)

    def update_readout(self, measurement):
        if measurement is None:
            self.readout.configure(text="Waiting for one visible face")
            return
        score, landmarks, width, height = measurement
        lines = [f"Turn score: {score:+.3f}", f"Face box: {width:.0f} × {height:.0f} px"]
        if self.sequence is not None and self.sequence.baseline is not None:
            lines.insert(1, f"From front: {score - self.sequence.baseline:+.3f}")
        for label, point in zip(LANDMARK_LABELS, landmarks):
            lines.append(f"{label}: {point[0]:.0f}, {point[1]:.0f}")
        self.readout.configure(text="\n".join(lines))

    def choose_save_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Choose where to save new scans", mustexist=True)
        if folder:
            self.save_parent = Path(folder)
            self.save_location.configure(text=str(self.save_parent))
            self.refresh_controls()

    def choose_reference_scan(self):
        folder = filedialog.askdirectory(parent=self, title="Select a completed scan folder", mustexist=True)
        if folder:
            scan_path = Path(folder)
            try:
                display_name = read_scan_name(scan_path)
            except (OSError, ValueError) as error:
                self.status.configure(text=f"Could not select scan: {error}")
                return
            self.reference_folder = scan_path
            self.show_reference_location(display_name)
            self.refresh_controls()

    def show_reference_location(self, display_name):
        label = PERSON_ID
        if display_name:
            label = f"{display_name} ({PERSON_ID})"
        self.reference_location.configure(text=f"{label}\n{self.reference_folder}")

    def name_reference_scan(self):
        if self.busy or self.reference_folder is None:
            return

        try:
            current_name = read_scan_name(self.reference_folder)
            # the optional display name stays separate from the stable person id
            name = simpledialog.askstring(
                "Name reference scan",
                f"Optional display name for {PERSON_ID}:",
                initialvalue=current_name,
                parent=self.winfo_toplevel(),
            )
            if name is None:
                return
            display_name = name.strip()
            if display_name != current_name:
                set_scan_name(self.reference_folder, display_name)
            self.show_reference_location(display_name)
            self.status.configure(text=f"Reference saved:\n{self.reference_folder}")
        except (OSError, ValueError) as error:
            self.status.configure(text=f"Could not update the display name: {error}")

    def choose_test_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Select a folder of test photos", mustexist=True)
        if folder:
            self.test_folder = Path(folder)
            self.test_location.configure(text=str(self.test_folder))
            self.refresh_controls()

    def return_to_live(self):
        if self.busy:
            return
        self.review_image = None
        self.preview.source_image = None
        self.has_readout_frame = False
        self.prompt.configure(text="Face the camera")

    def start_scan(self):
        if self.busy or self.save_parent is None:
            return
        self.return_to_live()
        self.generation += 1
        self.sequence = CaptureSequence()
        self.set_prompt()
        self.status.configure(text="Capturing reference poses")
        self.refresh_controls()

    def set_prompt(self):
        _, instruction = CAPTURE_STEPS[self.sequence.index]
        self.prompt.configure(text=instruction)
        self.prompt_time = monotonic()

    def repeat_prompt(self):
        self.set_prompt()
        self.status.configure(text="Still waiting for this pose. Hold it steady to capture.")

    def cancel_scan(self):
        if self.sequence is None:
            return
        self.sequence = None
        self.generation += 1
        self.return_to_live()
        self.status.configure(text="Scan canceled")
        self.refresh_controls()

    def observe_capture(self, frame, measurement):
        if not self.sequence.observe(frame, measurement):
            return
        if self.sequence.index < len(CAPTURE_STEPS):
            self.set_prompt()
            count = len(self.sequence.captures)
            self.status.configure(text=f"{count} of 5 reference views captured")
            return

        captures = self.sequence.captures
        self.sequence = None
        self.job = "saving"
        self.generation += 1
        self.prompt.configure(text="Five views captured")
        self.status.configure(text="Creating and saving embeddings…")
        self.refresh_controls()
        worker = Thread(target=self.save_references,
                        args=(self.save_parent, captures, self.generation), daemon=True)
        worker.start()

    def save_references(self, folder, captures, generation):
        try:
            session = self.get_embedder()

            def progress(message):
                self.messages.put(("progress", generation, message))

            saved_path = save_scan(folder, captures, session, progress)
            self.messages.put(("saved", generation, saved_path))
        except Exception as error:
            self.messages.put(("error", generation, f"Could not save scan: {error}"))

    def run_test(self):
        if self.busy or self.reference_folder is None or self.test_folder is None:
            return

        cutoff_text = self.cutoff.get().strip()
        cutoff = None
        if cutoff_text:
            try:
                cutoff = float(cutoff_text)
                if not -1 <= cutoff <= 1:
                    raise ValueError
            except ValueError:
                self.status.configure(text="Enter a cutoff between -1 and 1, or leave it blank.")
                return

        self.return_to_live()
        self.job = "testing"
        self.generation += 1
        self.prompt.configure(text="Testing photos")
        self.status.configure(text="Loading reference scan…")
        self.refresh_controls()
        worker = Thread(target=self.test_photos,
                        args=(self.reference_folder, self.test_folder, cutoff, self.generation), daemon=True)
        worker.start()

    def test_photos(self, reference_folder, test_folder, cutoff, generation):
        try:
            references, display_name = load_reference_scan(reference_folder)
            paths = []
            for path in sorted(test_folder.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    paths.append(path)
            if not paths:
                raise ValueError("The test folder contains no supported image files.")

            session = self.get_embedder()
            results = []
            for number, path in enumerate(paths, start=1):
                message = f"Testing photo {number} of {len(paths)}"
                self.messages.put(("progress", generation, message))
                rows = score_photo(path, self.detect, session, references, cutoff)
                results.extend(rows)
            self.messages.put(("tested", generation, (results, display_name)))
        except Exception as error:
            self.messages.put(("error", generation, f"Could not run test: {error}"))

    def handle_job_message(self, kind, value):
        if kind == "progress":
            self.status.configure(text=value)
            return

        finished_job = self.job
        self.job = None
        self.return_to_live()
        if kind == "saved":
            self.reference_folder = value
            self.show_reference_location("")
            self.prompt.configure(text="Scan complete")
            self.status.configure(text=f"Saved scan:\n{value}")
            self.name_reference_scan()
        elif kind == "tested":
            results, display_name = value
            self.show_test_results(results, display_name)
            self.prompt.configure(text="Test complete")
            self.status.configure(text="Test complete. Select a row to locate its face.")
        elif kind == "error":
            if finished_job == "saving":
                self.prompt.configure(text="Scan could not be saved")
            else:
                self.prompt.configure(text="Test could not run")
            self.status.configure(text=value)
        self.refresh_controls()

    def show_test_results(self, results, display_name=""):
        self.photo_results = results
        self.test_display_name = display_name
        for item in self.table.get_children():
            self.table.delete(item)
        for index, result in enumerate(results):
            face = "" if result.face_number is None else result.face_number
            score = "" if result.score is None else f"{result.score:.4f}"
            outcome = result.result
            if outcome == PERSON_ID and display_name:
                outcome = f"{display_name} ({PERSON_ID})"
            self.table.insert("", "end", iid=str(index),
                              values=(result.path.name, face, score, outcome))
        self.table.grid(row=2, column=0, sticky="ew", padx=(12, 28), pady=(0, 12))
        self.table_scroll.grid(row=2, column=0, sticky="nse", padx=(0, 12), pady=(0, 12))

    def print_results(self):
        # use the same rows as the results table, including the enrolled display name
        if self.busy or not self.photo_results:
            return
        print("Photo test results")
        print("Image | Face | Score | Result")
        for item in self.table.get_children():
            image, face, score, outcome = self.table.item(item, "values")
            face_text = "-" if face == "" else face
            score_text = "-" if score == "" else score
            print(f"{image} | {face_text} | {score_text} | {outcome}")
        self.status.configure(text="Results printed to the terminal.")

    def review_result(self, _event):
        selection = self.table.selection()
        if self.busy or not selection:
            return
        result = self.photo_results[int(selection[0])]
        try:
            image = read_photo(result.path)
        except (OSError, cv2.error):
            image = None
        if image is None:
            self.status.configure(text="Could not open this test photo.")
            return
        if result.box is not None:
            caption = f"Face {result.face_number}"
            if result.result == PERSON_ID:
                name = self.test_display_name or PERSON_ID
                caption = f"{caption}: {name}"
            image = draw_test_box(image, result.box, caption)
        self.generation += 1
        self.review_image = image
        self.update_readout(None)
        self.prompt.configure(text="Test photo")
        self.show_image(self.preview, image)
