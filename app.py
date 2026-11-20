from tkinter import *
from pathlib import Path
from datetime import datetime as dt
from queue import Queue, Empty
from threading import Lock, Thread

import cv2
from cam_input import *
from scan import ScanView

script_path = Path(__file__).resolve()
project_dir = script_path.parent
input_dir = project_dir / "kepek/landmarking test2/nyers"
# path type object. resolve() ->absolute path - the complete location of the script
# parent: the code + "kepek/test2/nyers" joins folder with the input folder
# advantage: this stays relative to the scripts location, regardless where i run it from.
input_dir.mkdir(parents=True, exist_ok=True) #in case thee folder path is incomplete, it creates missing folders


window = Tk() #initiates the window
window.title("Face processing")
window.geometry("1280x800")
window.minsize(640, 480)

mode_bar = Frame(window)
mode_bar.pack(fill=X)

befoglalo = Frame(window)
befoglalo.pack(fill=BOTH, expand = True)
for index in range(2):
    befoglalo.grid_rowconfigure(index, weight=1, uniform="image_rows")
    befoglalo.grid_columnconfigure(index, weight=1, uniform="image_columns")


def show_image(label, image):
    # keep the original pixels; resizing here is for display only.
    label.source_image = image
    width = label.winfo_width() - 8
    height = label.winfo_height() - 8
    limit = getattr(label, "display_limit", None)
    if limit is not None:
        width = min(width, limit[0])
        height = min(height, limit[1])
    if width <= 0 or height <= 0:
        return # the configure event will draw once the panel has a size.

    image_height, image_width = image.shape[:2]
    width_scale = width / image_width
    height_scale = height / image_height
    scale = min(width_scale, height_scale)
    scaled_width = int(image_width * scale)
    scaled_height = int(image_height * scale)
    display_width = max(1, scaled_width)
    display_height = max(1, scaled_height)
    display_size = (display_width, display_height)

    interpolation = cv2.INTER_LINEAR
    if scale < 1:
        interpolation = cv2.INTER_AREA
    display_image = cv2.resize(image, display_size, interpolation=interpolation)
    _, png_data = cv2.imencode(".png", display_image)
    png_bytes = png_data.tobytes()
    photo = PhotoImage(data=png_bytes)
    label.configure(image=photo, text="")
    label.image = photo # tkinter needs a python reference to keep the image alive.


def create_image_box(parent, title, row, column):
    panel = Frame(parent, bd=1, relief=SOLID)
    panel.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
    panel.grid_propagate(False) # image dimensions must not change the grid sizes.
    panel.grid_columnconfigure(0, weight=1)
    panel.grid_rowconfigure(1, weight=1)
    Label(panel, text=title).grid(row=0, column=0, sticky="ew", pady=4)
    label = Label(panel, text="", bg="#eeeeee",
                  width=1, height=1)
    label.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
    label.source_image = None
    label.image = None

    def redraw(event):
        if label.source_image is not None:
            show_image(label, label.source_image)

    label.bind("<Configure>", redraw)
    return label


preview_box = create_image_box(befoglalo, "Camera / capture", 0, 0)
annotation_box = create_image_box(befoglalo, "Annotated image", 0, 1)
alignment_box = create_image_box(befoglalo, "Aligned image", 1, 0)
crop_box = create_image_box(befoglalo, "Face crop", 1, 1)
preview_box.configure(text="Starting camera")


def clear_processing_boxes():
    for label in (annotation_box, alignment_box, crop_box):
        label.configure(image="", text="")
        label.source_image = None
        label.image = None

camera = open_cam()
current_frame = None
preview_active = True
processing_active = False
processing_results = Queue()
model = None
embedding_session = None
detector_lock = Lock()
scan_mode = False


def detect_image(image, quiet=False):
    # both views reuse one detector without running it from two threads at once
    global model
    from main import DETECTOR_PATH, YOLO

    with detector_lock:
        if model is None:
            model = YOLO(str(DETECTOR_PATH))
        if quiet:
            results = model(image, verbose=False)
        else:
            results = model(image)
        return results[0]


def get_embedder():
    # share the existing lazy-loaded Buffalo L session between both views
    global embedding_session
    from main import RECOGNIZER_PATH
    from preprocessing import load_embedder

    if embedding_session is None:
        embedding_session = load_embedder(RECOGNIZER_PATH)
    return embedding_session


def process_capture(image_path, snapshot):
    # only this worker uses the models. all widget updates stay in tkinter.
    try:
        # keep detection on the saved jpeg, but save it off the ui thread
        try:
            saved = cv2.imwrite(str(image_path), snapshot)
        except (cv2.error, OSError):
            saved = False
        if not saved:
            processing_results.put(("status", embedding_label, "Could not save the picture"))
            return

        from main import draw_BB, draw_landmarks, eredmeny_print, process_aligned_faces

        output_dir = input_dir.parent / "feldolgozott"
        aligned_dir = input_dir.parent / "aligned"
        cropped_dir = input_dir.parent / "cropped"
        for directory in (output_dir, aligned_dir, cropped_dir):
            directory.mkdir(parents=True, exist_ok=True)

        result = detect_image(str(image_path))
        annotated = result.orig_img.copy()
        for i, box in enumerate(result.boxes):
            eredmeny_print(box, i)
            draw_BB(box, annotated)
        if result.keypoints is not None:
            draw_landmarks(result.keypoints, annotated)
        annotated_path = output_dir / f"{image_path.stem}_processed{image_path.suffix}"
        cv2.imwrite(str(annotated_path), annotated)
        processing_results.put(("image", annotation_box, annotated))

        face_count = len(result.boxes)
        if face_count != 1 or result.keypoints is None:
            processing_results.put(("status", embedding_label, "Embedding unavailable · capture one face"))
            return

        session = None
        try:
            session = get_embedder()
        except Exception as error:
            # keep the inspection images available if the recognizer cannot load
            print(f"Embedding unavailable: {error}")

        images = process_aligned_faces(result, session, aligned_dir, cropped_dir)
        if images:
            aligned, cropped, embedding = images[0]
            processing_results.put(("image", alignment_box, aligned))
            if cropped is not None:
                processing_results.put(("image", crop_box, cropped))
            # show three real values from this capture without storing the webcam vector
            if embedding is None:
                message = "Embedding unavailable"
            else:
                first = embedding[0]
                second = embedding[1]
                third = embedding[2]
                message = f"Embedding created · {embedding.size} values: {first:.4f}, {second:.4f}, {third:.4f}"
            processing_results.put(("status", embedding_label, message))
        else:
            processing_results.put(("status", embedding_label, "Embedding unavailable"))
    except Exception as error:
        # keep completed panels and report the failed processing step in the terminal.
        print(f"Capture processing failed: {error}")
        processing_results.put(("status", embedding_label, "Embedding unavailable"))
    finally:
        processing_results.put(("done", None, None))


def poll_processing_results():
    global processing_active
    while True:
        try:
            kind, label, value = processing_results.get_nowait()
        except Empty:
            break
        if kind == "image":
            show_image(label, value)
        elif kind == "status":
            label.configure(text=value)
        elif kind == "done":
            processing_active = False
            retake_button.configure(state=NORMAL)
    scan_view.poll_results()
    can_switch = not processing_active and not scan_view.busy
    mode_button.configure(state=NORMAL if can_switch else DISABLED)
    window.after(50, poll_processing_results)


def update_preview():
    #keeps the live image updated
    global current_frame #so that the variable is accessible outside of the function

    if scan_mode:
        current_frame = read_frame(camera)
        scan_view.show_frame(current_frame)
    elif preview_active:
        current_frame = read_frame(camera) #fnc from cam_input.py
        show_image(preview_box, current_frame)

    # keep exactly one callback chain, including while the preview is frozen.
    window.after(30, update_preview) #calls fnc after 30ms


def foto():
    #saves the latest image when button pressed
    global preview_active, processing_active
    if current_frame is None or not preview_active or processing_active:
        return

    snapshot = current_frame.copy()
    capture_time = dt.now()
    filename = capture_time.strftime("capture_%Y%m%d_%H%M%S_%f.jpg")
    image_path = input_dir / filename

    preview_active = False
    show_image(preview_box, snapshot)
    clear_processing_boxes()
    embedding_label.configure(text="Processing capture...")
    button1.configure(state=DISABLED)
    retake_button.configure(state=DISABLED)
    processing_active = True
    worker = Thread(target=process_capture, args=(image_path, snapshot), daemon=True)
    worker.start()


def retake():
    global preview_active
    if processing_active:
        return
    clear_processing_boxes()
    embedding_label.configure(text="No embedding yet")
    preview_active = True
    button1.configure(state=NORMAL)
    retake_button.configure(state=DISABLED)


def switch_mode():
    global scan_mode
    if processing_active or scan_view.busy:
        return
    if scan_mode:
        scan_view.set_visible(False)
        befoglalo.pack(fill=BOTH, expand=True)
        mode_button.configure(text="Switch to Scan")
    else:
        befoglalo.pack_forget()
        scan_view.set_visible(True)
        mode_button.configure(text="Switch to Showcase")
    scan_mode = not scan_mode


button_row = Frame(befoglalo)
button_row.grid(row=2, column=0, columnspan=2, pady=(2, 8))
button1 = Button(button_row, text="Capture", background="purple", fg="white",
                 command=foto)
button1.pack(side=LEFT, padx=4)
retake_button = Button(button_row, text="Retake", command=retake, state=DISABLED)
retake_button.pack(side=LEFT, padx=4)

# keep the embedding status separate from the image panels
embedding_label = Label(befoglalo, text="No embedding yet")
embedding_label.grid(row=3, column=0, columnspan=2, pady=(0, 8))

scan_view = ScanView(window, detect_image, get_embedder, show_image)
mode_button = Button(mode_bar, text="Switch to Scan", command=switch_mode)
mode_button.pack(side=RIGHT, padx=8, pady=4)

try:
    update_preview()
    poll_processing_results()
    window.mainloop()
finally:
    close_cam(camera)
