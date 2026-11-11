from tkinter import *
from pathlib import Path
from datetime import datetime as dt
from queue import Queue, Empty
from threading import Thread

import cv2
from cam_input import *

script_path = Path(__file__).resolve()
project_dir = script_path.parent
input_dir = project_dir / "kepek/landmarking test2/nyers"
# path type object. resolve() ->absolute path - the complete location of the script
# parent: the code + "kepek/test2/nyers" joins folder with the input folder
# advantage: this stays relative to the scripts location, regardless where i run it from.
input_dir.mkdir(parents=True, exist_ok=True) #in case thee folder path is incomplete, it creates missing folders


window = Tk() #initiates the window
window.title("Face processing")
window.geometry("1000x760")
window.minsize(640, 480)

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


def process_capture(image_path):
    # only this worker uses the models. all widget updates stay in tkinter.
    global model, embedding_session
    try:
        from main import YOLO, draw_BB, draw_landmarks, eredmeny_print, process_aligned_faces
        from preprocessing import load_embedder

        weights = project_dir / "weights"
        if model is None:
            detector_path = weights / "yolo11s-pose_widerface.pt"
            model = YOLO(str(detector_path))

        output_dir = input_dir.parent / "feldolgozott"
        aligned_dir = input_dir.parent / "aligned"
        cropped_dir = input_dir.parent / "cropped"
        for directory in (output_dir, aligned_dir, cropped_dir):
            directory.mkdir(parents=True, exist_ok=True)

        results = model(str(image_path))
        result = results[0]
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

        processing_results.put(("status", alignment_box, "Aligning and cropping..."))
        if embedding_session is None:
            recognizer_path = weights / "w600k_r50.onnx"
            try:
                embedding_session = load_embedder(recognizer_path)
            except Exception as error:
                # keep the inspection images available if the recognizer cannot load
                print(f"Embedding unavailable: {error}")

        images = process_aligned_faces(result, embedding_session, aligned_dir, cropped_dir, show=False)
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
            for panel in (annotation_box, alignment_box, crop_box):
                panel.configure(text="")
            retake_button.configure(state=NORMAL)
    window.after(50, poll_processing_results)


def update_preview():
    #keeps the live image updated
    global current_frame #so that the variable is accessible outside of the function

    if preview_active:
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

    try:
        if not cv2.imwrite(str(image_path), snapshot):
            print("Could not save the picture.")
            return
    except (cv2.error, OSError) as error:
        print(f"Could not save the picture: {error}")
        return

    preview_active = False
    show_image(preview_box, snapshot)
    clear_processing_boxes()
    embedding_label.configure(text="Creating embedding...")
    button1.configure(state=DISABLED)
    retake_button.configure(state=DISABLED)
    processing_active = True
    annotation_box.configure(text="Detecting...")
    worker = Thread(target=process_capture, args=(image_path,), daemon=True)
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

try:
    update_preview()
    poll_processing_results()
    window.mainloop()
finally:
    close_cam(camera)
