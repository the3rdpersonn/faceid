from tkinter import *
from pathlib import Path
from datetime import datetime as dt
from queue import Queue, Empty
from threading import Thread

import cv2
from cam_input import *

input_dir = Path(__file__).resolve().parent / "kepek/landmarking test2/nyers"
# Path type object. resolve() ->absolute path - the complete location of the script
# parent: THE CODE + "kepek/test2/nyers" joins folder with the input folder
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
    # Keep the original pixels; resizing here is for display only.
    label.source_image = image
    width = label.winfo_width() - 8
    height = label.winfo_height() - 8
    if width <= 0 or height <= 0:
        return # The Configure event will draw once the panel has a size.

    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    display_size = (max(1, int(image_width * scale)),
                    max(1, int(image_height * scale)))
    display_image = cv2.resize(image, display_size, interpolation=(
        cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR))
    success, pngData = cv2.imencode(".png", display_image)
    if not success:
        raise RuntimeError("Could not prepare the display image.")
    photo = PhotoImage(data=pngData.tobytes())
    label.configure(image=photo, text="")
    label.image = photo # Tkinter needs a Python reference to keep the image alive.


def create_image_box(parent, title, row, column):
    panel = Frame(parent, bd=1, relief=SOLID)
    panel.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
    panel.grid_propagate(False) # Image dimensions must not change the grid sizes.
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
model2 = None


def process_capture(image_path):
    # Only this worker uses the models. All widget updates stay in Tkinter.
    global model, model2
    try:
        from main import YOLO, draw_BB, draw_landmarks, eredmeny_print, process_aligned_faces

        if model is None:
            weights = Path(__file__).resolve().parent / "weights/yolo11s-pose_widerface.pt"
            model = YOLO(str(weights))
            model2 = YOLO(str(weights))

        output_dir = input_dir.parent / "feldolgozott"
        aligned_dir = input_dir.parent / "aligned"
        cropped_dir = input_dir.parent / "cropped"
        for directory in (output_dir, aligned_dir, cropped_dir):
            directory.mkdir(parents=True, exist_ok=True)

        result = model(str(image_path))[0]
        annotated = result.orig_img.copy()
        for i, box in enumerate(result.boxes):
            eredmeny_print(box, i)
            draw_BB(box, annotated)
        if result.keypoints is not None:
            draw_landmarks(result.keypoints, annotated)
        annotated_path = output_dir / f"{image_path.stem}_processed{image_path.suffix}"
        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError(f"Could not save annotated image: {annotated_path}")
        processing_results.put(("image", annotation_box, annotated))

        face_count = len(result.boxes)
        if face_count != 1:
            print(f"Found {face_count} faces; alignment and crop require exactly one face.")
            return
        if result.keypoints is None:
            print("No facial landmarks returned; skipping alignment and crop.")
            return

        processing_results.put(("status", alignment_box, "Aligning and cropping..."))
        images = process_aligned_faces(result, model2, aligned_dir, cropped_dir, show=False)
        if images:
            aligned, cropped = images[0]
            processing_results.put(("image", alignment_box, aligned))
            if cropped is not None:
                processing_results.put(("image", crop_box, cropped))
    except Exception as error:
        # Keep completed panels and report the failed processing step in the terminal.
        print(f"Capture processing failed: {error}")
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

    # Keep exactly one callback chain, including while the preview is frozen.
    window.after(30, update_preview) #calls fnc after 30ms


def foto():
    #saves the latest image when button pressed
    global preview_active, processing_active
    if current_frame is None or not preview_active or processing_active:
        return

    snapshot = current_frame.copy()
    fName = dt.now().strftime("capture_%Y%m%d_%H%M%S_%f.jpg") #filename based on time of fotozas

    try:
        if not cv2.imwrite(str(input_dir / fName), snapshot):
            print("Could not save the picture.")
            return
    except (cv2.error, OSError) as error:
        print(f"Could not save the picture: {error}")
        return

    preview_active = False
    show_image(preview_box, snapshot)
    clear_processing_boxes()
    button1.configure(state=DISABLED)
    retake_button.configure(state=DISABLED)
    processing_active = True
    annotation_box.configure(text="Detecting...")
    Thread(target=process_capture, args=(input_dir / fName,), daemon=True).start()


def retake():
    global preview_active
    if processing_active:
        return
    clear_processing_boxes()
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

try:
    update_preview()
    poll_processing_results()
    window.mainloop()
finally:
    close_cam(camera)
