from tkinter import *
from pathlib import Path
from datetime import datetime as dt

import cv2
from cam_input import *

input_dir = Path(__file__).resolve().parent / "kepek/test2/nyers"
# Path type object. resolve() ->absolute path - the complete location of the script
# parent: THE CODE + "kepek/test2/nyers" joins folder with the input folder
# advantage: this stays relative to the scripts location, regardless where i run it from.
input_dir.mkdir(parents=True, exist_ok=True) #in case thee folder path is incomplete, it creates missing folders


window = Tk() #initiates the window

befoglalo = Frame(window)
befoglalo.pack(fill=BOTH, expand = True)

preview_box = Label(befoglalo) #image holder "box"
preview_box.pack(side = TOP, expand = True, fill= BOTH) #where do we place the box?
#the label expands but the frames dont, because we would need to resize the image every time.

camera = open_cam()
current_frame = None


def update_preview():
    #keeps the live image updated
    global current_frame #so that the variable is accessible outside of the function

    current_frame = read_frame(camera) #fnc from cam_input.py

    success, pngData = cv2.imencode(".png", current_frame) #converts image to png but doesnt save
    if not success:
        raise RuntimeError("Could not prepare the preview image.")
        #if conversion not successful

    photo = PhotoImage(data=pngData.tobytes()) #conversion to bytes to create image Tkinter can display
    preview_box.configure(image=photo) #Label1 displays the image
    preview_box.image = photo #reference??

    window.after(30, update_preview) #calls fnc after 30ms


def foto():
    #saves the latest image when button pressed
    if current_frame is None:
        return

    snapshot = current_frame.copy()
    fName = dt.now().strftime("capture_%Y%m%d_%H%M%S_%f.jpg") #filename based on time of fotozas

    if not cv2.imwrite(str(input_dir / fName), snapshot): 
        raise RuntimeError("Could not save the picture.")


button1 = Button(befoglalo, text="Capture", backgroun = "purple", fg = "white", command=foto)
button1.pack(side = BOTTOM, expand = True, fill= BOTH)

try:
    update_preview()
    window.mainloop()
finally:
    close_cam(camera)