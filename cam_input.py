import cv2
 
#Module for taking care of the webcam. 
#Functions for configuring default webcam,
#reading and returning current frame,
#closing camera.
#progression for this module: make it so that i no longer need to take a picture, recognition is achieved by continuous live feed 
#(might be out of scope for this project)
 
def open_cam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not access the camera.")

    return cap #camera connection variable

def read_frame(cap):
    #receives connection, and returns a frame, repeated calls result in a live preview feed for taking a picture later
    success, frame = cap.read()

    if not success or frame is None:
        raise RuntimeError("Could not read a camera frame.")

    return frame 
 
# Release the webcam
def close_cam(cap):
    cap.release()



   