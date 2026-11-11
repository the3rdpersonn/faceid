import cv2
 
# open the default webcam, read frames, and release the camera
 
def open_cam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not access the camera.")

    return cap #camera connection variable

def read_frame(cap):
    # each call returns one frame of the live preview
    success, frame = cap.read()

    if not success or frame is None:
        raise RuntimeError("Could not read a camera frame.")

    return frame
 
# release the webcam
def close_cam(cap):
    cap.release()



   