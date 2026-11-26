# opencv provides the camera connection and frame reading methods
import cv2
 
# open the default webcam, read frames, and release the camera
 
# open one connection that can be reused throughout the preview
def open_cam():
    # camera index zero selects the default webcam
    cap = cv2.VideoCapture(0)
    # check whether the operating system allowed the camera connection
    if not cap.isOpened():
        # release the attempted connection before reporting that opening failed
        cap.release()
        raise RuntimeError("Could not access the camera.")

    # return the camera object so other functions can read from the same connection
    return cap #camera connection variable

# the caller passes in the camera object created by open_cam
def read_frame(cap):
    # each call returns one frame of the live preview
    # read returns a success flag together with the captured image pixels
    success, frame = cap.read()

    # an unsuccessful read has no usable image to pass to the display
    if not success or frame is None:
        raise RuntimeError("Could not read a camera frame.")

    return frame
 
# release the webcam
def close_cam(cap):
    # releasing the camera makes it available to other applications after closing
    cap.release()



   