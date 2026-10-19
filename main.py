from ultralytics import YOLO
from pathlib import Path
import cv2
# globbing utility.
import glob
import shutil #for clearing output folder before writing new annotated images



#file folder paths
input_path = Path("kepek/landmarking test2/nyers")
output_dir = Path("kepek/landmarking test2/feldolgozott")

for item in output_dir.iterdir(): # wipe output folder (contains annotated images)
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()

# select the path
#path = "kepek/nyers/test1/*.jpg"
# Each result already contains its matching original image (result.orig_img).

# Load the pretrained face-detection model
model = YOLO("weights/yolo11s-pose_widerface.pt")

#---------fncs-----------------------
def eredmeny_print(box, i):
    coordinates = box.xyxy[0]
    confidence = box.conf[0]
    class_id = box.cls[0]

    print(f"Face {i + 1}")
    print(f"  Coordinates: {coordinates}")
    print(f"  Confidence: {confidence:.5f}")
    print(f"  Class ID: {int(class_id)}")

def draw_BB(box, images):
    #extract BB coordinates
    x1, y1, x2, y2 = box.xyxy[0]
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    confidence = float(box.conf[0]) #detection confidence stored for display
    #BB drawing:
    cv2.rectangle(
        images,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        2
    )
    label = f"Face {100*confidence:.2f}%"
    cv2.putText(
        images,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2
    )

def draw_rBB(box, images): #resized bounding boxes
    #extract BB coordinates
    x1, y1, x2, y2 = box.xyxy[0]
    x1 = int(x1/3)
    y1 = int(y1/3)
    x2 = int(x2/3)
    y2 = int(y2/3)
    confidence = float(box.conf[0]) #detection confidence stored for display
    #BB drawing:
    cv2.rectangle(
        images,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        2
    )
    label = f"Face {100*confidence:.2f}%"
    cv2.putText(
        images,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2
    )
def draw_landmarks(keypoints, image, scale=1.0, min_confidence=0.5):
    # Five points per face. Numbering preserves the model's landmark order.
    # Left/right refer to the picture as viewed, not the person's anatomy.
    landmark_names = ("left eye", "right eye", "nose tip",
                      "left mouth corner", "right mouth corner")
    # Coordinates are in original-image pixels; scale only changes the drawing.
    for face_i, points in enumerate(keypoints.data.cpu().tolist()):
        print(f"  Landmarks for face {face_i + 1} (original-image pixels):")
        for point_i, point in enumerate(points):
            x, y = point[:2]
            confidence = point[2] if len(point) == 3 else None
            valid = (x > 0 and y > 0 and
                     (confidence is None or confidence >= min_confidence))
            confidence_text = "n/a" if confidence is None else f"{confidence:.3f}"
            print(f"    {landmark_names[point_i]}: x={x:.2f}, y={y:.2f}, "
                  f"confidence={confidence_text}, drawn={valid}")
            if not valid:
                continue
            position = (round(x * scale), round(y * scale))
            cv2.circle(image, position, 3, (0, 0, 255), -1)
            cv2.putText(image, str(point_i),
                        (position[0] + 4, position[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

#------------------------------------


results = model(str(input_path))

max_width = 1080
max_height = 720 # processed images should not exceed a certain limit in display size


for i, result in enumerate(results): # enumerate for indexing
    image = result.orig_img.copy() # guaranteed to match this result's boxes/keypoints
    w = image.shape[1]
    h = image.shape[0] # to access image height and width
    newW = int(image.shape[1]/3)
    newH = int(image.shape[0]/3)
    
    resized_image = cv2.resize(image, (newW, newH))
    if w >= max_width or h >= max_height:
        for j, box in enumerate(result.boxes): #output
                eredmeny_print(box, j)
                draw_rBB(box, resized_image)
        if result.keypoints is not None:
            draw_landmarks(result.keypoints, resized_image, scale=1/3)
        cv2.imshow(input_path.name, resized_image)
    else: 
        for j, box in enumerate(result.boxes): #output
            eredmeny_print(box, j)
            draw_BB(box, image)
        if result.keypoints is not None:
            draw_landmarks(result.keypoints, image)
        cv2.imshow(input_path.name, image)
        

        
        
        
        
    
    #annotated images:
    source_path = Path(result.path)
    output_fName = f"{source_path.stem}_processed{source_path.suffix}"
    output_path = output_dir / output_fName
    if w >= max_width or h >= max_height:
        cv2.imwrite(str(output_path), resized_image)
    else:
        cv2.imwrite(str(output_path), image)




    cv2.waitKey(0)# Wait until a key is pressed
    cv2.destroyAllWindows() # Close the window
