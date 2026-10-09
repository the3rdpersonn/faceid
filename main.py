from ultralytics import YOLO
from pathlib import Path
import cv2
# globbing utility.
import glob

#file folder paths
input_path = Path("kepek/nyers/test1")
output_dir = Path("kepek/feldolgozott/test1")

# select the path
#path = "kepek/nyers/test1/*.jpg"
images = [cv2.imread(str(path)) for path in input_path.glob("*.jpg")] #line comprehension

# Load the pretrained face-detection model
model = YOLO("weights/yolo11n.pt")

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
        (255, 255, 255),
        2
    )
    label = f"Face {confidence:.4f}"
    cv2.putText(
        images,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

#------------------------------------


results = model(str(input_path))

# Run inference on images
#images = cv2.imread(str(input_path))
for i, result in enumerate(results):
    image =  images[i]
    for j, box in enumerate(result.boxes):
        #image = cv2.imread(file)
        eredmeny_print(box, j)
        draw_BB(box, image)
        cv2.imshow(input_path.name, image)
    
    #output
    output_fName = f"{input_path.stem}_processed{input_path.suffix}"
    output_path = output_dir / output_fName
    cv2.imwrite(str(output_path), image)



    cv2.waitKey(0)# Wait until a key is pressed
    cv2.destroyAllWindows() # Close the window