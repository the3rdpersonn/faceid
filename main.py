from ultralytics import YOLO
from pathlib import Path
import cv2
# globbing utility.
import glob
import shutil #for clearing output folder before writing new annotated images



#file folder paths
input_path = Path("kepek/nyers/test1")
output_dir = Path("kepek/feldolgozott/test1")

for item in output_dir.iterdir(): # wipe output folder (contains annotated images)
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()

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
        (255, 255, 0),
        2
    )
    label = f"Face {confidence:.2f}"
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
    label = f"Face {confidence:.2f}"
    cv2.putText(
        images,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2
    )
#------------------------------------


results = model(str(input_path))

max_width = 1080
max_height = 720 # processed images should not exceed a certain limit in display size


for i, result in enumerate(results): # enumerate for indexing
    image =  images[i]
    w = image.shape[1]
    h = image.shape[0] # to access image height and width
    newW = int(image.shape[1]/3)
    newH = int(image.shape[0]/3)
    
    resized_image = cv2.resize(image, (newW, newH))
    if w >= max_width or h >= max_height:
        for j, box in enumerate(result.boxes): #output
                eredmeny_print(box, j)
                draw_rBB(box, resized_image)
                cv2.imshow(input_path.name, resized_image)
    else: 
        for j, box in enumerate(result.boxes): #output
            eredmeny_print(box, j)
            draw_BB(box, image)
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