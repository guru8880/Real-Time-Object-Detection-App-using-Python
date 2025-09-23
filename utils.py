import cv2
import numpy as np
from PIL import Image

def opencv_to_pil(frame):
    # OpenCV BGR -> PIL RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

def pil_to_opencv(pil_image):
    # PIL RGB -> OpenCV BGR
    opencv_image = np.array(pil_image)
    return opencv_image[:, :, ::-1].copy()

def draw_boxes(frame, results, conf_threshold=0.3):
    """
    frame: BGR image (numpy array)
    results: YOLO results object
    returns: frame with boxes drawn (BGR)
    """
    img = frame.copy()
    
    # For Ultralytics YOLOv8
    if hasattr(results, 'boxes') and results.boxes is not None:
        boxes = results.boxes
        for box in boxes:
            conf = box.conf.item()
            if conf < conf_threshold:
                continue
                
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls.item())
            cls_name = results.names[cls]
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create label
            label = f"{cls_name} {conf:.2f}"
            
            # Get text size
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background for text
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
            
            # Put text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img