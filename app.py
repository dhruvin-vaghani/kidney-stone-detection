import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

st.title("YOLO Kidney Stone Detection!")
st.write("Upload a Kidney Image to detect Stones")

uploaded_file = st.file_uploader("Upload a Image", type="jpg")

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting stone in Kidney Image...")

    # Perform object detection
    results = model.predict(source=image)
    
    # Check if predictions were made
    if len(results) > 0 and hasattr(results[0], 'boxes'):
        # Draw bounding boxes on the image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id] if class_id < len(model.names) else "Unknown"
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
        # Display the image with bounding boxes
        st.image(image, caption='Stone detected Image', use_column_width=True)
        st.write("Detection Complete")
    else:
        st.write("No Stone Detected")
