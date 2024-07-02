from ultralytics import YOLO
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt
import cv2

#to take input of files 
input_image_path = input("Enter Image path :")

# Load a pretrained YOLOv8n model 
model = YOLO('./runs/detect/train/weights/best.pt')

# path to save output of kidney stone detection
output_path = "./test-output/"


# Run inference on 'test images' with
results = model.predict(input_image_path, save=True , conf=0.5,save_conf=True,show_boxes=True,show_labels=True,show=True,save_txt=True)

predicted_direcotry_path = results[0].save_dir

shutil.copy(os.path.join(predicted_direcotry_path,os.path.basename(input_image_path)),output_path)

output = cv2.imread(os.path.join(output_path,os.path.basename(input_image_path)))
output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
plt.imshow(output)
plt.axis('off')
plt.show()


