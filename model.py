from ultralytics import YOLO

model = YOLO("yolov8n.yaml") 


results = model.train(data="./datasets/data.yaml", epochs=50)

# $ yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml --> validate custom model 
# yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.5 source={dataset.location}/test/images save_txt=true save_conf=true





    



