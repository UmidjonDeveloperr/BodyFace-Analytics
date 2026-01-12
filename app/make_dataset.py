import os
from ultralytics import YOLO

images_folder = "/home/umidjon/Documents/CrowdHumanLast.v1i.yolov11/valid/images"
labels_folder = "/home/umidjon/Documents/CrowdHumanLast.v1i.yolov11/valid/labels"
model_path = "/home/umidjon/YOLOv11n-face-detection/model.pt"
face_class_id = 2

os.makedirs(labels_folder, exist_ok=True)

model = YOLO(model_path)

for img_name in os.listdir(images_folder):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    
    img_path = os.path.join(images_folder, img_name)
    
    results = model.predict(img_path, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    img_width, img_height = results[0].orig_shape[1], results[0].orig_shape[0]

    face_lines = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        face_lines.append(f"{face_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    label_file = os.path.join(labels_folder, os.path.splitext(img_name)[0] + ".txt")
    existing_lines = []
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            existing_lines = [line.strip() for line in f if line.strip()]
    
    all_lines = existing_lines + face_lines
    
    with open(label_file, "w") as f:
        f.write("\n".join(all_lines))

    print(f"Processed {img_name}, {len(face_lines)} faces added, total {len(all_lines)} boxes.")
