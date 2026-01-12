from ultralytics import YOLO
from deepface import DeepFace
import cv2

person_model = YOLO("/home/umidjon/Documents/runs/detect/train2/weight/best.pt")

source_video = "/home/umidjon/cv_projects/learnyolo/app/videos/7971061-uhd_3840_2160_30fps.mp4"

cap = cv2.VideoCapture(source_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("PersonsVideo.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, heigth))

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = person_model(frame, stream=False)
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    for box, conf, cls in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)

        if int(cls) == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
            cv2.putText(frame, f"Body: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, (255, 0, 0), 4)
        if int(cls) == 1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, f"Head: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, (0, 0, 255), 4)
        if int(cls) == 2:
            face_img = frame[y1:y2, x1:x2]
            face_img = cv2.resize(face_img, (224, 224))
            face_predictions = DeepFace.analyze(face_img, actions=["age", "gender"], enforce_detection=False)
            age = face_predictions[0]['age']
            gender = face_predictions[0]['gender']
            gender_label = max(gender, key=gender.get)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"Face: {conf:.2f}", (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, (255, 255, 255), 4)
            cv2.putText(frame, f"Age: {age}", (x1 + 10, y1 + 85), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, (255, 255, 255), 4)
            cv2.putText(frame, f"Gender: {gender_label}", (x1 + 10, y1 + 125), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, (255, 255, 255), 4)


    cv2.imshow("frame", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()