import cv2
from yolo_segmentation import YOLOSEG
import cvzone
ys = YOLOSEG("best_butterfly_seg.pt")

my_file = open("coco_butterfly.txt", "r")
data = my_file.read()
class_list = data.split("\n")

cap=cv2.VideoCapture(0)
count=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(640,480))
    overlay = frame.copy()
    alpha = 0.7

    bboxes, classes, segmentations, scores = ys.detect(frame)
    if scores[0] == 0.0:
        continue
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        (x, y, x2, y2) = bbox
        c= class_list[class_id]
        if 'butterfly' in c:
            if score > 0.5:
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
                cv2.polylines(frame, [seg], True, (0, 0, 255), 4)
                cv2.fillPoly(overlay, [seg], (0,0,255))
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 2, frame)
                cvzone.putTextRect(frame, f'{c , score}', (x,y),1,1)


    cv2.imshow("YOLOv8_seg", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()