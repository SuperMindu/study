import numpy as np
import cv2
import sys
# https://velog.io/@secdoc/%EB%B2%88%EC%97%AD-YOLOv3-%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EA%B8%B0-1%ED%83%84

# 1. Yolo 로드
net = cv2.dnn.readNet("D:/yolov3.weights", "D:/yolov3.cfg")
classes = []
with open("D:/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 2. 이미지 가져오기
img = cv2.imread("D:/PP/12.jpg")
img = cv2.resize(img, None, fx=1, fy=1)
height, width, channels = img.shape

# 네트워크에서 이미지를 바로 사용할 수 없기때문에 먼저 이미지를 Blob으로 변환해야 함
# Blob은 이미지에서 특징을 잡아내고 크기를 조정하는데 사용됨2

# 3. Detecting objects
# outs는 감지 결과이다. 탐지된 개체에 대한 모든 정보와 위치를 제공
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)


# 4. 결과 화면에 표시 / 신뢰도, 신뢰 임계값  계산
# 정보를 화면에 표시
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
# 신뢰도가 0.5 이상이라면 물체가 정확히 감지되었다고 간주한다. 아니라면 넘어감..
# 임계값은 0에서 1사이의 값을 가지는데 1에 가까울수록 탐지 정확도가 높고 , 0에 가까울수록 정확도는 낮아지지만 탐지되는 물체의 수는 많아짐

# 노이즈 제거 : 
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# 같은 물체에 대한 박스가 많은것을 제거
# Non maximum suppresion이라고 한답니다.
 

# 마지막으로 모든 정보를 추출하여 화면에 표시합니다.
# Box : 감지된 개체를 둘러싼 사각형의 좌표
# Label : 감지된 물체의 이름
# Confidence : 0에서 1까지의 탐지에 대한 신뢰도
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


