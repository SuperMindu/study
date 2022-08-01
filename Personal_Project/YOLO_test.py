# https://bong-sik.tistory.com/16
# https://www.youtube.com/watch?v=fdWx3QV5n44&t=1300s
# https://panggu15.github.io/detection/%EA%B0%84%EB%8B%A8%ED%95%9C-YOLO-%EA%B5%AC%ED%98%84(OpenCV)/

# https://diyver.tistory.com/169 <- opencv 설치

# 아래와 같은 순서로 설치
# >>> pip install opencv-python
# >>> pip install opencv-contrib-python

# 잘 설치됐나 프린트 찍어보고
# import cv2
# print(cv2.__version__)

# 이미지 불러와보기
import cv2
import sys

# print(cv2.__version__)

# image = cv2.imread("1.jpg",cv2.IMREAD_COLOR)
# cv2.imshow("1", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.imread('D:/PP/7.jfif')
if img is None:
    print('Image load failed')
    sys.exit()
cv2.imshow('test', img)
cv2.waitKey()

cv2.destroyAllWindows()
# 이미지 이름에 한글이 있으면 안되는 듯??
