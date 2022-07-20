import numpy as np
from keras.preprocessing.image import ImageDataGenerator # 

train_datagen = ImageDataGenerator( # 아래의 모든건 다 랜덤으로 적용됨, 증폭가능
    rescale=1./255, # MinMax 해주겠다는 것과 같음 (스케일링 기능임)
    horizontal_flip=True, # 수평 반전
    vertical_flip=True, # 수직 반전
    width_shift_range=0.1, # 가로 수평 이동 0.1 (10%)
    height_shift_range=0.1, # 상하 이동 
    rotation_range=5, # 돌려돌려돌림판
    zoom_range=1.2, # 확대 
    shear_range=0.7, # 기울기? 찌그러트리기? 비틀기? 짜부? 
    fill_mode='nearest'
) # 트레인 데이터 준비만 한거임. 선언만 한거임

test_datagen = ImageDataGenerator(
    rescale=1./255 # 테스트 데이터는 증폭하면 안됨
)

xy_train = train_datagen.flow_from_directory('d:/_data/image/brain/train/', # 폴더에서 가져와서 ImageDataGenerator
                                             target_size=(200, 200), # 이미지 크기 조절. 고르지 않은 크기들을 사이즈를 지정해줌. 내 맘대로 가능
                                             batch_size=5, # 
                                             class_mode='binary', # 0, 1 분류. 이진분류. (3가지 이상은 categorical)
                                             color_mode='grayscale', # 이걸 따로 지정해주지 않으면 디폴트값은 컬러로 나옴. 밑에 print(xy_train[31][0].shape) 참고
                                             shuffle=True,
                                             ) 
# Found 160 images belonging to 2 classes. flow_from_directory를 통과했을 때 160개의 이미지와 2개의 클래스가 됐음

xy_test = test_datagen.flow_from_directory('d:/_data/image/brain/test/', 
                                             target_size=(200, 200), 
                                             batch_size=5,
                                             class_mode='binary', 
                                             shuffle=True,
                                             ) 
# Found 120 images belonging to 2 classes.

# print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001D1D70A8D90> 
# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets) <- 데이터 자체가 나오진 않음. 설명만 주저리

print(xy_train[26]) # <- x_train에 y값이 포함 돼있다. (배치사이즈 5개니까 160/5=32개) 마지막 배치는 31개
# 근데 지금과 같이 이미지 개수가 160개일때 배치를 6으로 하는 경우(26.66666...) 나눠 떨어지지 않는 경우는 마지막 값을 찍어보면 그 나눈 나머지가 나옴 
# 이미지 데이터의 전체 사이즈보다 배치를 더 많이 넣어도 컴터가 알아서 해줌 
 
# print(xy_train[31][0]) # 0하면 x값만 나옴. 1이 y. 
# print(xy_train[31][1]) 
# print(xy_train[30]1][.shape) # (5, 150, 150, 3) 따로 컬러를 지정해주지 않으면 흑백도 컬러로 인식함. 따라서 (n, n, n, 3)
# print(xy_train[31][0].shape, xy_train[31][1].shape)

print(type(xy_train)) # 자료형을 보면 <class 'keras.preprocessing.image.DirectoryIterator'>  Iterator는 반복자라는 뜻. for문도 Iterator의 일종임
print(type(xy_train[0])) # <class 'tuple'> 수정불가
print(type(xy_train[0][0])) # x <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # y <class 'numpy.ndarray'>
# x 넘퍼이 y 넘파이가 배치 단위로 묶여있는 구조








