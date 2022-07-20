# 넘파이로 저장
# 이렇게 트레인 테스트 폴더가 따로 지정돼 있지 않은 경우는 전체 데이터를 합쳐서 불러와서 트레인과 테스트를 스플릿으로 따로 나눠줘야 함
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

# 1. 데이터
datagen = ImageDataGenerator()

xy = datagen.flow_from_directory('d:/study_data/_data/image/horse_human/', # 폴더에서 가져와서 ImageDataGenerator
                                target_size=(150, 150), # 이미지 크기 조절. 고르지 않은 크기들을 사이즈를 지정해줌. 내 맘대로 가능
                                batch_size=100000000,
                                class_mode='binary', # 0, 1 분류. 이진분류. (3가지 이상은 categorical)
                                #  color_mode='grayscale', # 이걸 따로 지정해주지 않으면 디폴트값은 컬러로 나옴. 밑에 print(xy_train[31][0].shape) 참고
                                shuffle=True,
                                ) 



x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8, shuffle=True)

print(xy_train[0][0].shape, xy_train[0][1].shape) # (1027, 150, 150, 3) (1027, 2)
print(xy_test[0][0].shape, xy_test[0][1].shape) # (1027, 150, 150, 3) (1027, 2)

np.save('d:/study_data/_save/_npy/keras47_2_horse_human_train_x.npy', arr=xy_train[0][0]) 
np.save('d:/study_data/_save/_npy/keras47_2_horse_human_train_y.npy', arr=xy_train[0][1]) 
np.save('d:/study_data/_save/_npy/keras47_2_horse_human_test_x.npy', arr=xy_test[0][0]) 
np.save('d:/study_data/_save/_npy/keras47_2_horse_human_test_y.npy', arr=xy_test[0][1]) 

