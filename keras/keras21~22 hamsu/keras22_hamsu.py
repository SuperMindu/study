import numpy as np


#1. 데이터 (행무시, 열우선)
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
              [9,8,7,6,5,4,3,2,1,0]]               
            )
y = np.array([11,12,13,14,15,16,17,18,19,20])
print(x.shape, y.shape) # (3, 10)  (10,) 

x = x.T
print(x.shape) # (10, 3)

#2. 모델 구성
from tensorflow.python.keras.models import Sequential, Model # 함수형 모델
from tensorflow.python.keras.layers import Dense, Input

# model = Sequential() # 시퀀셜 모델은 먼저 정의를 해주고 시작
# # model.add(Dense(10, input_dim=3))
# model.add(Dense(10, input_shape=(3,))) # input_shape=(컬럼의 갯수, ) 행을 제외한 나머지 컬럼들이 다 들어감
# model.add(Dense(5, activation='relu'))
# model.add(Dense(3, activation='sigmoid'))
# model.add(Dense(1))

input1 = Input(shape=(3,)) # input을 명시하고 shape 찍어주고 input1
dense1 = Dense(10)(input1) 
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(3, activation='sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1) # 함수형 모델은 마지막에 정의 해줌
model.summary()
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 3)]               0
# _________________________________________________________________
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 55
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 18
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 117
# Trainable params: 117
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10, batch_size=1)









                                