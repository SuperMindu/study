from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))

x_test = np.array([11,12,13]) 
y_test = np.array([11,12,13]) # evaluate, predict 에서 씀

x_val = np.array([14,15,16])
y_val = np.array([14,15,16])

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) # val_loss는 로스 값보다 조금 덜 떨어짐

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) # val은 여기서의 loss 보다는 중요하지 않음
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)




