import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import time


from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x)
# print(y)
# print(x.shape, y.shape)   # (506, 13) (506,)  
# print(datasets.feature_names)   #sklearn 에서만 가능
# print(datasets.DESCR)


#2.  모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()


# model.save_weights('./_save/keras23_5_save_weights1.h5')             # (5) <- 훈련 시키기 전 랜덤 weight


# model.load_weights('./_save/keras23_5_save_weights1.h5')             # (6) <- RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`. 

model = load_model('./_save/keras23_3_save_model.h5')                  # 
model.load_weights('./_save/keras23_5_save_weights2.h5')               # (7) <- 얘는 훈련한 다음의 가중치가 저장 돼 있어서 loss와 r2가 동일하게 나옴 (3단계에서 컴파일만 살리면 됨)


# 3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es], verbose=1) 

# model.save_weights('./_save/keras23_5_save_weights2.h5')             # <- weight 저장됨



# model.save는 모델과 가중치 저장. save_weights 는 가중치'만' 저장됨
# 그래서 load_weights만 하면 모델은 불러와지지 않음

# 저장된 weights를 불러올 때는 모델구성, compile을 해주면 됨(fit 생략)
# 문제점 : 최적의 weight인가? 따라서 epoch당 가중치 저장(체크포인트 형식으로 저장)


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  10.710644721984863
# r2 스코어 :  0.8718560785416583



