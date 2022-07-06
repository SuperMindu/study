from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_train)) # 
# print(np.max(x_train)) # 
# print(np.min(x_test)) # 
# print(np.max(x_test))

#2. 모델 구성 
model = Sequential()
model.add(Dense(32, input_dim=30))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=10, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')

y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 : ', r2)
y_predict = y_predict.round(0) # softmax를 통과해서 실수로 나오는 y_predict 값을 반올림 해서 0과 1로 맞춰줌 (acc 비교를 위해)

acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)

#        노말                                  MinMax                                   Standard                                    MaxAbs                               Robust                
# loss :  0.2723550498485565            loss :  0.08259253203868866              loss :  0.12399382144212723                                                                                       
# acc 스코어 :  0.9239766081871345       acc 스코어 :  0.9707602339181286         acc 스코어 :  0.9649122807017544                                                                                                    
#                                                                                                                   