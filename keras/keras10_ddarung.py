# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd   # 엑셀 데이터 불러올 때 사용
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = './_data/ddarung/'  # .은 현재폴더라는 뜻
train_set = pd.read_csv(path + 'train.csv',  # train.csv의 데이터들이 train_set에 수치화 돼서 들어간다 
                        index_col=0
                        )  # index_col=n. n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape)  # (1459, 10)


test_set = pd.read_csv(path + 'test.csv',  #예측에서 씀
                       index_col=0)
print(test_set)
print(test_set.shape)  # (715, 9)

print(train_set.columns)
print(train_set.info())  # 결측치 = 이빨 빠진 데이터
print(train_set.describe())  # 

#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum()) #null의 합계를 구함
train_set = train_set.dropna()
print(train_set.isnull().sum()) 
print(train_set.shape)  # (1328, 10)


x = train_set.drop(['count'], axis=1) 
print(x)
print(x.columns)
print(x.shape)  # (1459, 9)

y = train_set['count']
print(y)
print(y.shape)  # (1459,) 

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.80, shuffle=True, random_state=300)


#2. 모델 구성
model=Sequential()
model.add(Dense(30, input_dim=9))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(30))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=800, batch_size=50)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict): #괄호 안의 변수를 받아들인다 :다음부터 적용
    return np.sqrt(mean_squared_error(y_test, y_predict)) #루트를 씌워서 돌려줌 

rmse = RMSE(y_test, y_predict)  #y_test와 y_predict를 비교해서 rmse로 출력 (원래 데이터와 예측 데이터를 비교)
print("RMSE : ", rmse)


# loss :  2405.498046875
# RMSE :  49.045877295397666


#과제1. 함수에 대해서 공부하기 
#과제2. 낮은 로스값 스샷


'''
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.80, shuffle=True, random_state=72)
        '''

