# 19일 09시 아모레 시가(20%), 20일 종가(80%) 맞추기
# 거래량 반드시. 7개 이상의 컬럼 쓰기
# 삼전이랑 앙상블 해서 만들기
import numpy as np
import pandas as pd
from sqlalchemy import true 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, Conv1D
from tensorflow.keras.layers import Bidirectional
from keras.layers.recurrent import SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import datetime as dt

###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################

''' 1-1) 데이터 로드 '''
df_amore=pd.read_csv('./_data/test_amore_0718/아모레220718.csv', thousands=',', encoding='cp949') # 아모레 데이터 로드
df_samsung=pd.read_csv('./_data/test_amore_0718/삼성전자220718.csv', thousands=',', encoding='cp949') # 삼성전자 데이터 로드
df_amore.describe()
# print(df_amore.columns)
# Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
#       dtype='object')

''' 1-2) 데이터 정제 '''
# 결측치 확인
# print(df_amore.info()) 
# print(df_samsung.info()) 
df_amore = df_amore.dropna()
df_samsung = df_samsung.dropna()

# 이상치 확인
# q3 = df_amore.quantile(0.75) 
# q1 = df_amore.quantile(0.25)
# iqr = q3 - q1

# # 데이터 사용할 기간 설정
df_amore = df_amore.loc[df_amore['일자']>="2018/05/04"]
df_samsung = df_samsung.loc[df_samsung['일자']>="2018/05/04"]
print(df_amore.shape, df_samsung.shape) # (2990, 17) (2997, 17)


# 필요없는 컬럼 삭제
df_amore = df_amore.drop(['등락률'], axis=1) 
df_amore = df_amore.drop(['전일비'], axis=1) 
df_amore = df_amore.drop(['Unnamed: 6'], axis=1) 
df_samsung = df_samsung.drop(['등락률'], axis=1) 
df_samsung = df_samsung.drop(['Unnamed: 6'], axis=1) 
df_samsung = df_samsung.drop(['전일비'], axis=1) 
# print(df_amore)
# print(df_samsung)

# 년월일 분리 및 요일 추가
df_amore['날짜_datetime'] = pd.to_datetime(df_amore['일자'])
df_amore['년'] = df_amore['날짜_datetime'].dt.year
df_amore['월'] = df_amore['날짜_datetime'].dt.month
df_amore['일'] = df_amore['날짜_datetime'].dt.day
df_amore['요일'] = df_amore['날짜_datetime'].dt.day_name()
df_amore = df_amore.drop(['일자', '날짜_datetime'], axis=1) 

df_samsung['날짜_datetime'] = pd.to_datetime(df_samsung['일자'])
df_samsung['년'] = df_samsung['날짜_datetime'].dt.year
df_samsung['월'] = df_samsung['날짜_datetime'].dt.month
df_samsung['일'] = df_samsung['날짜_datetime'].dt.day
df_samsung['요일'] = df_samsung['날짜_datetime'].dt.day_name()
df_samsung = df_samsung.drop(['일자', '날짜_datetime'], axis=1) 
# print(df_amore)
# print(df_samsung)
print(df_amore.shape) # (3180, 18)
print(df_samsung.shape) # (3040, 18)

# 라벨인코딩
# df_amore, df_samsung=['요일']
# encoder = LabelEncoder()
for col in ['요일']:
    encoder = LabelEncoder()
    df_amore[col] = encoder.fit_transform(df_amore[col])
df_amore.loc[:,['요일']].head()
# print(labels)
# print(df_amore)

for col in ['요일']:
    encoder = LabelEncoder()
    df_samsung[col] = encoder.fit_transform(df_samsung[col])
df_samsung.loc[:,['요일']].head()
# print(labels)
# print(df_samsung)

# 필요 없는 행 삭제
# df_amore = df_amore.drop(index=[1773,1774,1775,1776,1777,1778,1779,1780,1781,1782])
# df_samsung = df_samsung.drop(index=[1037,1038,1039,2970,2950,2949,2917,2909,2886,2843])
print(df_amore.shape, df_samsung.shape) # (1035, 17) (1035, 17)

# # 스케일링 
# # scaler = MinMaxScaler()
# # scale_cols = ['시가', '고가', '저가', '종가', '거래량', '금액(백만)',
# #        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비', '년', '월', '일', '요일']
# # df_amore_scaled = scaler.fit_transform(df_amore[scale_cols])
# # df_amore_scaled = pd.DataFrame(df_amore_scaled)
# # df_amore_scaled.columns = scale_cols

# # scale_cols = ['시가', '고가', '저가', '종가', '거래량', '금액(백만)',
# #        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비', '년', '월', '일', '요일']
# # df_samsung = scaler.fit_transform(df_samsung[scale_cols])
# # df_samsung_scaled = scaler.fit_transform(df_samsung[scale_cols])
# # df_samsung_scaled = pd.DataFrame(df_samsung_scaled)
# # df_samsung_scaled.columns = scale_cols


# 피처, 타겟 컬럼 지정
feature_cols = ['시가', '고가', '저가', '종가', '거래량', '금액(백만)',
       '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비', '년', '월', '일', '요일']
target_cols = ['시가']

# 시계열 데이터 만드는 함수
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

x1 = split_x(df_amore[feature_cols], 10)
x2 = split_x(df_amore[feature_cols], 10)
y = split_x(df_samsung[target_cols], 10)
x1 = x1[:, :-1]                                
x2 = x2[:, :-1]                                
y = y[:, -1]

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, shuffle=False)
print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)
#    (820, 9, 17)    (206, 9, 17)    (820, 9, 17)   (206, 9, 17)   (820, 1)        (206, 1)

# reshape
x1_train = x1_train.reshape(820, 9*17)
x2_train = x2_train.reshape(820, 9*17)
x1_test = x1_test.reshape(206, 9*17)
x2_test = x2_test.reshape(206, 9*17)

# 스케일링 
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x1_train)      
x1_train = scaler.transform(x1_train) 
x1_test = scaler.transform(x1_test)    

scaler.fit(x2_train)    
x2_train = scaler.transform(x2_train) 
x2_test = scaler.transform(x2_test)    
# print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)

# re reshape
x1_train = x1_train.reshape(820, 9, 17)
x2_train = x2_train.reshape(820, 9, 17)
x1_test = x1_test.reshape(206, 9, 17)
x2_test = x2_test.reshape(206, 9, 17)
print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)


''' 2. 모델 구성 '''
# 2. 모델구성
# 2-1. 모델1
input1 = Input(shape=(9, 17))
dense1 = Conv1D(64, 2, activation='relu', name='d1')(input1)
dense2 = LSTM(128, activation='relu', name='d2')(dense1)
dense3 = Dense(64, activation='relu', name='d3')(dense2)
output1 = Dense(32, activation='relu', name='out_d1')(dense3)

# 2-2. 모델2
input2 = Input(shape=(9, 17))
dense11 = Conv1D(64, 2, activation='relu', name='d11')(input2)
dense12 = LSTM(128, activation='swish', name='d12')(dense11)
dense13 = Dense(64, activation='relu', name='d13')(dense12)
dense14 = Dense(32, activation='relu', name='d14')(dense13)
output2 = Dense(16, activation='relu', name='out_d2')(dense14)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='m1')
merge2 = Dense(64, activation='relu', name='mg2')(merge1)
merge3 = Dense(32, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])
# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto', verbose=1, restore_best_weights=True)        
hist = model.fit([x1_train, x2_train], y_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[es], verbose=1)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ' , loss)
y_predict = model.predict([x1_test, x2_test])
print('예측가격 : ', y_predict[-1:])
# r2= r2_score(last_output, y_test)
# print('loss : ' , loss)
# print('r2 스코어 : ', r2) 








