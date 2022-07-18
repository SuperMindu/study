import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.layers.recurrent import SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import datetime as dt

''' 1-1) 데이터 로드 '''
# path = './_data/test_amore_0718/'
df_amore=pd.read_csv('./_data/test_amore_0718/아모레220718.csv', thousands=',', encoding='cp949') # 아모레 데이터 로드
df_samsung=pd.read_csv('./_data/test_amore_0718/삼성전자220718.csv', thousands=',', encoding='cp949') # 삼성전자 데이터 로드
df_amore.describe()
print(df_amore.columns)
# Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
#       dtype='object')

# 결측치 확인
# print(df_amore.info()) 

# 이상치 확인
q3 = df_amore.quantile(0.75) 
q1 = df_amore.quantile(0.25)
iqr = q3 - q1
# print(q3, q1, iqr)
# 등락률     1.2100
# 신용비     0.2100
# 외인비    36.9025
# Name: 0.75, dtype: float64 등락률    -1.2625
# 신용비     0.0300
# 외인비    31.6500
# Name: 0.25, dtype: float64 등락률    2.4725
# 신용비    0.1800
# 외인비    5.2525
# dtype: float64
print(df_amore)
#             시가       고가     저가     종가   전일비  Unnamed: 6 등락률 거래량  금액(백만) 신용비 개인    기관   외인(수량) 외국계 프로그램 외인비
# 일자
# 2022/07/18  134,500  136,500  134,000  134,500   ▲      1,000  0.75   22,762   3,071  0.00        0        0       0   1,526     -75  26.13
# 2022/07/15  135,000  136,000  131,000  133,500   ▼     -1,000 -0.74  127,723  16,976  0.43   12,945  -31,990  14,919  16,536   4,448  26.13
# 2022/07/14  135,000  138,000  134,000  134,500   ▼     -1,500 -1.10  172,436  23,340  0.45    1,081  -45,222  38,958  52,163  21,716  26.11
# 2022/07/13  131,000  137,500  130,000  136,000   ▲      6,000  4.62  195,845  26,349  0.46  -83,921   12,972  77,941  65,850  59,430  26.04
# 2022/07/12  130,000  131,000  128,500  130,000   ▼     -1,000 -0.76  117,074  15,151  0.45   -1,385  -28,403  39,351  23,476  22,740  25.91
# ...             ...      ...      ...      ...  ..        ...   ...      ...     ...   ...      ...      ...     ...     ...     ...    ...
# 2009/09/07  745,000  751,000  731,000  731,000   ▼    -19,000 -2.53   18,171  13,395  0.00    3,313  -10,072   6,585     192  -2,356  33.21
# 2009/09/04  735,000  750,000  725,000  750,000   ▲     11,000  1.49   11,273   8,329  0.02   -2,119    3,381     769     540     368  33.10
# 2009/09/03  715,000  739,000  714,000  739,000   ▲     20,000  2.78   14,782  10,785  0.00   -3,540      764   2,752     253   4,257  33.09
# 2009/09/02  740,000  740,000  719,000  719,000   ▼    -21,000 -2.84   15,828  11,493  0.01    3,362   -7,416   4,007    -706   1,080  33.04
# 2009/09/01  714,000  740,000  714,000  740,000   ▲     12,000  1.65    8,574   6,300  0.01   -2,444      625   1,235   2,117   2,794  32.97
# [3180 rows x 16 columns]
df_amore = df_amore.drop(['Unnamed: 6'], axis=1) 
print(df_amore)

# Date 컬럼을 자료형으로 변환
# pd.to_datetime(df_amore['일자'], format='%Y%m%d')
# 0      2020-01-07
# 1      2020-01-06
# 2      2020-01-03
# 3      2020-01-02
# 4      2019-12-30
# pd.to_datetime(df_amore['일자'])
# df_amore.dtypes

df_amore['날짜_datetime'] = pd.to_datetime(df_amore['일자'])

df_amore['년'] = df_amore['날짜_datetime'].dt.year
df_amore['월'] = df_amore['날짜_datetime'].dt.month
df_amore['일'] = df_amore['날짜_datetime'].dt.day
df_amore['요일'] = df_amore['날짜_datetime'].dt.day_name()
# print(df_amore)

df_amore = df_amore.drop(['일자', '날짜_datetime'], axis=1) 
print(df_amore)




'''
# 이상치 제거
# '시가' 열에 대하여 이상치 여부를 판별해주는 함수
def is_Start_price_outlier(df_amore):
    Start_price = df_amore['시가']
    if Start_price > q3['시가'] + 1.5 * iqr['시가'] or Start_price < q1['시가'] - 1.5 * iqr['시가']:
        return True
    else:
        return False

# apply 함수를 통하여 각 값의 이상치 여부를 찾고 새로운 열에 결과 저장
df_amore['시가_이상치여부'] = df_amore.apply(is_Start_price_outlier, axis = 1) # axis = 1 지정 필수

print(df_amore)


# 날짜 datetime 포맷으로 변환
# pd.to_datetime(df_weather['Date Time'], format='%Y%m%d')
# # 0      2020-01-07
# # 1      2020-01-06
# # 2      2020-01-03
# # 3      2020-01-02
# # 4      2019-12-30

# df_weather['일자'] = pd.to_datetime(df_weather['Date Time'], format='%Y%m%d')
# df_weather['연도'] =df_weather['Date Time'].dt.year
# df_weather['월'] =df_weather['Date Time'].dt.month
# df_weather['일'] =df_weather['Date Time'].dt.day



# Normalization 정규화
# MinMaxScaler를 해주면 전체 데이터는 0, 1사이의 값을 가짐

scaler = RobustScaler()
scale_cols = ['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
       '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']
df_scaled = scaler.fit_transform(df_amore[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

print(df_scaled)



# 학습을 시킬 데이터 셋 생성
# TEST_SIZE = 200은 학습은 과거부터 200일 이전의 데이터를 학습하게 되고, TEST를 위해서 이후 200일의 데이터로 모델이 주가를 예측하도록 한 다음, 
# 실제 데이터와 오차가 얼마나 있는지 확인함
TEST_SIZE = 200

train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]

def make_dataset(data, label, window_size=20): # window_size=20 과거 20일을 기준으로 그 다음날의 데이터를 예측함
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)
# 위의 함수는 정해진 window_size에 기반하여 20일 기간의 데이터 셋을 묶어 주는 역할을 함
# 즉, 순차적으로 20일 동안의 데이터 셋을 묶고, 이에 맞는 label (예측 데이터)와 함께 return 해줌



# feature 와 label(예측 데이터) 정의
feature_cols = ['p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
       'wd (deg)']
label_cols = ['T (degC)']

train_feature = train[feature_cols]
train_label = train[label_cols]


# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 20)



# train, validation set 생성
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.2)

# print(x_train.shape, x_valid.shape)   #(336264, 20, 13) (84067, 20, 13)

# test dataset (실제 예측 해볼 데이터)
# test_feature, test_label = make_dataset(test_feature, test_label, 20)
# print(test_feature.shape, test_label.shape)




#2.모델 구성
model = Sequential()
model.add(LSTM(16, 
               input_shape=(train_feature.shape[1], train_feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,                       
#                       )

hist = model.fit(x_train, y_train, epochs=10, 
                 batch_size=1024, validation_split=0.2, 
                 callbacks=[es], 
                 verbose=1) 



# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
'''