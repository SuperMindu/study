# [과제]
# 3가지 원핫 인코딩 방식을 비교할 것
# 1. pandas의 get_dummies -> 
# 2. tensorflow의 to_categorical -> 무적권 0부터 시작함 (0이 없으면 만듦) (앞을 채워줘야 하는 경우 요놈 쓰면 좋음)
# 3. sklearn의 OneHotEncoder -> 

# 미세한 차이를 찾아라

'''
원-핫 인코딩(One-Hot Encoding) 이란? 
# 아웃풋 레이어에 그냥 3을 주고 실행을 시켜봤더니 'ValueError: Shapes (None, 1) and (None, 3) are incompatible' 이런 오류가 뜸 (iris)
# 처음에 y.shape가 (150, ) 였는데 이걸 (150, 3)으로 바꿔줘야 함
# 컴퓨터 또는 기계는 문자보다는 숫자를 더 잘 처리 할 수 있음
# 이를 위해 자연어 처리에서는 문자를 숫자로 바꾸는 여러가지 기법들이 있음 
# 원-핫 인코딩(One-Hot Encoding)은 그 많은 기법 중에서 단어를 표현하는 가장 기본적인 표현 방법이며, 
# 머신 러닝, 딥 러닝을 하기 위해서는 반드시 배워야 하는 표현 방법임
# 원핫인코딩은 문자를 숫자, 더 구체적으로는 벡터로 바꾸는 여러 방법 중의 하나임
# keras 에서 할 수 있는 방법이 있고, sklearn 에서도 방법이 있음

from tensorflow.python.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import to_categorical  <-- keras 에서 

from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder  <-- sklearn 에서

집가서 좀 더 아라보자 (사이킷런 원핫인코딩)
https://daily-studyandwork.tistory.com/36 <- 여기
https://psystat.tistory.com/136 <- 여기 
'''
