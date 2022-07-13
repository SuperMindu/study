# 시계열 데이터를 짜르는 함수
# https://thestoryofcosmetics.tistory.com/35 <- for문 여기 보고 공부해보자

# for문은 정해진 횟수만큼 반복하는 구조이고
# while문은 어떤 조건이 만족되는 동안, 계속 반복하는 구조임
# 
# for 변수 in range(종료값): # range()이건 함수임. range()함수는 입력 받은 숫자에 해당되는 범위의 값을 반복 가능한 객체로 만들어 출력해줌. 쉽게 말해서 range()함수를 이용하면 특정 구간의 정수들을 생성할 수 있음
import numpy as np

a = np.array(range(1, 11)) # 10개짜리 데이터를 size대로 짜르겠드아
size = 5

def split_x(dataset, size): # dataset을 size대로 split
    aaa = []
    for i in range(len(dataset) - size + 1): # for 변수 in range(종료값): # range()이건 함수임. range()함수는 입력 받은 숫자에 해당되는 범위의 값을 반복 가능한 객체로 만들어 출력해줌. 쉽게 말해서 range()함수를 이용하면 특정 구간의 정수들을 생성할 수 있음
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size) # a를 size대로 잘라서 bbb에 집어 넣겠다
print(bbb)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
print(bbb.shape) # (6, 5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]] 
# [ 5  6  7  8  9 10]
print(x.shape, y.shape) # (6, 4) (6,)