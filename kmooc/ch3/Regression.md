Regression
==========
 - 정의 : 데이터를 2차원 공간에 표시하여, 이 데이터를 잘 설명하는 직선 혹은 곡선을 찾는 문제. x, y가 실수일 때, 함수 f(x)를 예측하는 것
 - Linear Regression : 입력 데이터를 잘 설명하는 가중치(기울기) w와 bias(절편) b를 찾는 문제. y = wx + b
 - ex) 부모의 키 - 자녀의 키, 면적 - 주택 가격, 연령 - 실업률, 공부 시간 - 학점, CPU 속도 - 프로그램 실행시간 등
   <img width="364" alt="image" src="https://user-images.githubusercontent.com/70207093/209456941-9357b1d3-9fbd-47e4-bcb0-b42ca7ad22dd.png">

Object Function
===============
 - 정의 : 학습된 알고리즘이 training data set과의 변수를 얼마나 잘 모델링하는가 평가하는 함수
 - hypothesis : object function을 최적화 수행하여 training data set을 가장 잘 설명하는 설계 -> hypothesis 모델의 오차를 minimize
 - MSE : Mean Square Error</br>
   <img width="414" alt="image" src="https://user-images.githubusercontent.com/70207093/209456986-0ce92d4b-9691-4e3c-ada9-63ddc5bf4751.png">

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

lr = 0.01 # step_size
n_iter = 10 # 반복 횟수

x_train = np.array([[1], [2], [3]]) # input data set
y_train = np.array([[2], [4], [6]]) # output data set
x_train_b = np.c_[np.ones((len(x_train), 1)), x_train] # input data set with bias
theta = np.random.randn(2, 1) # parameter(b, w)

plt.scatter(x_train, y_train)
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.grid(True)
plt.show()
```
