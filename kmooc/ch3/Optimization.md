Optimization
============
 - 정의 : 임의의 model에 대해 output이 원하는 조건을 만족하도록 하는 최적의 input을 찾는 문제
 - ex) 점수 / 이윤 등의 최대화, 에러 / 손실 등의 최소화, 기계의 연료 소비를 최소화, 투자 금액의 이윤을 최대화, 행복 지수의 최대화 등

Optimization Problem
====================
 - 정의 : 함수 f(x)의 값을 최대화 / 최소화 하는 x를 찾는 문제</br>
   <img width="520" alt="image" src="https://user-images.githubusercontent.com/70207093/209456333-5a476c20-6ef7-40a5-b95a-09d7112a68f2.png"></br>
   <img width="256" alt="image" src="https://user-images.githubusercontent.com/70207093/210164400-d7580447-7b5b-4500-8811-57c41b71e945.png">
 - How to : 최대 / 최소값을 만족하는 가능한 x의 값을 모두 대입 -> Cost 발생이 큰 단점이 있으며, f(x, y, ... )과 같이 다변수 함수의 경우 불가능 -> Numerical Optimization 필요

Numerical Optimization
======================
 - 정의 : 반복적 시행착오에 의한 최적화 <img width="25" alt="image" src="https://user-images.githubusercontent.com/70207093/209456435-45f7cc88-8ffa-497d-94da-ff45b29f003d.png">값을 찾는 문제, 함수 위치가 최적점 x가 될때까지 가능한 적은 횟수로 x 값을 찾는다.
 - 과정
 - - 1. 현재 위치 <img width="29" alt="image" src="https://user-images.githubusercontent.com/70207093/209456453-e13b1fab-6473-4d4f-b600-dc3299c14245.png">가 최적점인지 판단
 - - 2. a를 시도 후, 최적점이 아닐시 다음 시도할 <img width="50" alt="image" src="https://user-images.githubusercontent.com/70207093/209456443-a58eefaa-1504-4e45-b2f3-d7b462aabadb.png">를 찾는 과정

Gradient Descent
================
 - 정의 : 현위치 x에서 기울기 <img width="70" alt="image" src="https://user-images.githubusercontent.com/70207093/209456410-9e5ec881-66a7-44b7-8ae5-e556c277f096.png">를 이용하여 다음 위치 <img width="50" alt="image" src="https://user-images.githubusercontent.com/70207093/209456422-8a559fe5-081f-44cd-9c3f-ce155094dcf7.png">
를 결정하는 방법
 - 수식 : <img width="282" alt="image" src="https://user-images.githubusercontent.com/70207093/209456459-04bd67f0-bfb7-4788-b2b3-b46b7c1909d8.png">
 - λ의 의미
 - - step size
 - - 위치를 옮기는 거리를 결정하는 비례상수
 - - 구현시, 사용자가 임의로 지정해주어야 하는 hyper parameter
 - - learning rate(학습률)
 - <img width="95" alt="image" src="https://user-images.githubusercontent.com/70207093/209456482-5e5d2377-9604-4261-8050-45155f347182.png">
 - - 가장 크게 감소하는 기울기 방향으로 λ만큼 이동(가장 급격한 경사를 따라 내려가다보면, 최저점에 도달 가능)

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def f1(x):
  retrun 3 * x ** 2 + 6 * x + 1
  
def fld(x):
  """f(x)의 도함수"""
  return 6 * x + 6
  
xx = np.linspace(-4, 2, 200) #linear space
  
step_size = 0.1

plt.plot(xx, f1(xx), '--k')

x = -3
plt.plot(x, f1(x), 'go', markersize = 10)
plt.text(x + 0.2, f1(x) + 0.1, "start")
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'b--')
print("start : x_0={}, g_0={}".format(x, f1d(x)))

x = x - step_size * f1d(x)
plt.plot(x, f1(x), 'go', markersize = 10)
plt.text(x + 0.2, f1(x) + 0.1, "x_1")
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'b--')
print("x_1 : x_1={}, g_1={}".format(x, f1d(x)))

x = x - step_size * f1d(x)
plt.plot(x, f1(x), 'go', markersize = 10)
plt.text(x + 0.2, f1(x) + 0.1, "x_2")
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'b--')
print("x_2 : x_2={}, g_2={}".format(x, f1d(x)))

plt.ylim(-5, 15)
plt.xlabel('x')
plt.grid(True)
plt.show()
```
<img width="298" alt="image" src="https://user-images.githubusercontent.com/70207093/209456598-3028e030-45d5-46d7-b9c5-d29766409733.png">

```python
step_size = 0.3 # step size -> 즉 람다 폭을 매우 크게 주면, model이 최적화 함수를 찾는데 오래 걸릴 것이다.

plt.plot(xx, f1(xx), '--k')

x = -3
plt.plot(x, f1(x), 'go', markersize = 10)
plt.text(x + 0.2, f1(x) + 0.1, "start")
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'b--')
print("start : x_0={}, g_0={}".format(x, f1d(x)))

x = x - step_size * f1d(x)
plt.plot(x, f1(x), 'go', markersize = 10)
plt.text(x + 0.2, f1(x) + 0.1, "x_1")
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'b--')
print("x_1 : x_1={}, g_1={}".format(x, f1d(x)))

x = x - step_size * f1d(x)
plt.plot(x, f1(x), 'go', markersize = 10)
plt.text(x + 0.2, f1(x) + 0.1, "x_2")
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'b--')
print("x_2 : x_2={}, g_2={}".format(x, f1d(x)))

plt.ylim(-5, 15)
plt.xlabel('x')
plt.grid(True)
plt.show()
```
<img width="295" alt="image" src="https://user-images.githubusercontent.com/70207093/209456615-11e52d37-fd27-4341-9a96-d4a0d4aede25.png">
