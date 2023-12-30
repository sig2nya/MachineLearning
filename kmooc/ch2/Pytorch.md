Numpy
=====
- 개요 : 행렬, 다차원 배열 등을 쉽게 처리하도록 도와주는 python의 라이브러리.
- Array : Numpy의 기본 단위. 배열을 통하여 데이터를 관리한다. 0차원 Scalar, 1차원 Vector, 2차원 행렬, 3차원 이상 Tensor 등으로 구성될 수 있다.
<img width="551" alt="image" src="https://github.com/sig2nya/MachineLearning/assets/70207093/c0c4be67-dd83-43c1-9fad-d6c2377c4b94">

```python
import numpy as np

arr1 = [1, 2, 3, 4, 5]
arr2 = np.array([1, 2, 3, 4, 5])
arr3 = np.array([[1, 2], [3, 4]])
arr4 = np.array([[5, 6], [7, 8]])

print(arr3 + arr4)
print(arr3 * arr4)

arr5 = np.arange(10)
print(arr5[0])
print(arr5[3:9])
print(arr5[:])

arr6 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr6[0, 1])
print(arr6[0, :])
print(arr6[:, 0:2]
```

<img width="376" alt="image" src="https://github.com/sig2nya/MachineLearning/assets/70207093/569847aa-fbdd-44fc-be40-521329872554">
