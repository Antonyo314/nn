import numpy as np
from joblib import load

model = load('model.joblib')

X = list()

for i in range(87):
    print(f'введите значение input_{i}')
    X.append(float(input()))

result_dict = {0: 'система не заражена', 1: 'система заражена'}
print(result_dict[model.predict(np.array(X).reshape((1, -1)))[0]])