import numpy as np
from joblib import load

# загружаем модель (весь пайплайн)
model = load('model.joblib')

# создаем пустой список
X = list()

# с клавиатуры вводим занчения всех признаков и добавляем в X
for i in range(87):
    print(f'введите значение input_{i}')
    X.append(float(input()))

# делаем предсказание, выводим результат переводя из чисел в названия классов
result_dict = {0: 'система не заражена', 1: 'система заражена'}
print(result_dict[model.predict(np.array(X).reshape((1, -1)))[0]])
