import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

print('подготовка данных и тренировка...')

# открываем excel файлы
df_train = pd.read_excel('data/nn1.xlsx')
df_test = pd.read_excel('data/nn2.xlsx')

# разделяем тренеровочные на входные признаки и целевую переменную (X и y)
X_train = df_train.drop(['output'], axis=1)
y_train = df_train['output']

# разделяем тестовые на входные признаки и целевую переменную (X и y)
X_test = df_test.drop(['output'], axis=1)
y_test = df_test['output']

# меняем незаполненные значения на 0 (возможно лучше было бы поменять на медиану или среднее по признаку, надо пробовать)
X_train = X_train.replace({' ': 0})

# обьеденяем нормализацию данных и обучение в один пайплайн
steps = [('scaler', StandardScaler()), ('MLPClassifier', MLPClassifier(hidden_layer_sizes=(10, 10,)))]
pipeline = Pipeline(steps)

# вызываем для него метод fit (он выполянет fit_transform для scaler и fit для MLPClassifier)
pipeline.fit(X_train, y_train)

# делаем предсказание на тестовых данных
y_test_pred = pipeline.predict(X_test.replace({' ': 0}))

# выводим точномть модели
print(f'точность модели на тестовой выборке: {round(accuracy_score(y_test, y_test_pred) * 100, 2)}%')

# сохраняем модель
dump(pipeline, 'model.joblib')
