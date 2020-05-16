import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

print('подготовка данных и тренировка...')

df_train = pd.read_excel('data/nn1.xlsx')
df_test = pd.read_excel('data/nn2.xlsx')

X_train = df_train.drop(['output'], axis=1)
y_train = df_train['output']

X_test = df_test.drop(['output'], axis=1)
y_test = df_test['output']

scaler = StandardScaler()
X_train = X_train.replace({' ': 0})
X_train = scaler.fit_transform(X_train)

steps = [('scaler', StandardScaler()), ('SVM', MLPClassifier(hidden_layer_sizes=(10, 10,)))]
pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

y_test_pred = pipeline.predict(scaler.transform(X_test.replace({' ': 0})))

print(f'точность модели на тестовой выборке: {round(accuracy_score(y_test, y_test_pred) * 100, 2)}%')

dump(pipeline, 'model.joblib')