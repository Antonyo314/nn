import pandas as pd
from joblib import load

# открываем excel файл
df_to_pred = pd.read_excel('data/to_predict.xlsx')

#загружаем модель (весь пайплайн)
model = load('model.joblib')

#делаем предсказание, выводим результат переводя из чисел в названия классов
result_dict = {0: 'система не заражена', 1: 'система заражена'}
print(result_dict[model.predict(df_to_pred)[0]])