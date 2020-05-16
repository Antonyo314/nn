import pandas as pd
from joblib import load

df_to_pred = pd.read_excel('data/to_predict.xlsx')
model = load('model.joblib')

result_dict = {0: 'система не заражена', 1: 'система заражена'}
print(result_dict[model.predict(df_to_pred)[0]])
