from sklearn.feature_selection import SelectKBest
import pandas as pd
import pickle
import numpy as np

print("Loading CSV Data")
sef_data = pd.read_csv("data/SEF_filled_mean.csv", dtype={"cty": str, "name": str, "_merge": str, "_merge1": str, "_merge2": str}, index_col=0)

data_values = sef_data.loc[:, 'median_rent2016':'has_mom_rh_gp_p50_l']
cases = sef_data['v131']

selector = SelectKBest(k=100).fit(data_values, cases)
selected_data = selector.transform(data_values)
print(selected_data)

print("Saving Output")
selected_data = pd.DataFrame(selected_data, index=sef_data.index)

selected_data.to_csv("data/selected_data_100.csv")
selected_data.to_excel("data/selected_data_100.xlsx", sheet_name='SelectKBest from SEF')