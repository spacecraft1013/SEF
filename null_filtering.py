import pandas as pd

print("Loading CSV Data")
sef_data = pd.read_csv("data/SEF.csv", index_col=0)
print("Original DataFrame Shape: " + str(sef_data.shape))

print("Filtering Values")
sef_data.dropna(axis=1, thresh=2500, inplace=True)
sef_data.dropna(axis=0, how='all', thresh=10, inplace=True)

print("New DataFrame Shape: " + str(sef_data.shape))

print("Saving DataFrame")
sef_data.to_csv("data/SEF_filtered.csv")
sef_data.to_excel("data/SEF_filtered.xlsx", sheet_name='SEF Data Filtered')  