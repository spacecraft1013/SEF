import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

print("Loading CSV Data")
sef_data = pd.read_csv("data/SEF_filled_mean.csv", index_col=0)

data_values = sef_data.loc[:, 'median_rent2016':'has_mom_rh_gp_p50_l']

kmeans = KMeans(verbose=1).fit(data_values)
transformed_data = kmeans.transform(data_values)
print(transformed_data.shape)

print("Saving Output")

print(data_values.columns)
transformed_dataframe = pd.DataFrame(transformed_data, index=data_values.index)
print(transformed_dataframe)

transformed_dataframe.to_csv("data/kmeans_data.csv")