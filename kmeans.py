import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

print("Loading CSV Data")
sef_data = pd.read_csv("data/SEF_filled_mean.csv", index_col=0)

data_values = sef_data.loc[:, 'median_rent2016':'has_mom_rh_gp_p50_l']

kmeans = KMeans(verbose=1).fit(data_values)
transformed_data = kmeans.transform(data_values)

print("Saving Output")
labels = pd.DataFrame(kmeans.labels_, index=data_values.index)
transformed_dataframe = pd.DataFrame(transformed_data, index=data_values.index)

transformed_dataframe.to_csv("data/kmeans_data.csv")
labels.to_csv("data/kmeans_cluster_labels.csv")