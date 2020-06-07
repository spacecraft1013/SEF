import pandas as pd
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("Loading CSV Data")
sef_data = pd.read_csv("data/SEF_filled_mean.csv", dtype={"cty": str, "name": str, "_merge": str, "_merge1": str, "_merge2": str}, index_col=0)

data_values = sef_data.loc[:, 'median_rent2016':'has_mom_rh_gp_p50_l']
print(data_values)

data_values = StandardScaler().fit_transform(data_values)

pca = PCA(n_components=100)

pca_output = pca.fit_transform(data_values)

sef_pca = pd.DataFrame(data = pca_output, index=sef_data.index)

print(sef_pca)

sef_pca.to_csv("data/SEF_PCA_100.csv")
sef_pca.to_excel("data/SEF_PCA_100.xlsx", sheet_name='SEF PCA Output')  

# plot = sef_pca.plot.scatter(x='Principal component 1', y='Principal component 2')

# plt.show()