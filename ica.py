import pandas as pd
import sklearn as sk
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("Loading CSV Data")
sef_data = pd.read_csv("data/SEF_filled_mean.csv", dtype={"cty": str, "name": str, "_merge": str, "_merge1": str, "_merge2": str}, index_col=0)

data_values = sef_data.loc[:, 'median_rent2016':'has_mom_rh_gp_p50_l']

print("Running ICA")
data_values = StandardScaler().fit_transform(data_values)

ica = FastICA()

ica_output = ica.fit_transform(data_values)

sef_ica = pd.DataFrame(data = ica_output, index=sef_data.index)

print(sef_ica)

sef_ica.to_csv("data/SEF_ICA.csv")
sef_ica.to_excel("data/SEF_ICA.xlsx", sheet_name='SEF ICA Output')  

# plot = sef_pca.plot.scatter(x='Principal component 1', y='Principal component 2')

# plt.show()