import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

print("Loading CSV Data")
sef_data = pd.read_csv("data/SEF_filtered.csv", index_col=0)

data_values = sef_data.loc[:, 'median_rent2016':'has_mom_rh_gp_p50_l']

print("Filling Values using Multivariate Imputation")
imp = IterativeImputer(verbose=2)
data_values_imputed = imp.fit_transform(data_values)
sef_data.replace(data_values, data_values_imputed)

print("Saving Output")
sef_data.to_csv("data/SEF_filled_imputed.csv")
sef_data.to_excel("data/SEF_filled_imputed.xlsx", sheet_name='SEF Data')