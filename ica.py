import pandas as pd
import sklearn as sk
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

print("Loading CSV Data")
sef_data = pd.read_csv("data/SEF_filled_mean.csv", dtype={"cty": str, "name": str, "_merge": str, "_merge1": str, "_merge2": str}, index_col=0)

data_values = sef_data.loc[:, 'median_rent2016':'has_mom_rh_gp_p50_l']

print("Running ICA")
data_values = StandardScaler().fit_transform(data_values)

ica = FastICA(n_components=100)

ica_output = ica.fit_transform(data_values)

# def g(x):
#     return np.tanh(x)
# def g_der(x):
#     return 1 - g(x) * g(x)
    
# def center(X):
#     X = np.array(X)
    
#     mean = X.mean(axis=1, keepdims=True)
    
#     return X- mean

# def whitening(X):
#     cov = np.cov(X)
#     d, E = np.linalg.eigh(cov)
#     D = np.diag(d)
#     D_inv = np.sqrt(np.linalg.inv(D))
#     X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
#     return X_whiten

# def calculate_new_w(w, X):
#     w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
#     w_new /= np.sqrt((w_new ** 2).sum())
#     return w_new

# def ica(X, iterations, tolerance=1e-5):
#     X = center(X)
    
#     X = whitening(X)
        
#     components_nr = X.shape[0]
#     W = np.zeros((components_nr, components_nr), dtype=X.dtype)

#     for i in range(components_nr):
            
#         w = np.random.rand(components_nr)
#         for j in range(iterations):
            
#             w_new = calculate_new_w(w, X)
            
#             if i >= 1:
#                 w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            
#             distance = np.abs(np.abs((w * w_new).sum()) - 1)
            
#             w = w_new
            
#             if distance < tolerance:
#                 break
                
#         W[i, :] = w

#         S = np.dot(W, X)

#         return S

# ica_output = ica(data_values, 10)

sef_ica = pd.DataFrame(data = ica_output, index=sef_data.index)

print(sef_ica)

print("Saving output")
sef_ica.to_csv("data/SEF_ICA.csv")
sef_ica.to_excel("data/SEF_ICA.xlsx", sheet_name='SEF ICA Output')  

# plot = sef_pca.plot.scatter(x='Principal component 1', y='Principal component 2')

# plt.show()