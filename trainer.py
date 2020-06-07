import keras
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import random

print("Loading CSV Data")
sef_data = pd.read_csv("data/SEF_filled_mean.csv", dtype={"cty": str, "name": str, "_merge": str, "_merge1": str, "_merge2": str}, index_col=0)

data_values = StandardScaler().fit_transform(sef_data.loc[:, 'median_rent2016':'has_mom_rh_gp_p50_l'])

features = np.reshape(data_values, (-1, 3188, 1))
cases = pd.DataFrame(sef_data['v131']).values

# random.Random(5).shuffle(features)
# random.Random(5).shuffle(cases)

def r2(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


inputs = keras.layers.Input(shape=(3188, 1))
x = keras.layers.Dense(512, activation='relu')(inputs)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(16, activation='relu')(x)
x = keras.layers.Flatten()(x)
output = keras.layers.Dense(1, activation='relu')(x)

model = keras.models.Model(inputs=inputs, outputs=output)

X_train, X_test, y_train, y_test = train_test_split(features, cases, test_size=0.3, random_state=25)

model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=[r2])

model.fit(X_train, y_train, epochs=10000, validation_data=(X_test, y_test))

#y_pred = model.predict(X_test, verbose=1)

# y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(y_test, axis=1)

#matrix = confusion_matrix(y_test, y_pred)

model.save("data/model_1000.h5")