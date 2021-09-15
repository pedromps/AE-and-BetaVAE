# -*- coding: utf-8 -*-
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Masking
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

#uncomment this line to get the code to work. the data has to be in appropriate dimensions for LSTMs and in numpy format
# x_data = np.load(os.path.join(os.getcwd(), "vae_data/data.npy"))

#normalise data
#maxima and minima of the metrics across all counties
norm_max = np.nanmax(x_data, axis=1)
norm_min = np.nanmin(x_data, axis=1)
for i in range(x_data.shape[0]):
    x_data[i,:,:] = (x_data[i,:,:]-norm_min[i])/(norm_max[i]-norm_min[i])

#we train with all data because the VAE generates data, it doesnt need a test set
x_train = np.copy(x_data)
NanIndex = np.where(np.isnan(x_train))
#NotNanIndex = np.where(np.isnan(x_train)==False)
x_train[NanIndex] = -1


outer_dim = int(64)
latent_dim = int(32)

#layers for longitudinal data based on https://github.com/cran2367/understanding-lstm-autoencoder 
model = Sequential()
model.add(Masking(mask_value = -1, input_shape = (x_train.shape[1], x_train.shape[2])))
model.add(LSTM(outer_dim, activation = 'tanh', input_shape = (x_train.shape[1], x_train.shape[2]), return_sequences = True))
model.add(LSTM(latent_dim, activation = 'tanh', return_sequences = False))
model.add(RepeatVector(x_train.shape[1]))
model.add(LSTM(latent_dim, activation = 'tanh', return_sequences = True))
model.add(LSTM(outer_dim, activation = 'tanh', return_sequences = True))
model.add(TimeDistributed(Dense(x_train.shape[2], activation = 'relu')))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(x_train, x_train, epochs=1000, batch_size=128, verbose=1)


plt.figure(figsize=(18.64, 9.48))
plt.plot(model.history.history["loss"], label="Training Loss")
plt.legend()
plt.grid()
plt.show()

x_aux = np.copy(x_train)
samp = model.predict(x_train, verbose=0)
x_aux[NanIndex] = samp[NanIndex]