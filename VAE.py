# -*- coding: utf-8 -*-
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from scipy.stats import norm
import tensorflow as tf


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = 1
        dim = z_mean.shape[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean = 0, stddev = 1)
        return z_mean + tf.exp(z_log_var*0.5)*epsilon
    
    
#sampling class based on https://keras.io/examples/generative/vae/
class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        #beta-vae will need the beta, as described in the paper https://openreview.net/forum?id=Sy2fzU9gl
        self.beta = beta
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[1]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            #data >= 0 because the data with negative values (-1) is masked 
            #RMSE vs MAE?
            reconstruction_loss = tf.reduce_mean(tf.sqrt(tf.keras.losses.MSE(data[data>=0], reconstruction[data>=0])))
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            
            #beta to turn this into a beta-VAE
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    

#uncomment this line to get the code to work. the data has to be in appropriate dimensions for LSTMs and in numpy format
#x_data = np.load(os.path.join(os.getcwd(), "vae_data/data.npy"))

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
encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
x = layers.Masking(mask_value = -1, input_shape = (x_train.shape[1], x_train.shape[2]))(encoder_inputs)
x = layers.LSTM(outer_dim, activation = 'tanh', input_shape = (x_train.shape[1], x_train.shape[2]), return_sequences = True)(x)
x = layers.LSTM(latent_dim, activation = 'tanh', input_shape = (x_train.shape[1], x_train.shape[2]), return_sequences = False)(x)
z_mean = layers.Dense(1, activation=None)(x)
z_mean = layers.RepeatVector(x_train.shape[1], name = "z_mean")(x)
z_log_var = layers.Dense(1, activation=None)(x)
z_log_var = layers.RepeatVector(x_train.shape[1], name = "z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name = "encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(x_train.shape[1], latent_dim))
x = layers.LSTM(latent_dim, activation = 'tanh', return_sequences = True)(latent_inputs)
x = layers.LSTM(outer_dim, activation = 'tanh', return_sequences = True)(x)
x = layers.TimeDistributed(layers.Dense(x_train.shape[2], activation = 'relu'))(x)
decoder = keras.Model(latent_inputs, x, name = "decoder")
decoder.summary()

#beta value is input here
vae = VAE(encoder, decoder, beta=0.1)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
vae.fit(x_train, epochs=100, batch_size=128, verbose=1)


plt.figure(figsize=(18.64, 9.48))
plt.plot(vae.history.history["loss"], label="Training Loss")
plt.plot(vae.history.history["kl_loss"], label="Kullback-Leibler Divergence")
plt.legend()
plt.grid()
plt.show()

#imputing data
x_samp = np.copy(x_train)
x_aux = np.copy(x_samp)
#alternatively you can include this step. missing data can be either nan or 0
# x_samp[NanIndex] = 0
x_m, x_l, x_samp = encoder(x_samp)
#z mean and z log var, respectively
x_m = x_m.numpy()
x_l = x_l.numpy()
x_sampled = x_samp.numpy()
samp = decoder(x_samp)
#samp contains an example of fully generated data
samp = samp.numpy()
x_aux[NanIndex] = samp[NanIndex]
x_aux = x_aux*(norm_max-norm_min) + norm_min




# statistics: histogram and normal distribution to better see the latent space and how it follows a normal distr.
input_x = x_sampled
mean_val = np.mean(input_x)
stddev = np.std(input_x)
domain = np.linspace(np.min(input_x), np.max(input_x))
plt.figure(figsize=(18.64, 9.48))
plt.plot(domain, norm.pdf(domain, mean_val, stddev))
plt.hist(np.reshape(x_sampled, (input_x.shape[0]*input_x.shape[1], input_x.shape[2])), edgecolor = 'black', density=True)
plt.title("Distribution of latent space points with $\mu$ = {:.2f} and $\sigma$ = {:.2f}".format(mean_val, stddev))
plt.xlabel("Value")
plt.ylabel("Frequency of each result")
plt.grid()
plt.show()