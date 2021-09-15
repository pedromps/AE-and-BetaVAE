# covidcurvefit
This repository contains code for an Autoencoder and a β Variational Autoencoder and some results. The code was developed for longitudinal time series data in my Master's Thesis.


The methods are adapted from other repositories, namely the following:


https://keras.io/examples/generative/vae/ - for the VAE structure, namely the the Sampling layer and the keras based model


https://github.com/cran2367/understanding-lstm-autoencoder - for the LSTM Autoencoder, which was an adequated arquitecture for the problem at hand.


And the paper below:


https://openreview.net/pdf?id=Sy2fzU9gl - where the "β part" is introduced into the standard VAE and after reading the paper the changes to code were noted and implemented.


The code was adjusted around the data used, which cannot be disclosed fully due to the nature of the Master's Thesis for which they were used. Naturally the hyperparameters of these algorithms can be changed and adapted to match the needs of a different dataset.


Finally, the folder with the results contains a plot taken from my Master's Thesis where there are imputation results with the β-VAE, another generative method and simpler imputation methods. The results are described in a file included in that directory.