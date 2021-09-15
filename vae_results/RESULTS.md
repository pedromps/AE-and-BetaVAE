# covidcurvefit
The data used to produce the plot present in this folder is from the medical research field.


The image comes from an experiment summed up as follows:


Full data would be partially hidden from the imputation algorithms, henceforth called introduced missing data, in percentages varying as shown in the X axis. The introduced missing data would be subjected to the imputation algorithms labelled in the image. The results of each imputation method would be compared with the real (hidden) value by computing the mean squared error (MSE) between all instances of imputed values and the real values. The plot reports the MSE of each method.


For the purpose of this repository, the GAIN method will not be discussed here.


First and foremost, the Random imputation is used as a "sanity check" in the sense that it is expected to report the worst MSE for every situation and thus it should always perform worst.

The Mean and Forward Imputations have always the best performance. This happened as, in the scope of the thesis, the data didn't have much variance and thus the real values of data would always be in a relatively small range and in such conditions these two imputations would always perform best. However, the VAE imputation had a competitive performance. It's error would always have fluctuations between each percentage, due to its generative nature, but it had an advantage that MSE does not highlight. The generative method, even when having more MSE, produced outputs that resembled more those of real patients than the Forward or Mean imputations.