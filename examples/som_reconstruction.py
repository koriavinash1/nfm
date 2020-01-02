import os, sys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
percentages = [0.01, 0.1, 0.2, 0.4, 0.5, 0.6]

for percentage in percentages:
	data = []

	save_path = '../logs/SOM_weights_MNIST_noise_{}.npy'.format(percentage)
	wts  = np.load(save_path).reshape(-1, 784)
	print ("============{}============".format(wts.shape))
	kmeans = KMeans(n_clusters=10).fit(wts)
	centers = kmeans.cluster_centers_


	
	for i in range(2):

		for j in range(5):
			plt.subplot(2, 5, i*5 + j + 1)
			plt.imshow(centers[i*5+j].reshape(28, 28).T)
			if (i == 0) and (j == 0): plt.title("MNIST Noise {}".format(percentage))
	
	plt.show()	
	
