import os, sys
import numpy as np
sys.path.append('..')
from nfm.helper.GaussianStatistics import *
from nfm.helper.configure import Config
from nfm.SOM import SOM
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Gstat  = GaussianStatistics()
config = Config()

# Data Generation....

"""
for angle in range(0, 180, 2):
    _bar = Gstat.OrientationBar(N = config.N,
                                theta = angle,
                                mu = config.mu,
                                Sigma = config.std,
                                display = False)
    data.append(_bar.flatten('F'))
data = 1.0*(np.array(data) > 0.2)
"""

percentages = [0.01, 0.1, 0.2, 0.4, 0.5, 0.6]

for percentage in percentages:
	data = []
	for _data_ in x_test[:30]:
		noise = percentage*np.random.randn(_data_.shape[0]*_data_.shape[1])
		data.append(_data_.flatten('F')/255.0 + noise)

	data = 1.0*(np.array(data))
	print (data.shape)

	save_path = '../logs/SOM_weights_MNIST_noise_{}.npy'.format(percentage)

	# ##
	som = SOM((28, 28), 25, learning_rate=1e-2, rad = 5, sig = 3)
	# som.fit(data)
	# som.save_weights(save_path)

	som.load_weights(save_path)
	som.moveresp(data)
	som.view_weights(path = '../logs/SOM_weights_MNIST_noise_{}.png'.format(percentage))
