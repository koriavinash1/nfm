import os, sys
import numpy as np
sys.path.append('..')
from nfm.helper.GaussianStatistics import *
from nfm.helper.configure import Config
from nfm.SOM import SOM
from nfm.NFM import NFM
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

percentages = [0.01]#, 0.1, 0.2, 0.4, 0.5, 0.6]
som_output = (28, 28)

som = SOM(som_output, 25, learning_rate=1e-2, rad = 5, sig = 3)
nfm = NFM(size = som_output)


for percentage in percentages:
	save_path = '../logs/SOM_weights_MNIST_noise_{}.npy'.format(percentage)
	wts = som.load_weights(save_path)

	for _data_ in x_test[:30]:
		noise = percentage*np.random.randn(_data_.shape[0]*_data_.shape[1])
		data  = _data_.flatten('F')/255.0 + noise
		data = 1.0*(np.array(data))

		som_response = som.response(data, wts)
		nfm.spatioTemporalData(T = 5, 
								dynamics     = 'FHN',
                                visualize    = '3d',
                                dynamic_args = {'alpha': 0.5,
                                                'beta' : 0.1,
                                                'gamma': 0.1,
                                                'tau'  : 0.08,
                                                'ci'   : -0.2})
