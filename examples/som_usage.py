import os, sys
import numpy as np
sys.path.append('..')
from nfm.helper.GaussianStatistics import *
from nfm.helper.configure import Config
from nfm.SOM import SOM

Gstat  = GaussianStatistics()
config = Config()

# Data Generation....
data = []
for angle in range(0, 180, 1):
    _bar = Gstat.OrientationBar(N = config.N,
                                theta = angle,
                                mu = config.mu,
                                Sigma = config.std,
                                display = False)
    data.append(_bar.flatten('F'))
data = np.array(data)

print (data.shape)
# ##
SOM = SOM((32, 32), data, 2000)
SOM.fit()
SOM.save_weights(config.SOM_weights_path)

SOM.load_weights(config.SOM_weights_path)
SOM.moveresp()
SOM.view_weights()