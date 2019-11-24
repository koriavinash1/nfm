

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from configure import Config
from SOM import SOM

Gstat = GaussianStatistics()
config= Config()

# NFM 2D implementation

class NFM(object):
    def __init__(self, size, exe_rad = 2,inhb_rad = 4, exe_ampli = 2., inhb_ampli = 1., aff = 1):
        self.eRad  = exe_rad
        self.iRad  = inhb_rad
        # gaussian statistics...
        self.eA  = exe_ampli
        self.iA  = inhb_ampli
        # neural field parameters
        self.Z     = 0.05*abs(np.random.randn(size[0], size[1]))
        self.W     = 0.05*abs(np.random.randn(size[0], size[1]))
        self.aff   = aff
        # lateral weights
        gauss1, gauss2   = Gstat.DOG(self.iRad, self.eRad, self.iA, self.eA)
        # Gstat.Visualize(gauss1+gauss2, _type = '2d')
        self.kernel= gauss1 + gauss2


    def Normalize(self, mat):
        # mat = mat/ np.max(mat)
        # mat = mat / np.sum(abs(mat))
        mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
        # mat = (mat - np.mean(mat))/ np.var(mat)**0.5
        return mat

    def lateralDynamics(self, verbose = True, ci=config.ci):
        temp_eff = self.aff
        temp_inh = self.Normalize(signal.convolve2d(self.Z, self.kernel, mode = 'same') + self.Z)
        I = config.ai*temp_eff + config.bi*temp_inh + ci
        I[I < 0] = 0

        v1 = self.Z
        w1 = self.W

        fv   = v1*(config.a - v1)*(v1 - 1)
        vdot = (fv - w1 + I)/config.freq_ctrl
        wdot = config.b*v1 - config.gamma*w1

        v1 = v1 + vdot*config.dt
        w1 = w1 + wdot*config.dt

        self.Z = v1
        self.W = w1

        # self.Z = self.Normalize(self.Z)
        # self.W = self.Normalize(self.W)
        if verbose: print 'max I: {}'.format(np.max(I)) + '  min: {}'.format(np.max(I))

    def fanoFactor(self, sig):
        return np.var(sig)/np.mean(sig)

    def sanity_check(self, I, Zold, Wold, dt=0.01):
        """
            To find oscillatory regimes for single neuron
        """
        v1 = Zold
        w1 = Wold

        fv = v1*(0.5 - v1)*(v1 - 1)
        vdot = float(fv - w1 + I)/config.freq_ctrl
        wdot = 0.1*v1 - 0.1*w1

        v1 = v1 + vdot*dt
        w1 = w1 + wdot*dt

        # v1[v1 < 0] = 0
        return v1, w1
