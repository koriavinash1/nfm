

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from SOM import SOM

Gstat = GaussianStatistics()

# NFM 2D implementation

class CoupledNFM(object):
    def __init__(self, size, 
                    dt   = 0.01,
                    sdt  = 0.1,
                    Iext = 1,
                    exe_rad    = 2, 
                    inhb_rad   = 4, 
                    exe_ampli  = 2., 
                    inhb_ampli = 1.):
        """

        """
        self.eRad  = exe_rad
        self.iRad  = inhb_rad

        # gaussian statistics...
        self.eA  = exe_ampli
        self.iA  = inhb_ampli
        # neural field parameters
        self.X     = 0.05*abs(np.random.randn(size[0], size[1]))
        self.Y     = 0.05*abs(np.random.randn(size[0], size[1]))
        self.Iext  = Iext
        self.dt    = dt
        self.sdt   = sdt

        # lateral weights
        gauss1, gauss2   = Gstat.DOG(self.iRad, self.eRad, self.iA, self.eA)
        self.kernel      = gauss1 + gauss2
        

    def Normalize(self, mat, type_='MinMax'):
        """

        """

        if type_ == 'L1':
            mat = mat/ (np.sum(np.abs(mat),axis=(0, 1)) + np.sum(np.abs(mat),axis=(2,3)))[:, :, None, None]
        elif type_ == 'MinMax':
            mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
        elif type_ == 'Zscore':
            mat = (mat - np.mean(mat))/ np.var(mat)**0.5
        elif type_ == 'tanh':
            mat = np.tanh(mat)
        else:
            raise ValueError("Invalid Type found")
        return mat


    def lateralDynamicsFHN(self, 
                            verbose = True, 
                            alpha = 0.5, 
                            beta  = 0.1, 
                            gamma = 0.1, 
                            tau   = 0.08, 
                            ci    = -0.2):
        """

        """
        temp_inh = self.Normalize(signal.convolve2d(self.X, self.kernel, mode = 'same') + self.Z)
        I = alpha*self.Iext + beta*temp_inh + ci
        I[I < 0] = 0

        fv   = self.X*(alpha - self.X)*(self.X - 1)
        xdot = (fv - self.Y + I)/tau
        ydot = beta*self.X - gamma*self.Y

        self.X = self.X + xdot*self.dt
        self.Y = self.Y + ydot*self.dt

        if verbose: print ('max I: {}'.format(np.max(I)) + '  min: {}'.format(np.max(I)))


    def lateralDynamicsHopf(self,
                            verbose = True,
                            alpha   = 0.5,
                            beta    = 0.1,
                            gamma   = 0.1):
        """

        """
        temp_inh = self.Normalize(signal.convolve2d(self.X, self.kernel, mode = 'same') + self.Z)
        I = beta*temp_inh + gamma*self.Iext
        I[I < 0] = 0

        xdot = -self.X + alpha*self.Y + self.Y*self.X**2
        ydot = I - alpha*self.Y - self.Y*self.X**2

        self.X = self.X + xdot*self.dt
        self.Y = self.Y + ydot*self.dt

        if verbose: print ('max I: {}'.format(np.max(I)) + '  min: {}'.format(np.max(I)))


    def lateralDynamicsVPole(self,
                            verbose = True, 
                            alpha = 0.5, 
                            beta  = 0.1, 
                            mu    = 0.08, 
                            ci    = -0.2):
        """

        """
        temp_inh = self.Normalize(signal.convolve2d(self.X, self.kernel, mode = 'same') + self.Z)
        I = alpha*self.Iext + beta*temp_inh + ci
        I[I < 0] = 0

        fv   = self.X*(alpha - self.X)*(self.X - 1)
        xdot = (fv - self.Y + I)*mu
        ydot = self.X / tau

        self.X = self.X + xdot*self.dt
        self.Y = self.Y + ydot*self.dt

        if verbose: print ('max I: {}'.format(np.max(I)) + '  min: {}'.format(np.max(I)))


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