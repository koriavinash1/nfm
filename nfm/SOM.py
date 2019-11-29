import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from tqdm import tqdm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from datetime import datetime

class SOM():
    """
    """
    def __init__(self, output_size, input_shape, epochs=1, learning_rate=1e-3, rad=4, sig=1):
        self.epochs  = epochs
        self.output_size = output_size
        self.input_num   = np.prod(input_shape)
        self.nrad    = rad 
        self.sig     = sig
        self.alpha   = learning_rate
        self.weights = np.random.rand(self.output_size[0], self.output_size[1], self.input_num)

    def Normalize(self, mat):
        # mat = mat/ np.max(mat)
        # mat = mat / np.sum(abs(mat))
        mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
        # mat = (mat - np.mean(mat))/ np.var(mat)**0.5
        return mat


    def nstnbridx(self, row_data):
        """
        """
        dist_neurons = np.linalg.norm((row_data[None, None, ...] -self.weights), axis=2)
        x_, y_ = np.where(dist_neurons == np.min(dist_neurons)) # Best matching unit (BMU)
        return np.array([x_[0], y_[0]])


    def nbridxdis(self, index_bmu):

        nbrind = np.zeros(((2*self.nrad + 1)**2, 2))
        xindx  = np.linspace(0, 2*self.nrad, 2*self.nrad + 1).astype("int")
        yindx  = np.linspace(0, 2*self.nrad, 2*self.nrad + 1).astype("int")
        xindx, yindx = np.meshgrid(xindx, yindx)
        xindx, yindx = xindx.flatten('F'), yindx.flatten('F')

        idx = xindx*(2*self.nrad + 1) + yindx
        nbrind[idx,:] = np.concatenate([(xindx - self.nrad)[..., None], 
                                        (yindx- self.nrad)[..., None]], axis=1) + index_bmu.T
        nbrind = nbrind[np.where((nbrind[:,0] >= 0) * (nbrind[:,1] >= 0))]
        nbrind = nbrind[np.where((nbrind[:,0] < self.output_size[0]) * (nbrind[:,1] < self.output_size[1]))]

        diff = nbrind - index_bmu.T
        nbrdist = np.linalg.norm(diff, axis=1)**2

        return nbrind, nbrdist


    def fit(self, data):
        """
        """
        iterations = len(data)*self.epochs
        for itter in tqdm(range(iterations)):
            initial_dis = float("inf")
            row_index = np.random.randint(len(data))
            learning_rate = self.alpha*np.exp(-itter/iterations)
            row_data = data[row_index]
            bmu_idx  = self.nstnbridx(row_data)
            nbrind, nbrdist = self.nbridxdis(bmu_idx)
            mx, _ = nbrind.shape
            for i in range(mx):
                idx = nbrind[i,:]
                wt  = self.weights[int(idx[0]), int(idx[1]), :]
                diff = row_data - wt
                dis  = nbrdist[i]/self.sig **2
                delta = learning_rate*np.exp(-dis)*diff
                self.weights[int(idx[0]), int(idx[1]), :] = delta + wt
        print ("SOM Training done!!...")
        pass


    def response(self, X, wt):
        """
        """
        x = X.flatten('F')
        assert len(x) == wt.shape[2]

        diff = 1.0* (wt - x[None, None, ...])
        dis  = np.linalg.norm(diff, axis=2)**2.0
        Y    = np.exp(-1.*dis / self.sig**2)

        return self.Normalize(Y)


    def view_weights(self, path):
        shape = self.weights.shape
        m = int(np.sqrt(shape[2]))
        img = np.zeros((shape[0]*m, shape[1]*m))
        for i in range(shape[0]):
            for j in range(shape[1]):
                img[i*m:(i+1)*m, j*m:(j+1)*m] = self.weights[i,j,:].reshape(m, m).T

        plt.imshow(img)
        plt.savefig(path)


    def moveresp(self, data, display=True):
        """
        """
        # plt.ion()
        for i, x in enumerate(data):
            N = np.sqrt(len(x))
            X = x.reshape(int(N), int(N), order='F')
            y = self.response(X, self.weights)
            now = datetime.now()
            if display:
                plt.subplot(1,2,1)
                plt.imshow(X)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplot(1,2,2)
                plt.imshow(y)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.pause(1)
                # plt.savefig(now.strftime("%H:%M:%S:%f")+'.png', bbox_inches='tight')
        plt.close()
        pass


    def load_weights(self, path):
        self.weights = np.load(path)
        return self.weights


    def save_weights(self, path):
        """
        """
        np.save(path, self.weights)
        return self.weights