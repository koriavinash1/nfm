import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def display(Iext, title, fig,  _type = '3d'):
	"""

	"""
	X = np.arange(0, Iext.shape[0])
	Y = np.arange(0, Iext.shape[1])
	X, Y = np.meshgrid(X, Y)

	plt.clf()
	plt.ion()

	if _type == '3d':
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(X, Y, Iext/np.max(Iext) , cmap=cm.coolwarm,
		               linewidth=0, antialiased=False)
		ax.set_title(title)

		# Customize the z axis.
		ax.set_zlim(-1.01, 1.01)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
		fig.colorbar(surf, aspect=10)
	elif _type == '2d':
		plt.imshow(Iext)
	else:
		raise ValueError("Expected type argument ['3d', '2d'], but given {}".format(_type))

	plt.draw()
	plt.xlabel(title)
	plt.pause(0.005)
	pass