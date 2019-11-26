import numpy as np
import os

class Config():
	"""
	"""
	# FHN params
	a        = 0.5
	b        = 0.1
	gamma    = 0.1
	freq_ctrl= 0.08
	dt       = 0.01
	sdt      = 0.1
	T        = 50
	transtime= 150
	nSimulations = 20
	nOrientations = 4
	nPrototype   = 3

	# regime params
	# >> note:
	#   >>  for ai = 0.2, bi = 0.05
	#   >>  ci:- oscillatory = (0.1, 0.5)
	#   >>  ci: excitatory   = (-0.2, 0)
	#

	ai = 0.2
	bi = 0.05
	ci = -0.2
	ci_range = np.array(range(-20, 50, 4))/100.0

	# Gaussian params
	N    = 32
	iRad = 4
	eRad = 2
	iA   = 1
	eA   = 2

	# for orientation bar
	mu  = np.array([5., 5.])
	std = np.array([[20., -19.8], [-19.8, 20]])

	savepath = './logs'
	SOM_weights_path = './SOM_weights.npy'
	Orientation_path = './Orientation_bars.npy'

	# if not os.path.exists(savepath):
	# 	os.mkdir(savepath)
