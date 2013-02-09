#!/usr/bin/env python
# encoding: utf-8

from datagenerator import *
from randomnetwork import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *

import numpy as np
import pylab as plt

# import scipy.special as scsp
# import scipy.optimize as spopt
# import scipy.interpolate as spint
# import os.path
# from matplotlib.patches import Ellipse
import glob
import re


def do_plots_networks():
    if True:
        # Plot conj coverage for abstract
        R = 2
        M = 400
        MM = int(np.floor(M ** 0.5) ** 2.)
        plt.rcParams['font.size'] = 17
        # rn = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(0.45, 0.001), ratio_moments=(1.0, 0.001))
        rn = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(0.5, 0.2), ratio_moments=(1.0, 0.2))
        # ax = rn.plot_coverage_feature_space(nb_stddev=0.7)
        ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.3, specific_neurons=np.arange(MM))
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

    if True:
        # Plt feat coverage for abstract
        M = 50
        rn = RandomFactorialNetwork.create_full_features(M, R=R, scale=0.5, ratio=25, nb_feature_centers=1)
        # ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.2)
        ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.3, facecolor='r')
        boxsize = 4.4
        ax.set_xlim(-boxsize, boxsize)
        ax.set_ylim(-boxsize, boxsize)
        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

    if True:
        # Plot mixed coverage

        # %run experimentlauncher.py --code_type mixed --inference_method none --rc_scale 1.9 --rc_scale2 0.1 --feat_ratio -150
        ax = random_network.plot_coverage_feature_space(alpha_ellipses=0.2, specific_neurons=np.arange(100, 140, 2), facecolor='r', height_factor=2.5)
        ax = random_network.plot_coverage_feature_space(alpha_ellipses=0.2, specific_neurons=np.arange(220, 260, 2), facecolor='r', width_factor=2.5, ax=ax)
        ax = random_network.plot_coverage_feature_space(alpha_ellipses=0.2, specific_neurons=np.arange(60), facecolor='b', ax=ax)
        
        boxsize = np.pi*1.1
        ax.set_xlim(-boxsize, boxsize)
        ax.set_ylim(-boxsize, boxsize)
        

    plt.show()





if __name__ == '__main__':
     
    results = plot_probabilities_mixtures()    

    plt.show()










