'''
    Small class system to simplify the process of loading Experimental datasets
'''

import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# import matplotlib.patches as plt_patches
# import matplotlib.gridspec as plt_grid
import os
import os.path
import cPickle as pickle
# import bottleneck as bn
import em_circularmixture
import em_circularmixture_allitems_uniquekappa
import pandas as pd

import dataio as DataIO

import utils

from experimentalloader import ExperimentalLoader

class ExperimentalLoaderGorgo11(ExperimentalLoader):
    """
        Loader for Gorgo11, both simultaneous and sequential
    """
    def __init__(self, dataset_description):
        super(self.__class__, self).__init__(dataset_description)


    def preprocess(self, parameters):
        raise NotImplementedError('should be overridden')



