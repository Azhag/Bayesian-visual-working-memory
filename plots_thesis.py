#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from experimentlauncher import *

from datagenerator import *
from hierarchicalrandomnetwork import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from dataio import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
from launchers import *
import load_experimental_data

import cPickle as pickle

import utils

plt.rcParams['font.size'] = 17

set_colormap = plt.cm.cubehelix


def do_plots_multiscale_random_populations():


    dataio = DataIO(label='thesis_population_code')
    plt.rcParams['font.size'] = 18

    if True:
        # Plot conj coverage for abstract

        scales_number = 4

        M = np.int((4**scales_number-1)/3.)

        plt.ion()

        rn = RandomFactorialNetwork.create_wavelet(M, scales_number=scales_number)

        ax = rn.plot_coverage_feature_space(alpha_ellipses=0.3, facecolor='g', lim_factor=1.1, nb_stddev=1.0)

        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

        ax.set_xlabel('')
        ax.set_ylabel('')

        # set_colormap()

        ax.get_figure().canvas.draw()

        dataio.save_current_figure('multiscale_populationcode_thesis_{label}_{unique_id}.pdf')

        # To be run in ETS_TOOLKIT=qt4 mayavi2
        if False:
            rn.plot_neuron_activity_3d(selected_neuron, precision=100, weight_deform=0.0, draw_colorbar=False)
            try:
                import mayavi.mlab as mplt

                mplt.view(0.0, 45.0, 45.0, [0., 0., 0.])
                mplt.draw()
            except:
                pass

    return locals()



if __name__ == '__main__':

    all_vars = {}

    all_vars = do_plots_multiscale_random_populations()


    if 'experiment_launcher' in all_vars:
        all_vars.update(all_vars['experiment_launcher'].all_vars)


    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio', 'experiment_launcher']

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in all_vars:
            vars()[var_reinst] = all_vars[var_reinst]

    plt.show()

