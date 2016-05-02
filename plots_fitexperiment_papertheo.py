#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experimentlauncher import ExperimentLauncher
from dataio import DataIO
import plots_experimental_data
import em_circularmixture_parametrickappa


# import matplotlib.animation as plt_anim
# from mpl_toolkits.mplot3d import Axes3D

import re
import inspect
import imp

import utils
# import submitpbs
import load_experimental_data



class PlotsFitExperimentAllTPaperTheo(object):
    """
        This class does all plots as in our PLOSCB2015 paper, for new
        FitExperimentAllT instances.

    """
    def __init__(self, fit_experiment_allt,
                 do_distrib_errors_fig5=False,
                 do_memcurves_fig6=False,
                 do_mixtcurves_fig13=False,
                 do_distrib_errors_data_fig2=False,
                ):

        self.fit_exp = fit_experiment_allt
        self.experiment_id = self.fit_exp.experiment_id

        self.do_distrib_errors_fig5 = do_distrib_errors_fig5
        self.do_memcurves_fig6 = do_memcurves_fig6
        self.do_mixtcurves_fig13 = do_mixtcurves_fig13
        self.do_distrib_errors_data_fig2 = do_distrib_errors_data_fig2

        print "Doing Paper plots for %s. \nFig5 %d, Fig6 %d, Fig13 %d" % (
            self.experiment_id,
            self.do_distrib_errors_fig5,
            self.do_memcurves_fig6,
            self.do_mixtcurves_fig13
            )


    def do_plots(self):
        '''
            Do all plots for that FitExperimentAllT.

            These correspond to a particular experiment_id only, not multiple.
        '''
        if self.do_distrib_errors_data_fig2:
            self.plots_distrib_errors_data_fig2()
        if self.do_distrib_errors_fig5:
            self.plots_distrib_errors_fig5()
        if self.do_memcurves_fig6:
            self.plots_memcurves_fig6()
        if self.do_mixtcurves_fig13:
            self.plots_mixtcurves_fig13()



    def plots_distrib_errors_data_fig2(self):
        '''
            Same as plots_distrib_errors_fig5, but for the Experimental data

            Series of plots reproducing Fig 2 - Distribution of errors in human subjects
        '''
        f, axes = plt.subplots(nrows=2, ncols=self.fit_exp.T_space.size, figsize=(14, 10))

        for t_i, T in enumerate(self.fit_exp.T_space):
            print "DATA T %d" % T
            self.fit_exp.setup_experimental_stimuli_T(T)
            self.fit_exp.sampler.N = self.fit_exp.sampler.data_gen.N

            self.fit_exp.sampler.plot_histogram_errors(bins=41, ax_handle=axes[0, t_i], norm='density')
            axes[0, t_i].set_title('')
            axes[0, t_i].set_ylim((0, 2))

            if T > 1:
                self.fit_exp.sampler.plot_histogram_bias_nontarget(bins=41, ax_handle=axes[1, t_i], show_parameters=False)
                axes[1, t_i].set_title('')
                axes[1, t_i].set_ylim((0, 0.3))
            else:
                axes[1, t_i].axis('off')

        f.suptitle('Fig2 - Human distribution errors')
        return axes

    def plots_distrib_errors_fig5(self):
        '''
            Series of plots reproducing Fig 5 - Distribution of errors of the
            model

            Arranged as in the paper.
        '''

        f, axes = plt.subplots(nrows=2, ncols=self.fit_exp.T_space.size, figsize=(14, 10))

        for t_i, T in enumerate(self.fit_exp.T_space):
            print "MODEL T %d" % T
            self.fit_exp.setup_experimental_stimuli_T(T)
            # Only use small subset of N
            self.fit_exp.sampler.N = 1000

            self.fit_exp.sampler.force_sampling_round()

            self.fit_exp.sampler.plot_histogram_errors(bins=41, ax_handle=axes[0, t_i], norm='density')
            axes[0, t_i].set_title('')
            axes[0, t_i].set_ylim((0, 2))

            if T > 1:
                self.fit_exp.sampler.plot_histogram_bias_nontarget(bins=41, ax_handle=axes[1, t_i], show_parameters=False)
                axes[1, t_i].set_title('')
                axes[1, t_i].set_ylim((0, 0.3))
            else:
                axes[1, t_i].axis('off')

        f.suptitle('Fig5 - Model distribution errors')
        return axes

    def plots_memcurves_fig6(self):
        pass

    def plots_mixtcurves_fig13(self):
        pass




