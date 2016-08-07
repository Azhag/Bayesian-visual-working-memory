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
from launchers import *
import load_experimental_data

import cPickle as pickle

import utils

plt.rcParams['font.size'] = 17

set_colormap = plt.cm.cubehelix


def do_plots_multiscale_random_populations():


    dataio = DataIO(label='thesis_population_code')
    plt.rcParams['font.size'] = 18

    if False:
        # Plot multiscale population code

        scales_number = 4
        M = np.int((4**scales_number-1)/3.)
        plt.ion()

        random_network = RandomFactorialNetwork.create_wavelet(M, scales_number=scales_number)

        ax = random_network.plot_coverage_feature_space(alpha_ellipses=0.3, facecolor='g', lim_factor=1.1, nb_stddev=1.0)

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
            random_network.plot_neuron_activity_3d(selected_neuron, precision=100, weight_deform=0.0, draw_colorbar=False)
            try:
                import mayavi.mlab as mplt

                mplt.view(0.0, 45.0, 45.0, [0., 0., 0.])
                mplt.draw()
            except:
                pass

    if True:
        # Plot random population code coverage

        M= 500
        kappa = RandomFactorialNetwork.compute_optimal_rcscale(M)/5.

        plt.ion()

        random_network = RandomFactorialNetwork.create_full_conjunctive(M, scale_moments=(kappa, 4.0), ratio_moments=(1.0, 3.0))
        random_network.assign_prefered_stimuli(tiling_type='random', reset=True)

        ax = random_network.plot_coverage_feature_space(alpha_ellipses=0.3, facecolor='m', lim_factor=1.1, nb_stddev=1.0)

        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

        ax.set_xlabel('')
        ax.set_ylabel('')

        # set_colormap()

        ax.get_figure().canvas.draw()

        dataio.save_current_figure('random_populationcode_thesis_{label}_{unique_id}.pdf')

    return locals()


def do_plot_best_fit_bays09():
    '''
        Takes best fitted parameters from CMA/ES runs and re-run plotting on those
    '''

    num_repetitions = 5
    best_parameters = dict(
        ratio_conj=9.16142946e-01,
        M=int(2.26351315e+02),
        sigmax=2.69370127e-01,
        sigma_output=1.66844398e-01
    )

    arguments_dict = dict(action_to_do='launcher_do_memory_curve_marginal_fi_withplots_live',
                          subaction='collect_responses',
                          collect_responses=None,
                          inference_method='sample',
                          N=300,
                          num_samples=200,
                          M=100,
                          T=6,
                          num_repetitions=num_repetitions,
                          renormalize_sigmax=None,
                          autoset_parameters=None,
                          session_id='cmaes_fitting_experiments_relaunchs',
                          label='thesis_bestfit_bays09_cmaes_Mratiosigmaxsigmaoutput',
                          code_type='mixed',
                          output_directory='./Figures/thesis/plot_best_fit_bays09',
                          ratio_conj=0.5,
                          sigmax=0.1,
                          sigmay=0.000001,
                          sigma_output=0.0,
                          selection_num_samples=1,
                          selection_method='last',
                          slice_width=0.07,
                          burn_samples=200,
                          enforce_min_distance=0.17,
                          specific_stimuli_random_centers=None,
                          stimuli_generation='random',
                          stimuli_generation_recall='random',
                          experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                          )

    arguments_dict.update(best_parameters)

    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    return locals()


def do_plot_launcher_check_fisher_fit_1obj_2016():
    '''
        This reproduces Figure 7 from the paper, but with the latest codebase.

        Need to check exactly which computation of the Fisher Information was used for that...
        (and if the cov(mu) term should have T or T-1)

        CHECK IPTHON NOTEBOOK
    '''

    # !!! CHECK IPYTHON NOTEBOOK ./Experiments/fisherinfo_singleitem/ !!!

    arguments_dict = dict(
        action_to_do='launcher_check_fisher_fit_1obj_2016',
        collect_responses=None,
        inference_method='sample',
        N=200,
        num_samples=100,
        M=196,
        T=1,
        num_repetitions=1,
        renormalize_sigmax=None,
        autoset_parameters=None,
        label='thesis_fisherinfo_fit_1obj_newcodebase',
        code_type='conj',
        output_directory='./Experiments/fisherinfo_singleitem/thesisrerun_do_plot_fisher_info_fit_1obj_newcodebase_050816',
        ratio_conj=1.,
        sigmax=0.1,
        sigmay=0.000001,
        sigma_output=0.0,
        selection_num_samples=1,
        selection_method='last',
        slice_width=0.07,
        burn_samples=100,
        enforce_min_distance=0.17,
        specific_stimuli_random_centers=None,
        stimuli_generation='constant',
        stimuli_generation_recall='constant',
        experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
    )
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    return locals()




if __name__ == '__main__':

    all_vars = {}

    # all_vars = do_plots_multiscale_random_populations()
    # all_vars = do_plot_best_fit_bays09()
    all_vars = do_plot_launcher_check_fisher_fit_1obj_2016()


    if 'experiment_launcher' in all_vars:
        all_vars.update(all_vars['experiment_launcher'].all_vars)

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio', 'experiment_launcher', 'arguments_dict', 'post_processing_outputs', 'fit_exp', 'all_outputs_data']

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in all_vars:
            vars()[var_reinst] = all_vars[var_reinst]

    plt.show()

