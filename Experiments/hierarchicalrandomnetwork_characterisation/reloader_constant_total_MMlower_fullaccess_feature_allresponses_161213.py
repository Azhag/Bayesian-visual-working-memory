"""
    ExperimentDescriptor for hierarchicalrandomnetwork parameter sweep.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from experimentlauncher import *
from dataio import *
from smooth import *
import inspect

import utils
import re

# # Commit @473b36f +


def plots_3dvolume_hierarchical_M_Mlayerone(data_pbs, generator_module=None):
    '''
        Reload 3D volume runs from PBS and plot them

    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plots_pcolors = True
    plots_singleaxe = True
    plots_multipleaxes = True

    # caching_bootstrap_filename = None
    # caching_bootstrap_filename = os.path.join(generator_module.pbs_submission_infos['simul_out_dir'], 'outputs', 'cache_bootstrap_misbinding_mixed.pickle')

    plt.rcParams['font.size'] = 16
    #
    #### /SETUP


    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    results_precision_constant_M_Mlower = np.squeeze(utils.nanmean(data_pbs.dict_arrays['results_precision_M_T']['results'], axis=-1))
    results_precision_constant_M_Mlower_std = np.squeeze(utils.nanstd(data_pbs.dict_arrays['results_precision_M_T']['results'], axis=-1))

    result_responses = np.squeeze(data_pbs.dict_arrays['result_responses']['results'])
    result_targets = np.squeeze(data_pbs.dict_arrays['result_targets']['results'])
    result_nontargets = np.squeeze(data_pbs.dict_arrays['result_nontargets']['results'])

    M_space = data_pbs.loaded_data['parameters_uniques']['M']
    M_layer_one_space = data_pbs.loaded_data['parameters_uniques']['M_layer_one']

    print M_space
    print M_layer_one_space
    print results_precision_constant_M_Mlower.shape
    # print results_precision_constant_M_Mlower

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    T = results_precision_constant_M_Mlower.shape[-1]
    results_filtered = results_precision_constant_M_Mlower[np.arange(M_space.size), np.arange(-M_layer_one_space.size, 0)[::-1]]
    results_filtered_std = results_precision_constant_M_Mlower_std[np.arange(M_space.size), np.arange(-M_layer_one_space.size, 0)[::-1]]


    if plots_pcolors:
        results_filtered_smoothed = np.apply_along_axis(smooth, 0, results_filtered, *(10, 'bartlett'))

        ratio_MMlower = M_space/generator_module.filtering_function_parameters['target_M_total']
        utils.pcolor_2d_data(results_filtered, log_scale=True, x=ratio_MMlower, y=np.arange(1, T+1), xlabel="$\\frac{M}{M+M_{layer one}}$", ylabel='$T$', ticks_interpolate=10)
        plt.plot(np.argmax(results_filtered, axis=0), np.arange(results_filtered.shape[-1]), 'ko', markersize=10)

        if savefigs:
            dataio.save_current_figure('results_2dlog_{label}_global_{unique_id}.pdf')

        utils.pcolor_2d_data(results_filtered/np.max(results_filtered, axis=0), x=ratio_MMlower, y=np.arange(1, T+1), xlabel="$\\frac{M}{M+M_{layer one}}$", ylabel='$T$', ticks_interpolate=10)
        plt.plot(np.argmax(results_filtered, axis=0), np.arange(results_filtered.shape[-1]), 'ko', markersize=10)

        if savefigs:
            dataio.save_current_figure('results_2dnorm_{label}_global_{unique_id}.pdf')


        utils.pcolor_2d_data(results_filtered_smoothed/np.max(results_filtered_smoothed, axis=0), x=ratio_MMlower, y=np.arange(1, T+1), xlabel="$\\frac{M}{M+M_{layer one}}$", ylabel='$T$', ticks_interpolate=10)
        plt.plot(np.argmax(results_filtered_smoothed, axis=0), np.arange(results_filtered_smoothed.shape[-1]), 'ko', markersize=10)

        if savefigs:
            dataio.save_current_figure('results_2dsmoothnorm_{label}_global_{unique_id}.pdf')

    if plots_singleaxe:

        plt.figure()
        plt.plot(ratio_MMlower, results_filtered_smoothed/np.max(results_filtered_smoothed, axis=0), linewidth=2)
        plt.plot(ratio_MMlower[np.argmax(results_filtered_smoothed, axis=0)], np.ones(results_filtered_smoothed.shape[-1]), 'ro', markersize=10)
        plt.grid()
        plt.ylim((0., 1.1))
        plt.subplots_adjust(right=0.8)
        plt.legend(['%d item' % i + 's'*(i>1) for i in xrange(1, T+1)], loc='center right', bbox_to_anchor=(1.3, 0.5))
        plt.xticks(np.linspace(0, 1.0, 5))

        if savefigs:
            dataio.save_current_figure('results_1dsmoothnormsame_{label}_global_{unique_id}.pdf')

        plt.figure()
        moved_results_filtered_smoothed = 1.2*np.arange(results_filtered_smoothed.shape[-1]) + results_filtered_smoothed/np.max(results_filtered_smoothed, axis=0)
        all_lines = []
        for i, max_i in enumerate(np.argmax(results_filtered_smoothed, axis=0)):
            curr_lines = plt.plot(ratio_MMlower, moved_results_filtered_smoothed[:, i], linewidth=2)
            plt.plot(ratio_MMlower[max_i], moved_results_filtered_smoothed[max_i, i], 'o', markersize=10, color=curr_lines[0].get_color())
            all_lines.extend(curr_lines)

        plt.plot(np.linspace(0.0, 1.0, 100), np.outer(np.ones(100), 1.2*np.arange(1, results_filtered_smoothed.shape[-1])), 'k:')
        plt.grid()
        plt.legend(all_lines, ['%d item' % i + 's'*(i>1) for i in xrange(1, T+1)], loc='best')
        plt.ylim((0., moved_results_filtered_smoothed.max()*1.05))
        plt.yticks([])
        plt.xticks(np.linspace(0, 1.0, 5))

        if savefigs:
            dataio.save_current_figure('results_1dsmoothnorm_{label}_global_{unique_id}.pdf')

    if plots_multipleaxes:
        # Plot smooth precisions, all T on multiple subplots.
        all_lines_bis = []
        f, axes = plt.subplots(nrows=T, ncols=1, sharex=True, figsize=(10, 12))
        for i, max_ind in enumerate(np.argmax(results_filtered_smoothed, axis=0)):
            curr_lines = axes[i].plot(ratio_MMlower, results_filtered_smoothed[:, i], linewidth=2, color=all_lines[i].get_color())
            axes[i].plot(ratio_MMlower[max_ind], results_filtered_smoothed[max_ind, i], 'o', markersize=10, color=curr_lines[0].get_color())
            axes[i].grid()
            axes[i].set_xticks(np.linspace(0., 1.0, 5))
            axes[i].set_xlim((0.0, 1.0))
            # axes[i].set_yticks([])
            axes[i].set_ylim((np.min(results_filtered_smoothed[:, i]), results_filtered_smoothed[max_ind, i]*1.1))
            axes[i].locator_params(axis='y', tight=True, nbins=4)

            all_lines_bis.extend(curr_lines)

        f.subplots_adjust(right=0.75)
        plt.figlegend(all_lines_bis, ['%d item' % i + 's'*(i>1) for i in xrange(1, T+1)], loc='right', bbox_to_anchor=(1.0, 0.5))
        # f.tight_layout()

        if savefigs:
            dataio.save_current_figure('results_subplots_1dsmoothnorm_{label}_global_{unique_id}.pdf')


        # Plot precisions with standard deviation around
        f, axes = plt.subplots(nrows=T, ncols=1, sharex=True, figsize=(10, 12))
        # all_lines_bis = []
        for i, max_ind in enumerate(np.argmax(results_filtered, axis=0)):
            utils.plot_mean_std_area(ratio_MMlower, results_filtered[:, i], results_filtered_std[:, i], ax_handle=axes[i], linewidth=2, color=all_lines[i].get_color())
            # curr_lines = axes[i].plot(ratio_MMlower, results_filtered[:, i], linewidth=2, color=all_lines[i].get_color())
            axes[i].grid()
            axes[i].set_xticks(np.linspace(0., 1.0, 5))
            axes[i].set_xlim((0.0, 1.0))
            # axes[i].set_yticks([])
            axes[i].set_ylim((np.min(results_filtered[:, i]), results_filtered[max_ind, i]*1.1))
            axes[i].locator_params(axis='y', tight=True, nbins=4)

            # all_lines_bis.extend(curr_lines)

        f.subplots_adjust(right=0.75)
        plt.figlegend(all_lines_bis, ['%d item' % i + 's'*(i>1) for i in xrange(1, T+1)], loc='right', bbox_to_anchor=(1.0, 0.5))

        if savefigs:
            dataio.save_current_figure('results_subplots_1dnorm_{label}_global_{unique_id}.pdf')


    variables_to_save = []

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='hierarchicalrandomnetwork_characterisation')

    plt.show()




this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]
# generator_script = 'generator_specific_stimuli_mixed_fixedemfit_otherrange_201113.py'

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Hierarchical network. Assume we want to allocate a fixed number of neurons between the two layers. Do that by constraining the sum of M and M_layer_one to be some constant. Corrected logic so that whole population is accessible now. Outputs all responses for later fits. Sampling.',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['M', 'M_layer_one'],
                     variables_to_load=['results_precision_M_T', 'result_responses', 'result_targets', 'result_nontargets'],
                     variables_description=['Precision volume'],
                     post_processing=plots_3dvolume_hierarchical_M_Mlayerone,
                     save_output_filename='3dvolume_hierarchical_constant_M_Mlayerone_feature',
                     concatenate_multiple_datasets=True
                     )




if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio']

    if 'variables_to_save' in experiment_launcher.all_vars:
        # Also reinstantiate the variables we saved
        variables_to_reinstantiate.extend(experiment_launcher.all_vars['variables_to_save'])

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]

