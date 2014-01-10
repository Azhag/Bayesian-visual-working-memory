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

import cPickle as pickle

import utils
import re
import hashlib

import em_circularmixture_allitems_uniquekappa

# # Commit @473b36f +


def plots_3dvolume_hierarchical_M_Mlayerone(data_pbs, generator_module=None):
    '''
        Reload 3D volume runs from PBS and plot them

    '''

    #### SETUP
    #
    savefigs = True
    savedata = False  # warning, huge file created...

    plots_pcolors = False
    plots_singleaxe = False
    plots_multipleaxes = True
    plots_multipleaxes_emfits = True

    load_fit_mixture_model = True

    # caching_emfit_filename = None
    caching_emfit_filename = os.path.join(generator_module.pbs_submission_infos['simul_out_dir'], 'outputs', 'cache_emfit.pickle')

    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    results_precision_constant_M_Mlower = np.squeeze(utils.nanmean(data_pbs.dict_arrays['results_precision_M_T']['results'], axis=-1))
    results_precision_constant_M_Mlower_std = np.squeeze(utils.nanstd(data_pbs.dict_arrays['results_precision_M_T']['results'], axis=-1))
    results_responses = np.squeeze(data_pbs.dict_arrays['result_responses']['results'])
    results_targets = np.squeeze(data_pbs.dict_arrays['result_targets']['results'])
    results_nontargets = np.squeeze(data_pbs.dict_arrays['result_nontargets']['results'])
    # results_emfits_M_T = np.squeeze(data_pbs.dict_arrays['results_emfits_M_T']['results'])

    M_space = data_pbs.loaded_data['parameters_uniques']['M']
    M_layer_one_space = data_pbs.loaded_data['parameters_uniques']['M_layer_one']
    ratio_MMlower_space = M_space/generator_module.filtering_function_parameters['target_M_total']

    filtering_indices = (np.arange(M_space.size), np.arange(-M_layer_one_space.size, 0)[::-1])

    T = results_precision_constant_M_Mlower.shape[-1]
    T_space = np.arange(T)
    N = results_nontargets.shape[-2]
    num_repetitions = data_pbs.loaded_data['args_list'][0]['num_repetitions']

    # Num_repetitions sometimes is in wrong position...
    if results_responses.shape[-2] == num_repetitions:
        results_responses.shape = (M_space.size, M_layer_one_space.size, T, num_repetitions*N)
        results_targets.shape = (M_space.size, M_layer_one_space.size, T, num_repetitions*N)
        results_nontargets.shape = (M_space.size, M_layer_one_space.size, T, num_repetitions*N, T-1)

    print M_space
    print M_layer_one_space
    print results_precision_constant_M_Mlower.shape
    # print results_precision_constant_M_Mlower

    # Reorder results_nontargets if multiple runs concatenated
    if data_pbs.loaded_data['nb_datasets_per_parameters'] > 1:
        print "Reorder results_nontargets"

        nontargets_shape = list(results_nontargets.shape)
        nontargets_shape[-2] *= data_pbs.loaded_data['nb_datasets_per_parameters']
        nontargets_shape[-1] /= data_pbs.loaded_data['nb_datasets_per_parameters']
        results_nontargets_tmp = np.empty(nontargets_shape)
        for run_i in xrange(data_pbs.loaded_data['nb_datasets_per_parameters']):
            print "reorder", run_i
            results_nontargets_tmp[..., N*run_i:N*(run_i+1), :] = results_nontargets[..., (T-1)*run_i:(T-1)*(run_i+1)]

        del results_nontargets
        results_nontargets = results_nontargets_tmp

    T = results_precision_constant_M_Mlower.shape[-1]
    results_precision_filtered = results_precision_constant_M_Mlower[filtering_indices]
    del results_precision_constant_M_Mlower
    results_precision_filtered_std = results_precision_constant_M_Mlower_std[filtering_indices]
    del results_precision_constant_M_Mlower_std
    results_responses_filtered = results_responses[filtering_indices]
    del results_responses
    results_targets_filtered = results_targets[filtering_indices]
    del results_targets
    results_nontargets_filtered = results_nontargets[filtering_indices]
    del results_nontargets
    # results_emfits_M_T_filtered = results_emfits_M_T[filtering_indices]
    # del results_emfits_M_T

    results_precision_filtered_smoothed = np.apply_along_axis(smooth, 0, results_precision_filtered, *(10, 'bartlett'))

    if load_fit_mixture_model:
        # Fit the mixture model on the samples

        if caching_emfit_filename is not None:
            if os.path.exists(caching_emfit_filename):
                # Got file, open it and try to use its contents
                try:
                    with open(caching_emfit_filename, 'r') as file_in:
                        # Load and assign values
                        cached_data = pickle.load(file_in)
                        result_emfits_filtered = cached_data['result_emfits_filtered']
                        results_responses_sha1_loaded = cached_data.get('results_responses_sha1', '')
                        # Check that the sha1 is the same, if not recompute!
                        if results_responses_sha1_loaded == hashlib.sha1(results_responses_filtered).hexdigest():
                            print "Loading from cache file %s" % caching_emfit_filename
                            load_fit_mixture_model = False
                        else:
                            print "Tried loading from cache file %s, but data changed, recomputing..." % caching_emfit_filename
                            load_fit_mixture_model = True

                except IOError:
                    print "Error while loading ", caching_emfit_filename, "falling back to computing the EM fits"
                    load_fit_mixture_model = False


        if load_fit_mixture_model:

            result_emfits_filtered = np.nan*np.empty((ratio_MMlower_space.size, T, 5))

            # Fit EM model
            print "fitting EM model"
            for ratio_MMlower_i, ratio_MMlower in enumerate(ratio_MMlower_space):
                for T_i in T_space:
                    if np.any(~np.isnan(results_responses_filtered[ratio_MMlower_i, T_i])):
                        print "ratio MM, T:", ratio_MMlower, T_i+1
                        curr_em_fits = em_circularmixture_allitems_uniquekappa.fit(results_responses_filtered[ratio_MMlower_i, T_i], results_targets_filtered[ratio_MMlower_i, T_i], results_nontargets_filtered[ratio_MMlower_i, T_i, :, :T_i])

                        curr_em_fits['mixt_nontargets_sum'] = np.sum(curr_em_fits['mixt_nontargets'])
                        result_emfits_filtered[ratio_MMlower_i, T_i] = [curr_em_fits[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL')]

            # Save everything to a file, for faster later plotting
            if caching_emfit_filename is not None:
                try:
                    with open(caching_emfit_filename, 'w') as filecache_out:
                        results_responses_sha1 = hashlib.sha1(results_responses_filtered).hexdigest()
                        data_emfit = dict(result_emfits_filtered=result_emfits_filtered, results_responses_sha1=results_responses_sha1)

                        pickle.dump(data_emfit, filecache_out, protocol=2)
                        print "cache file %s written" % caching_emfit_filename
                except IOError:
                    print "Error writing out to caching file ", caching_emfit_filename


    if plots_multipleaxes:

        # Plot precisions with standard deviation around
        f, axes = plt.subplots(nrows=T, ncols=1, sharex=True, figsize=(10, 12))
        # all_lines_bis = []
        for i, max_ind in enumerate(np.argmax(results_precision_filtered, axis=0)):
            utils.plot_mean_std_area(ratio_MMlower_space, results_precision_filtered[:, i], results_precision_filtered_std[:, i], ax_handle=axes[i], linewidth=2) #, color=all_lines[i].get_color())
            # curr_lines = axes[i].plot(ratio_MMlower_space, results_precision_filtered[:, i], linewidth=2, color=all_lines[i].get_color())
            axes[i].grid()
            axes[i].set_xticks(np.linspace(0., 1.0, 5))
            axes[i].set_xlim((0.0, 1.0))
            # axes[i].set_yticks([])
            axes[i].set_ylim((np.min(results_precision_filtered[:, i]), results_precision_filtered[max_ind, i]*1.1))
            axes[i].locator_params(axis='y', tight=True, nbins=4)

            # all_lines_bis.extend(curr_lines)

        f.subplots_adjust(right=0.75)
        # plt.figlegend(all_lines_bis, ['%d item' % i + 's'*(i>1) for i in xrange(1, T+1)], loc='right', bbox_to_anchor=(1.0, 0.5))

        if savefigs:
            dataio.save_current_figure('results_subplots_1dnorm_{label}_global_{unique_id}.pdf')

    if plots_multipleaxes_emfits:

        f, axes = plt.subplots(nrows=T, ncols=1, sharex=True, figsize=(10, 12))
        all_lines_bis = []

        for i, max_ind in enumerate(np.nanargmax(result_emfits_filtered[..., 0], axis=0)):
            # Plot Target mixture
            utils.plot_mean_std_area(ratio_MMlower_space, result_emfits_filtered[:, i, 1:4], 0*result_emfits_filtered[:, i, 1:4], ax_handle=axes[i], linewidth=2) #, color=all_lines[i].get_color())
            # curr_lines = axes[i].plot(ratio_MMlower_space, results_precision_filtered[:, i], linewidth=2, color=all_lines[i].get_color())
            axes[i].grid()
            axes[i].set_xticks(np.linspace(0., 1.0, 5))
            axes[i].set_xlim((0.0, 1.0))
            # axes[i].set_yticks([])
            # axes[i].set_ylim((np.min(result_emfits_filtered[:, i, 0]), result_emfits_filtered[max_ind, i, 0]*1.1))
            axes[i].set_ylim((0.0, 1.0))
            axes[i].locator_params(axis='y', tight=True, nbins=4)
            # all_lines_bis.extend(curr_lines)

        if savefigs:
            dataio.save_current_figure('results_subplots_emtarget_{label}_global_{unique_id}.pdf')

        f, axes = plt.subplots(nrows=T, ncols=1, sharex=True, figsize=(10, 12))

        for i, max_ind in enumerate(np.nanargmax(result_emfits_filtered[..., 0], axis=0)):

            # Plot kappa mixture
            utils.plot_mean_std_area(ratio_MMlower_space, result_emfits_filtered[:, i, 0], 0*result_emfits_filtered[:, i, 0], ax_handle=axes[i], linewidth=2) #, color=all_lines[i].get_color())
            # curr_lines = axes[i].plot(ratio_MMlower_space, results_precision_filtered[:, i], linewidth=2, color=all_lines[i].get_color())
            axes[i].grid()
            axes[i].set_xticks(np.linspace(0., 1.0, 5))
            axes[i].set_xlim((0.0, 1.0))
            # axes[i].set_yticks([])
            # axes[i].set_ylim((np.min(result_emfits_filtered[:, i, 0]), result_emfits_filtered[max_ind, i, 0]*1.1))
            # axes[i].set_ylim((0.0, 1.0))
            axes[i].locator_params(axis='y', tight=True, nbins=4)
            # all_lines_bis.extend(curr_lines)

        # f.subplots_adjust(right=0.75)
        # plt.figlegend(all_lines_bis, ['%d item' % i + 's'*(i>1) for i in xrange(1, T+1)], loc='right', bbox_to_anchor=(1.0, 0.5))

        if savefigs:
            dataio.save_current_figure('results_subplots_emkappa_{label}_global_{unique_id}.pdf')

    variables_to_save = []

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)

    dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='hierarchicalrandomnetwork_characterisation')

    plt.show()

    return locals()



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
                     concatenate_multiple_datasets=False
                     )



if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio', 'post_processing_outputs', 'fit_exp']

    if 'variables_to_save' in experiment_launcher.all_vars:
        # Also reinstantiate the variables we saved
        variables_to_reinstantiate.extend(experiment_launcher.all_vars['variables_to_save'])

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]

    for var_reinst in post_processing_outputs:
        vars()[var_reinst] = post_processing_outputs[var_reinst]



