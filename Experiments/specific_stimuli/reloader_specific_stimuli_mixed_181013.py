"""
    ExperimentDescriptor for Fitting experiments in a mixed population code
"""

import os
import numpy as np
from experimentlauncher import *
from dataio import *
from utils import *
import re

import inspect

from utils import *

# Commit @2042319 +


def plots_specific_stimuli_mixed(data_pbs, generator_module=None):
    '''
        Reload and plot behaviour of mixed population code on specific Stimuli
        of 3 items.
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plot_per_min_dist_all = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_all_precisions_mean = nanmean(np.squeeze(data_pbs.dict_arrays['result_all_precisions']['results']), axis=-1)
    result_all_precisions_std = nanstd(np.squeeze(data_pbs.dict_arrays['result_all_precisions']['results']), axis=-1)
    result_em_fits_mean = nanmean(np.squeeze(data_pbs.dict_arrays['result_em_fits']['results']), axis=-1)
    result_em_fits_std = nanstd(np.squeeze(data_pbs.dict_arrays['result_em_fits']['results']), axis=-1)
    result_em_resp_all = np.squeeze(data_pbs.dict_arrays['result_em_resp']['results'])


    enforce_min_distance_space = data_pbs.loaded_data['parameters_uniques']['enforce_min_distance']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    ratio_space = data_pbs.loaded_data['datasets_list'][0]['ratio_space']

    print enforce_min_distance_space
    print sigmax_space
    print ratio_space
    print result_all_precisions_mean.shape, result_em_fits_mean.shape, result_em_resp_all.shape

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])


    if plot_per_min_dist_all:
        # Do one plot per min distance.
        for min_dist_i, min_dist in enumerate(enforce_min_distance_space):
            # Show log precision
            pcolor_2d_data(result_all_precisions_mean[min_dist_i].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='Precision, min_dist=%.3f' % min_dist)
            if savefigs:
                dataio.save_current_figure('precision_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)

            # Show log precision
            pcolor_2d_data(result_all_precisions_mean[min_dist_i].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='Precision, min_dist=%.3f' % min_dist, log_scale=True)
            if savefigs:
                dataio.save_current_figure('logprecision_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)


            # Plot estimated model precision
            pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., 0].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='EM precision, min_dist=%.3f' % min_dist, log_scale=False)
            if savefigs:
                dataio.save_current_figure('logemprecision_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)

            # Plot estimated Target, nontarget and random mixture components, in multiple subplots
            f, axes = plt.subplots(1, 3, figsize=(18, 6))
            plt.subplots_adjust(left=0.05, right=0.97, wspace = 0.3, bottom=0.15)
            pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., 1].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='Target, min_dist=%.3f' % min_dist, log_scale=False, ax_handle=axes[0], ticks_interpolate=5)
            pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., 2].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='Nontarget, min_dist=%.3f' % min_dist, log_scale=False, ax_handle=axes[1], ticks_interpolate=5)
            pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., 3].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='Random, min_dist=%.3f' % min_dist, log_scale=False, ax_handle=axes[2], ticks_interpolate=5)

            if savefigs:
                dataio.save_current_figure('em_mixtureprobs_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)

            # Plot Log-likelihood of Mixture model, sanity check
            pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., -1].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='EM loglik, min_dist=%.3f' % min_dist, log_scale=False)
            if savefigs:
                dataio.save_current_figure('em_loglik_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)

    #     # Plot the evolution of loglike as a function of sigmax, with std shown
    #     for ratio_conj_i, ratio_conj in enumerate(ratio_space):
    #         ax = plot_mean_std_area(sigmax_space, result_log_posterior_mean[ratio_conj_i], result_log_posterior_std[ratio_conj_i])

    #         ax.get_figure().canvas.draw()

    #         if savefigs:
    #             dataio.save_current_figure('results_fitexp_loglike_ratioconj%.2f_{label}_global_{unique_id}.pdf' % ratio_conj)

    # if plot_2d_pcolor:
    #     # Plot the mean loglikelihood as a 2d surface
    #     pcolor_2d_data(result_log_posterior_mean, x=ratio_space, y=sigmax_space, xlabel="Ratio conj", ylabel="Sigma x", title="Loglikelihood of experimental data, \n3 items dualrecall, rcscale automatically set", ticks_interpolate=5, cmap=colormap)
    #     # plt.tight_layout()

    #     if savefigs:
    #         dataio.save_current_figure('results_fitexp_loglike_2d_ratiosigmax_{label}_global_{unique_id}.pdf')


    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['result_all_precisions_mean', 'result_em_fits_mean', 'result_all_precisions_std', 'result_em_fits_std', 'enforce_min_distance_space', 'sigmax_space', 'ratio_space', 'all_args']

    if savedata:
        dataio.save_variables(variables_to_save, locals())


    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='See patterns of errors on Specific Stimuli, with Mixed population code. Internally vary ratio_conj. Vary sigmax and enforce_min_distance here. Still need to play around with M, do it in different folders.',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['enforce_min_distance', 'sigmax'],
                     variables_to_load=['result_all_precisions', 'result_em_fits', 'result_em_resp'],
                     variables_description=['Precision of recall', 'Fits mixture model', 'Responsibilities mixture model'],
                     post_processing=plots_specific_stimuli_mixed,
                     save_output_filename='plots_specificstimuli_mixed'
                     )




if __name__ == '__main__':

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

