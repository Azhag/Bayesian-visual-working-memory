"""
    ExperimentDescriptor for Fitting experiments in a hierarchical population code
"""

import os
import numpy as np
from experimentlauncher import *
from dataio import *
import re
import imp

import inspect

import utils

# Commit @0134c44


def plots_specific_stimuli_hierarchical(data_pbs, generator_module=None):
    '''
        Reload and plot behaviour of mixed population code on specific Stimuli
        of 3 items.
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plot_per_min_dist_all = True
    specific_plots_paper = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_all_precisions_mean = utils.nanmean(np.squeeze(data_pbs.dict_arrays['result_all_precisions']['results']), axis=-1)
    result_all_precisions_std = utils.nanstd(np.squeeze(data_pbs.dict_arrays['result_all_precisions']['results']), axis=-1)
    result_em_fits_mean = utils.nanmean(np.squeeze(data_pbs.dict_arrays['result_em_fits']['results']), axis=-1)
    result_em_fits_std = utils.nanstd(np.squeeze(data_pbs.dict_arrays['result_em_fits']['results']), axis=-1)
    result_em_kappastddev_mean = utils.nanmean(utils.kappa_to_stddev(np.squeeze(data_pbs.dict_arrays['result_em_fits']['results'])[..., 0, :]), axis=-1)
    result_em_kappastddev_std = utils.nanstd(utils.kappa_to_stddev(np.squeeze(data_pbs.dict_arrays['result_em_fits']['results'])[..., 0, :]), axis=-1)
    # result_em_resp_all = np.squeeze(data_pbs.dict_arrays['result_em_resp']['results'])


    enforce_min_distance_space = data_pbs.loaded_data['parameters_uniques']['enforce_min_distance']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']

    MMlower_valid_space = data_pbs.loaded_data['datasets_list'][0]['MMlower_valid_space']
    ratio_space = MMlower_valid_space[:, 0]/float(np.sum(MMlower_valid_space[0]))

    print enforce_min_distance_space
    print sigmax_space
    print MMlower_valid_space
    print result_all_precisions_mean.shape, result_em_fits_mean.shape

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])


    if plot_per_min_dist_all:
        # Do one plot per min distance.
        for min_dist_i, min_dist in enumerate(enforce_min_distance_space):
            # Show log precision
            utils.pcolor_2d_data(result_all_precisions_mean[min_dist_i].T, x=ratio_space, y=sigmax_space, xlabel='ratio layer two', ylabel='sigma_x', title='Precision, min_dist=%.3f' % min_dist)
            if savefigs:
                dataio.save_current_figure('precision_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)

            # Show log precision
            utils.pcolor_2d_data(result_all_precisions_mean[min_dist_i].T, x=ratio_space, y=sigmax_space, xlabel='ratio layer two', ylabel='sigma_x', title='Precision, min_dist=%.3f' % min_dist, log_scale=True)
            if savefigs:
                dataio.save_current_figure('logprecision_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)


            # Plot estimated model precision
            utils.pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., 0].T, x=ratio_space, y=sigmax_space, xlabel='ratio layer two', ylabel='sigma_x', title='EM precision, min_dist=%.3f' % min_dist, log_scale=False)
            if savefigs:
                dataio.save_current_figure('logemprecision_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)

            # Plot estimated Target, nontarget and random mixture components, in multiple subplots
            _, axes = plt.subplots(1, 3, figsize=(18, 6))
            plt.subplots_adjust(left=0.05, right=0.97, wspace = 0.3, bottom=0.15)
            utils.pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., 1].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='Target, min_dist=%.3f' % min_dist, log_scale=False, ax_handle=axes[0], ticks_interpolate=5)
            utils.pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., 2].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='Nontarget, min_dist=%.3f' % min_dist, log_scale=False, ax_handle=axes[1], ticks_interpolate=5)
            utils.pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., 3].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='Random, min_dist=%.3f' % min_dist, log_scale=False, ax_handle=axes[2], ticks_interpolate=5)

            if savefigs:
                dataio.save_current_figure('em_mixtureprobs_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)

            # Plot Log-likelihood of Mixture model, sanity check
            utils.pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., -1].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='EM loglik, min_dist=%.3f' % min_dist, log_scale=False)
            if savefigs:
                dataio.save_current_figure('em_loglik_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)

    if specific_plots_paper:
        # We need to choose 3 levels of min_distances
        target_sigmax = 0.25
        target_mindist_low = 0.09
        target_mindist_medium = 0.36
        target_mindist_high = 1.5

        sigmax_level_i = np.argmin(np.abs(sigmax_space - target_sigmax))
        min_dist_level_low_i = np.argmin(np.abs(enforce_min_distance_space - target_mindist_low))
        min_dist_level_medium_i = np.argmin(np.abs(enforce_min_distance_space - target_mindist_medium))
        min_dist_level_high_i = np.argmin(np.abs(enforce_min_distance_space - target_mindist_high))

        ## Do for each distance
        for min_dist_i in [min_dist_level_low_i, min_dist_level_medium_i, min_dist_level_high_i]:
            # Plot precision
            utils.plot_mean_std_area(ratio_space, result_all_precisions_mean[min_dist_i, sigmax_level_i], result_all_precisions_std[min_dist_i, sigmax_level_i]) #, xlabel='Ratio conjunctivity', ylabel='Precision of recall')
            # plt.title('Min distance %.3f' % enforce_min_distance_space[min_dist_i])

            if savefigs:
                dataio.save_current_figure('mindist%.2f_precisionrecall_forpaper_{label}_{unique_id}.pdf' % enforce_min_distance_space[min_dist_i])

            # Plot kappa fitted
            utils.plot_mean_std_area(ratio_space, result_em_fits_mean[min_dist_i, sigmax_level_i, :, 0], result_em_fits_std[min_dist_i, sigmax_level_i, :, 0]) #, xlabel='Ratio conjunctivity', ylabel='Fitted kappa')
            # plt.title('Min distance %.3f' % enforce_min_distance_space[min_dist_i])
            if savefigs:
                dataio.save_current_figure('mindist%.2f_emkappa_forpaper_{label}_{unique_id}.pdf' % enforce_min_distance_space[min_dist_i])

            # Plot kappa-stddev fitted. Easier to visualize
            utils.plot_mean_std_area(ratio_space, result_em_kappastddev_mean[min_dist_i, sigmax_level_i], result_em_kappastddev_std[min_dist_i, sigmax_level_i]) #, xlabel='Ratio conjunctivity', ylabel='Fitted kappa_stddev')
            # plt.title('Min distance %.3f' % enforce_min_distance_space[min_dist_i])
            if savefigs:
                dataio.save_current_figure('mindist%.2f_emkappastddev_forpaper_{label}_{unique_id}.pdf' % enforce_min_distance_space[min_dist_i])


            # Plot LLH
            utils.plot_mean_std_area(ratio_space, result_em_fits_mean[min_dist_i, sigmax_level_i, :, -1], result_em_fits_std[min_dist_i, sigmax_level_i, :, -1]) #, xlabel='Ratio conjunctivity', ylabel='Loglikelihood of Mixture model fit')
            # plt.title('Min distance %.3f' % enforce_min_distance_space[min_dist_i])
            if savefigs:
                dataio.save_current_figure('mindist%.2f_emllh_forpaper_{label}_{unique_id}.pdf' % enforce_min_distance_space[min_dist_i])

            # Plot mixture parameters
            utils.plot_multiple_mean_std_area(ratio_space, result_em_fits_mean[min_dist_i, sigmax_level_i, :, 1:4].T, result_em_fits_std[min_dist_i, sigmax_level_i, :, 1:4].T)
            # plt.legend("Target", "Non-target", "Random")
            if savefigs:
                dataio.save_current_figure('mindist%.2f_emprobs_forpaper_{label}_{unique_id}.pdf' % enforce_min_distance_space[min_dist_i])



    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['result_all_precisions_mean', 'result_em_fits_mean', 'result_all_precisions_std', 'result_em_fits_std', 'result_em_kappastddev_mean', 'result_em_kappastddev_std', 'enforce_min_distance_space', 'sigmax_space', 'ratio_space', 'all_args']

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
                     post_processing=plots_specific_stimuli_hierarchical,
                     save_output_filename='plots_specificstimuli_hierarchical',
                     concatenate_multiple_datasets=True
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

