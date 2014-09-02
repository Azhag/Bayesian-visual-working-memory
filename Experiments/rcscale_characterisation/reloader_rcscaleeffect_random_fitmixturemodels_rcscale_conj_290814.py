"""
    ExperimentDescriptor for Fitting experiments in a mixed population code
"""

import os
import numpy as np

import matplotlib as mpl
if 'DISPLAY' in os.environ and mpl.get_backend() == 'pdf':
    # Most likely Gatsby machine interactively, change backend
    mpl.use('TkAgg')

import matplotlib.pyplot as plt

from experimentlauncher import ExperimentLauncher
from dataio import DataIO
import launchers

# import matplotlib.animation as plt_anim
from mpl_toolkits.mplot3d import Axes3D

import re
import inspect
import imp

import utils
# import submitpbs
import load_experimental_data

# Commit @2042319 +


def plots_fitmixtmodel_rcscale_effect(data_pbs, generator_module=None):
    '''
        Reload runs from PBS
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plots_all_T = True
    plots_per_T = True

    # do_relaunch_bestparams_pbs = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", data_pbs.dataset_infos['parameters']
    # parameters: M, ratio_conj, sigmax

    # Extract data
    T_space = data_pbs.loaded_data['datasets_list'][0]['T_space']

    result_em_fits_flat = np.array(data_pbs.dict_arrays['result_em_fits']['results_flat'])
    result_precisions_flat = np.array(data_pbs.dict_arrays['result_all_precisions']['results_flat'])
    result_dist_bays09_flat = np.array(data_pbs.dict_arrays['result_dist_bays09']['results_flat'])
    result_dist_gorgo11_flat = np.array(data_pbs.dict_arrays['result_dist_gorgo11']['results_flat'])
    result_dist_bays09_emmixt_KL = np.array(data_pbs.dict_arrays['result_dist_bays09_emmixt_KL']['results_flat'])
    result_dist_gorgo11_emmixt_KL = np.array(data_pbs.dict_arrays['result_dist_gorgo11_emmixt_KL']['results_flat'])
    result_parameters_flat = np.array(data_pbs.dict_arrays['result_em_fits']['parameters_flat'])

    rc_scale_space = data_pbs.loaded_data['parameters_uniques']['rc_scale']
    num_repetitions = generator_module.num_repetitions
    parameter_names_sorted = data_pbs.dataset_infos['parameters']

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    # Load bays09
    data_bays09 = load_experimental_data.load_data_bays09(fit_mixture_model=True)
    bays09_nitems = data_bays09['data_to_fit']['n_items']
    bays09_em_target = np.nan*np.empty((bays09_nitems.max(), 4))  #kappa, prob_target, prob_nontarget, prob_random
    bays09_em_target[bays09_nitems - 1] = data_bays09['em_fits_nitems_arrays']['mean'].T
    bays09_emmixt_target = bays09_em_target[:, 1:]


    ## Compute some stuff
    result_parameters_flat = result_parameters_flat.flatten()

    result_em_fits_all_avg = utils.nanmean(result_em_fits_flat, axis=-1)
    result_em_kappa_allT = result_em_fits_all_avg[..., 0]
    result_em_emmixt_allT = result_em_fits_all_avg[..., 1:4]

    result_precisions_all_avg = utils.nanmean(result_precisions_flat, axis=-1)

    # Square distance to kappa
    result_dist_bays09_allT_avg = utils.nanmean(result_dist_bays09_flat, axis=-1)
    result_dist_bays09_emmixt_KL_allT_avg = utils.nanmean(result_dist_bays09_emmixt_KL, axis=-1)

    result_dist_bays09_kappa_allT = result_dist_bays09_allT_avg[..., 0]

    # result_dist_bays09_allT_avg = utils.nanmean((result_em_fits_flat[:, :, :4] - bays09_em_target[np.newaxis, :, :, np.newaxis])**2, axis=-1)
    # result_dist_bays09_kappa_sum = np.nansum(result_dist_bays09_allT_avg[:, :, 0], axis=-1)

    # result_dist_bays09_kappa_T1_sum = result_dist_bays09_allT_avg[:, 0, 0]
    # result_dist_bays09_kappa_T25_sum = np.nansum(result_dist_bays09_allT_avg[:, 1:, 0], axis=-1)

    # # Square and KL distance for EM Mixtures
    # result_dist_bays09_emmixt_sum = np.nansum(np.nansum(result_dist_bays09_allT_avg[:, :, 1:], axis=-1), axis=-1)
    # result_dist_bays09_emmixt_T1_sum = np.nansum(result_dist_bays09_allT_avg[:, 0, 1:], axis=-1)
    # result_dist_bays09_emmixt_T25_sum = np.nansum(np.nansum(result_dist_bays09_allT_avg[:, 1:, 1:], axis=-1), axis=-1)


    # result_dist_bays09_emmixt_KL = utils.nanmean(utils.KL_div(result_em_fits_flat[:, :, 1:4], bays09_emmixt_target[np.newaxis, :, :, np.newaxis], axis=-2), axis=-1)   # KL over dimension of mixtures, then mean over repetitions
    # result_dist_bays09_emmixt_KL_sum = np.nansum(result_dist_bays09_emmixt_KL, axis=-1)  # sum over T
    # result_dist_bays09_emmixt_KL_T1_sum = result_dist_bays09_emmixt_KL[:, 0]
    # result_dist_bays09_emmixt_KL_T25_sum = np.nansum(result_dist_bays09_emmixt_KL[:, 1:], axis=-1)


    # result_dist_bays09_both_normalised = result_dist_bays09_emmixt_sum/np.max(result_dist_bays09_emmixt_sum) + result_dist_bays09_kappa_sum/np.max(result_dist_bays09_kappa_sum)

    # # Mask kappa for performance too bad
    # result_dist_bays09_kappa_sum_masked = np.ma.masked_greater(result_dist_bays09_kappa_sum, 2*np.median(result_dist_bays09_kappa_sum))
    # result_dist_bays09_emmixt_KL_sum_masked = np.ma.masked_greater(result_dist_bays09_emmixt_KL_sum, 2*np.median(result_dist_bays09_emmixt_KL_sum))
    # result_dist_bays09_both_normalised_mult_masked = 1-(1. - result_dist_bays09_emmixt_KL_sum/np.max(result_dist_bays09_emmixt_KL_sum))*(1. - result_dist_bays09_kappa_sum_masked/np.max(result_dist_bays09_kappa_sum_masked))

    # Compute optimal rc_scale
    all_args = data_pbs.loaded_data['args_list']
    specific_arg = all_args[0]
    specific_arg['autoset_parameters'] = True
    (_, _, _, sampler) = launchers.init_everything(specific_arg)
    optimal_rc_scale = sampler.random_network.rc_scale[0]

    if plots_all_T:
        # Show Kappa evolution wrt rc_scale
        f, ax = plt.subplots()
        # utils.plot_mean_std_from_samples(result_parameters_flat, np.nansum(result_em_kappa_allT, axis=-1), bins=60, bins_y=150, xlabel='rc_scale', ylabel='EM kappa', title='Kappa, summed T',  ax_handle=ax, show_scatter=False)
        utils.plot_mean_std_from_samples_rolling(result_parameters_flat, np.nansum(result_em_kappa_allT, axis=-1), window=35, xlabel='rc_scale', ylabel='EM kappa', title='Kappa, summed T',  ax_handle=ax, show_scatter=False)
        ax.axvline(x=optimal_rc_scale, color='g', linewidth=2)
        ax.axvline(x=2*optimal_rc_scale, color='r', linewidth=2)
        f.canvas.draw()

        if savefigs:
            dataio.save_current_figure('rcscaleeffect_kappa_summedT_{label}_{unique_id}.pdf')

        # Show Mixt proportions
        f, ax = plt.subplots()
        for i in xrange(3):
            # utils.plot_mean_std_from_samples(result_parameters_flat, np.nansum(result_em_emmixt_allT[..., i], axis=-1), bins=60, bins_y=100, xlabel='rc_scale', ylabel='EM mixt proportions', title='EM mixtures, summed T',  ax_handle=ax, show_scatter=False)
            utils.plot_mean_std_from_samples_rolling(result_parameters_flat, np.nansum(result_em_emmixt_allT[..., i], axis=-1), window=35, xlabel='rc_scale', ylabel='EM mixt proportions', title='EM mixtures, summed T',  ax_handle=ax, show_scatter=False)
        ax.axvline(x=optimal_rc_scale, color='g', linewidth=2)
        ax.axvline(x=2*optimal_rc_scale, color='r', linewidth=2)
        f.canvas.draw()

        if savefigs:
            dataio.save_current_figure('rcscaleeffect_mixtprop_summedT_{label}_{unique_id}.pdf')

        # Show Precision
        f, ax = plt.subplots()
        # utils.plot_mean_std_from_samples(result_parameters_flat, np.nansum(result_precisions_all_avg, axis=-1), bins=60, bins_y=150, xlabel='rc_scale', ylabel='Precision', title='Precision, summed T',  ax_handle=ax, show_scatter=False)
        utils.plot_mean_std_from_samples_rolling(result_parameters_flat, np.nansum(result_precisions_all_avg, axis=-1), window=35, xlabel='rc_scale', ylabel='Precision', title='Precision, summed T',  ax_handle=ax, show_scatter=False)
        ax.axvline(x=optimal_rc_scale, color='g', linewidth=2)
        ax.axvline(x=2*optimal_rc_scale, color='r', linewidth=2)
        f.canvas.draw()

        if savefigs:
            dataio.save_current_figure('rcscaleeffect_precision_summedT_{label}_{unique_id}.pdf')


        plt.close('all')


    if plots_per_T:
        for T_i, T in enumerate(T_space):
            # Show Kappa evolution wrt rc_scale
            f, ax = plt.subplots()
            # utils.plot_mean_std_from_samples(result_parameters_flat, result_em_kappa_allT[:, T_i], bins=40, bins_y=100, xlabel='rc_scale', ylabel='EM kappa', title='Kappa, T %d' % T,  ax_handle=ax, show_scatter=False)
            utils.plot_mean_std_from_samples_rolling(result_parameters_flat, result_em_kappa_allT[:, T_i], window=35, xlabel='rc_scale', ylabel='EM kappa', title='Kappa, T %d' % T,  ax_handle=ax, show_scatter=False)
            ax.axvline(x=optimal_rc_scale, color='g', linewidth=2)
            ax.axvline(x=2*optimal_rc_scale, color='r', linewidth=2)
            f.canvas.draw()

            if savefigs:
                dataio.save_current_figure('rcscaleeffect_kappa_T%d_{label}_{unique_id}.pdf' % T)

            # Show Mixt proportions
            f, ax = plt.subplots()
            for i in xrange(3):
                # utils.plot_mean_std_from_samples(result_parameters_flat, result_em_emmixt_allT[:, T_i, i], bins=40, bins_y=100, xlabel='rc_scale', ylabel='EM mixt proportions', title='EM mixtures, T %d' % T,  ax_handle=ax, show_scatter=False)
                utils.plot_mean_std_from_samples_rolling(result_parameters_flat, result_em_emmixt_allT[:, T_i, i], window=35, xlabel='rc_scale', ylabel='EM mixt proportions', title='EM mixtures, T %d' % T,  ax_handle=ax, show_scatter=False)
            ax.axvline(x=optimal_rc_scale, color='g', linewidth=2)
            ax.axvline(x=2*optimal_rc_scale, color='r', linewidth=2)
            f.canvas.draw()

            if savefigs:
                dataio.save_current_figure('rcscaleeffect_mixtprop_T%d_{label}_{unique_id}.pdf' % T)

            # Show Precision
            f, ax = plt.subplots()
            # utils.plot_mean_std_from_samples(result_parameters_flat, result_precisions_all_avg[:, T_i], bins=40, bins_y=100, xlabel='rc_scale', ylabel='Precision', title='Precision, T %d' % T,  ax_handle=ax, show_scatter=False)
            utils.plot_mean_std_from_samples_rolling(result_parameters_flat, result_precisions_all_avg[:, T_i], window=35, xlabel='rc_scale', ylabel='Precision', title='Precision, T %d' % T,  ax_handle=ax, show_scatter=False)
            ax.axvline(x=optimal_rc_scale, color='g', linewidth=2)
            ax.axvline(x=2*optimal_rc_scale, color='r', linewidth=2)
            f.canvas.draw()

            if savefigs:
                dataio.save_current_figure('rcscaleeffect_precision_T%d_{label}_{unique_id}.pdf' % T)

            plt.close('all')





    # # Interpolate
    # if plots_interpolate:

    #     sigmax_target = 0.9

    #     M_interp_space = np.arange(6, 625, 5)
    #     ratio_interp_space = np.linspace(0.01, 1.0, 50)
    #     # sigmax_interp_space = np.linspace(0.01, 1.0, 50)
    #     sigmax_interp_space = np.array([sigmax_target])
    #     params_crossspace = np.array(utils.cross(M_interp_space, ratio_interp_space, sigmax_interp_space))

    #     interpolated_data = rbf_interpolator(params_crossspace[:, 0], params_crossspace[:, 1], params_crossspace[:, 2]).reshape((M_interp_space.size, ratio_interp_space.size))

    #     utils.pcolor_2d_data(interpolated_data, M_interp_space, ratio_interp_space, 'M', 'ratio', 'interpolated, fixing sigmax= %.2f' % sigmax_target)

    #     points_closeby = ((result_parameters_flat[:, 2] - sigmax_target)**2)< 0.01
    #     plt.figure()
    #     # plt.imshow(interpolated_data, extent=(M_interp_space.min(), M_interp_space.max(), ratio_interp_space.min(), ratio_interp_space.max()))
    #     plt.imshow(interpolated_data)
    #     plt.scatter(result_parameters_flat[points_closeby, 0], result_parameters_flat[points_closeby, 1], s=100, c=result_fitexperiments_bic_avg[points_closeby], marker='o')


    # if plot_per_ratio:
    #     # Plot the evolution of loglike as a function of sigmax, with std shown
    #     for ratio_conj_i, ratio_conj in enumerate(ratio_space):
    #         ax = utils.plot_mean_std_area(sigmax_space, result_log_posterior_mean[ratio_conj_i], result_log_posterior_std[ratio_conj_i])

    #         ax.get_figure().canvas.draw()

    #         if savefigs:
    #             dataio.save_current_figure('results_fitexp_%s_loglike_ratioconj%.2f_{label}_global_{unique_id}.pdf' % (exp_dataset, ratio_conj))



    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['parameter_names_sorted']

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)
        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='rcscale_characterisation')


    plt.show()

    return locals()


def mask_outliers(result_dist_to_use, sigma_outlier=3):
    '''
        Mask outlier datapoints.
        Compute the mean of the results and assume that points with:
          result > mean + sigma_outlier*std
        are outliers.

        As we want the minimum values, do not mask small values
    '''
    return np.ma.masked_greater(result_dist_to_use, np.mean(result_dist_to_use) + sigma_outlier*np.std(result_dist_to_use))



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Fitting of experimental data. All experiments. Random sampling of parameter space. Perhaps too big, be careful...',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['rc_scale'],
                     variables_to_load=['result_em_fits', 'result_em_fits_allnontargets', 'result_dist_bays09', 'result_dist_gorgo11', 'result_dist_bays09_emmixt_KL', 'result_dist_gorgo11_emmixt_KL', 'result_all_precisions'],
                     variables_description=['EM fits, using all nontargets (may be wrong)', 'EM fits, with nontargets mixtures', 'Distance of EM fits to Bays09 dataset (kappa, mixtures)', 'Distance to Gorgo11 dataset (kappa, mixtures)', 'KL Distance of EM fits to Bays09 dataset (kappa, mixtures)', 'KL Distance to Gorgo11 dataset (kappa, mixtures)', 'Precision'],
                     post_processing=plots_fitmixtmodel_rcscale_effect,
                     save_output_filename='plots_fitmixtmodel_rcscale_effect',
                     construct_multidimension_npyarr=False
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

