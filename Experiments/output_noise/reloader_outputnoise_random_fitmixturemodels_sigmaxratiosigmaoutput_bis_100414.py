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

# import matplotlib.animation as plt_anim
from mpl_toolkits.mplot3d import Axes3D

import re
import inspect
import imp

import utils
# import submitpbs
import load_experimental_data

# Commit @2042319 +


def plots_fit_mixturemodels_random(data_pbs, generator_module=None):
    '''
        Reload runs from PBS
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True
    savemovies = True

    plots_dist_bays09 = True
    plots_per_T = False
    plots_interpolate = False

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
    result_dist_bays09_flat = np.array(data_pbs.dict_arrays['result_dist_bays09']['results_flat'])
    result_dist_gorgo11_flat = np.array(data_pbs.dict_arrays['result_dist_gorgo11']['results_flat'])
    result_parameters_flat = np.array(data_pbs.dict_arrays['result_em_fits']['parameters_flat'])

    sigmaoutput_space = data_pbs.loaded_data['parameters_uniques']['sigma_output']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    ratio_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
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

    # result_dist_bays09_kappa_T1_avg = utils.nanmean(result_dist_bays09_flat[:, 0, 0], axis=-1)
    # result_dist_bays09_kappa_allT_avg = np.nansum(utils.nanmean(result_dist_bays09_flat[:, :, 0], axis=-1), axis=1)

    # Square distance to kappa
    result_dist_bays09_allT_avg = utils.nanmean((result_em_fits_flat[:, :, :4] - bays09_em_target[np.newaxis, :, :, np.newaxis])**2, axis=-1)
    result_dist_bays09_kappa_sum = np.nansum(result_dist_bays09_allT_avg[:, :, 0], axis=-1)

    result_dist_bays09_kappa_T1_sum = result_dist_bays09_allT_avg[:, 0, 0]
    result_dist_bays09_kappa_T25_sum = np.nansum(result_dist_bays09_allT_avg[:, 1:, 0], axis=-1)

    # Square and KL distance for EM Mixtures
    result_dist_bays09_emmixt_sum = np.nansum(np.nansum(result_dist_bays09_allT_avg[:, :, 1:], axis=-1), axis=-1)
    result_dist_bays09_emmixt_T1_sum = np.nansum(result_dist_bays09_allT_avg[:, 0, 1:], axis=-1)
    result_dist_bays09_emmixt_T25_sum = np.nansum(np.nansum(result_dist_bays09_allT_avg[:, 1:, 1:], axis=-1), axis=-1)


    result_dist_bays09_emmixt_KL = utils.nanmean(utils.KL_div(result_em_fits_flat[:, :, 1:4], bays09_emmixt_target[np.newaxis, :, :, np.newaxis], axis=-2), axis=-1)   # KL over dimension of mixtures, then mean over repetitions
    result_dist_bays09_emmixt_KL_sum = np.nansum(result_dist_bays09_emmixt_KL, axis=-1)  # sum over T
    result_dist_bays09_emmixt_KL_T1_sum = result_dist_bays09_emmixt_KL[:, 0]
    result_dist_bays09_emmixt_KL_T25_sum = np.nansum(result_dist_bays09_emmixt_KL[:, 1:], axis=-1)


    result_dist_bays09_both_normalised = result_dist_bays09_emmixt_sum/np.max(result_dist_bays09_emmixt_sum) + result_dist_bays09_kappa_sum/np.max(result_dist_bays09_kappa_sum)

    # Mask kappa for performance too bad
    result_dist_bays09_kappa_sum_masked = np.ma.masked_greater(result_dist_bays09_kappa_sum, 2*np.median(result_dist_bays09_kappa_sum))
    result_dist_bays09_emmixt_KL_sum_masked = np.ma.masked_greater(result_dist_bays09_emmixt_KL_sum, 2*np.median(result_dist_bays09_emmixt_KL_sum))
    result_dist_bays09_both_normalised_mult_masked = 1-(1. - result_dist_bays09_emmixt_KL_sum/np.max(result_dist_bays09_emmixt_KL_sum))*(1. - result_dist_bays09_kappa_sum_masked/np.max(result_dist_bays09_kappa_sum_masked))

    if plots_dist_bays09:
        nb_best_points = 30
        size_normal_points = 8
        size_best_points = 50

        def plot_scatter(all_vars, result_dist_to_use_name, title='', log_color=True, downsampling=1, label_file=''):

            fig = plt.figure()
            ax = Axes3D(fig)

            result_dist_to_use = all_vars[result_dist_to_use_name]
            if not log_color:
                result_dist_to_use = np.exp(result_dist_to_use)

            utils.scatter3d(result_parameters_flat[:, 0], result_parameters_flat[:, 1], result_parameters_flat[:, 2], s=size_normal_points, c=np.log(result_dist_to_use), xlabel=parameter_names_sorted[0], ylabel=parameter_names_sorted[1], zlabel=parameter_names_sorted[2], title=title, ax_handle=ax)
            best_points_result_dist_to_use = np.argsort(result_dist_to_use)[:nb_best_points]
            utils.scatter3d(result_parameters_flat[best_points_result_dist_to_use, 0], result_parameters_flat[best_points_result_dist_to_use, 1], result_parameters_flat[best_points_result_dist_to_use, 2], c='r', s=size_best_points, ax_handle=ax)
            print "Best points, %s:" % title
            print '\n'.join(['sigma output %.4f, ratio %.4f, sigmax %.4f:  %f' % (result_parameters_flat[i, 0], result_parameters_flat[i, 1], result_parameters_flat[i, 2], result_dist_to_use[i]) for i in best_points_result_dist_to_use])

            if savefigs:
                dataio.save_current_figure('scatter3d_%s%s_{label}_{unique_id}.pdf' % (result_dist_to_use_name, label_file))

                if savemovies:
                    try:
                        utils.rotate_plot3d(ax, dataio.create_formatted_filename('scatter3d_%s%s_{label}_{unique_id}.mp4' % (result_dist_to_use_name, label_file)), bitrate=8000, min_duration=8)
                        utils.rotate_plot3d(ax, dataio.create_formatted_filename('scatter3d_%s%s_{label}_{unique_id}.gif' % (result_dist_to_use_name, label_file)), nb_frames=30, min_duration=8)
                    except Exception:
                        # Most likely wrong aggregator...
                        print "failed when creating movies for ", result_dist_to_use_name

                ax.view_init(azim=90, elev=10)
                dataio.save_current_figure('scatter3d_view2_%s%s_{label}_{unique_id}.pdf' % (result_dist_to_use_name, label_file))

            return ax

        # Distance for kappa, all T
        # plot_scatter(locals(), 'result_dist_bays09_kappa_sum', 'kappa all T')

        # # Distance for em fits, all T, Squared distance
        # plot_scatter(locals(), 'result_dist_bays09_emmixt_sum', 'em fits, all T')

        # # Distance for em fits, all T, KL distance
        # plot_scatter(locals(), 'result_dist_bays09_emmixt_KL_sum', 'em fits, all T, KL')

        # # Distance for sum of normalised em fits + normalised kappa, all T
        # plot_scatter(locals(), 'result_dist_bays09_both_normalised', 'summed normalised em mixt + kappa')

        # Distance for product of normalised em fits + normalised kappa, all T
        plot_scatter(locals(), 'result_dist_bays09_both_normalised_mult_masked', 'mult normalised em mixt KL, kappa, masked')

        # Distance kappa T = 1
        # plot_scatter(locals(), 'result_dist_bays09_kappa_T1_sum', 'Kappa T=1')

        # # Distance kappa T = 2...5
        # plot_scatter(locals(), 'result_dist_bays09_kappa_T25_sum', 'Kappa T=2/5')

        # # Distance em fits T = 1
        # plot_scatter(locals(), 'result_dist_bays09_emmixt_T1_sum', 'em fits T=1')

        # # Distance em fits T = 2...5
        # plot_scatter(locals(), 'result_dist_bays09_emmixt_T25_sum', 'em fits T=2/5')

        # # Distance em fits T = 1, KL
        # plot_scatter(locals(), 'result_dist_bays09_emmixt_KL_T1_sum', 'em fits T=1, KL')

        # # Distance em fits T = 2...5, KL
        # plot_scatter(locals(), 'result_dist_bays09_emmixt_KL_T25_sum', 'em fits T=2/5, KL')



    if plots_per_T:
        for T_i, T in enumerate(T_space):

            # Kappa per T, fit to Bays09
            result_dist_bays09_kappa_currT = result_dist_bays09_allT_avg[:, T_i, 0]
            result_dist_bays09_kappa_currT_masked = mask_outliers(result_dist_bays09_kappa_currT)

            plot_scatter(locals(), 'result_dist_bays09_kappa_currT_masked', 'kappa T %d masked' % T, label_file="T{}".format(T))

            # EM Mixt per T, fit to Bays09
            result_dist_bays09_emmixt_sum_currT = np.nansum(result_dist_bays09_allT_avg[:, T_i, 1:], axis=-1)
            result_dist_bays09_emmixt_sum_currT_masked = mask_outliers(result_dist_bays09_emmixt_sum_currT)

            plot_scatter(locals(), 'result_dist_bays09_emmixt_sum_currT_masked', 'EM mixt T %d masked' % T, label_file="T{}".format(T))

            # EM Mixt per T, fit to Bays09 KL divergence
            result_dist_bays09_emmixt_KL_sum_currT = result_dist_bays09_emmixt_KL[:, T_i]
            plot_scatter(locals(), 'result_dist_bays09_emmixt_KL_sum_currT', 'KL EM mixt T %d masked' % T, label_file="T{}".format(T))




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
        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='output_noise')


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
                     parameters=['sigma_output', 'ratio_conj', 'sigmax'],
                     variables_to_load=['result_em_fits', 'result_em_fits_allnontargets', 'result_dist_bays09', 'result_dist_gorgo11'],
                     variables_description=['EM fits, using all nontargets (may be wrong)', 'EM fits, with nontargets mixtures', 'Distance of EM fits to Bays09 dataset (kappa, mixtures)', 'Distance to Gorgo11 dataset (kappa, mixtures)'],
                     post_processing=plots_fit_mixturemodels_random,
                     save_output_filename='plots_fitmixtmodel_random_mixed'
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

