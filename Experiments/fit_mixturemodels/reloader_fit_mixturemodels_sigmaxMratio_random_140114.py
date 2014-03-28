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
    result_em_fits_flat = np.array(data_pbs.dict_arrays['result_em_fits']['results_flat'])
    result_dist_bays09_flat = np.array(data_pbs.dict_arrays['result_dist_bays09']['results_flat'])
    result_dist_gorgo11_flat = np.array(data_pbs.dict_arrays['result_dist_gorgo11']['results_flat'])
    result_parameters_flat = np.array(data_pbs.dict_arrays['result_em_fits']['parameters_flat'])

    M_space = data_pbs.loaded_data['parameters_uniques']['M']
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


    ## Compute some stuff

    # result_dist_bays09_kappa_T1_avg = utils.nanmean(result_dist_bays09_flat[:, 0, 0], axis=-1)
    # result_dist_bays09_kappa_allT_avg = np.nansum(utils.nanmean(result_dist_bays09_flat[:, :, 0], axis=-1), axis=1)

    result_dist_bays09_allT_avg = utils.nanmean((result_em_fits_flat[:, :, :4] - bays09_em_target[np.newaxis, :, :, np.newaxis])**2, axis=-1)
    result_dist_bays09_emmixt_sum = np.nansum(np.nansum(result_dist_bays09_allT_avg[:, :, 1:], axis=-1), axis=-1)
    result_dist_bays09_kappa_sum = np.nansum(result_dist_bays09_allT_avg[:, :, 0], axis=-1)

    result_dist_bays09_kappa_T1_sum = result_dist_bays09_allT_avg[:, 0, 0]
    result_dist_bays09_kappa_T25_sum = np.nansum(result_dist_bays09_allT_avg[:, 1:, 0], axis=-1)

    result_dist_bays09_emmixt_T1_sum = np.nansum(result_dist_bays09_allT_avg[:, 0, 1:], axis=-1)
    result_dist_bays09_emmixt_T25_sum = np.nansum(np.nansum(result_dist_bays09_allT_avg[:, 1:, 1:], axis=-1), axis=-1)

    result_dist_bays09_both_normalised = result_dist_bays09_emmixt_sum/np.max(result_dist_bays09_emmixt_sum) + result_dist_bays09_kappa_sum/np.max(result_dist_bays09_kappa_sum)

    if plots_dist_bays09:
        nb_best_points = 30
        size_normal_points = 8
        size_best_points = 50

        def plot_scatter(all_vars, result_dist_to_use_name, title=''):
            fig = plt.figure()
            ax = Axes3D(fig)

            result_dist_to_use = all_vars[result_dist_to_use_name]
            utils.scatter3d(result_parameters_flat[:, 0], result_parameters_flat[:, 1], result_parameters_flat[:, 2], s=size_normal_points, c=np.log(result_dist_to_use), xlabel=parameter_names_sorted[0], ylabel=parameter_names_sorted[1], zlabel=parameter_names_sorted[2], title=title, ax_handle=ax)
            best_points_result_dist_to_use = np.argsort(result_dist_to_use)[:nb_best_points]
            utils.scatter3d(result_parameters_flat[best_points_result_dist_to_use, 0], result_parameters_flat[best_points_result_dist_to_use, 1], result_parameters_flat[best_points_result_dist_to_use, 2], c='r', s=size_best_points, ax_handle=ax)
            print "Best points, %s:" % title
            print '\n'.join(['M %d, ratio %.2f, sigmax %.2f:  %f' % (result_parameters_flat[i, 0], result_parameters_flat[i, 1], result_parameters_flat[i, 2], result_dist_to_use[i]) for i in best_points_result_dist_to_use])

            if savefigs:
                dataio.save_current_figure('scatter3d_%s_{label}_{unique_id}.pdf' % result_dist_to_use_name)

                if savemovies:
                    try:

                        utils.rotate_plot3d(ax, dataio.create_formatted_filename('scatter3d_%s_{label}_{unique_id}.mp4' % result_dist_to_use_name), bitrate=8000, min_duration=8)
                        utils.rotate_plot3d(ax, dataio.create_formatted_filename('scatter3d_%s_{label}_{unique_id}.gif' % result_dist_to_use_name), nb_frames=30, min_duration=8)
                    except Exception:
                        # Most likely wrong aggregator...
                        print "failed when creating movies for ", result_dist_to_use_name

                ax.view_init(azim=90, elev=10)
                dataio.save_current_figure('scatter3d_%s_view2_{label}_{unique_id}.pdf' % result_dist_to_use_name)

            return ax

        # Distance for kappa, all T
        plot_scatter(locals(), 'result_dist_bays09_kappa_sum', 'kappa all T')

        # Distance for em fits, all T
        plot_scatter(locals(), 'result_dist_bays09_emmixt_sum', 'em fits, all T')

        # Distance for sum of normalised em fits + normalised kappa, all T
        plot_scatter(locals(), 'result_dist_bays09_both_normalised', 'summed normalised em mixt + kappa')

        # Distance kappa T = 1
        plot_scatter(locals(), 'result_dist_bays09_kappa_T1_sum', 'Kappa T=1')

        # Distance kappa T = 2...5
        plot_scatter(locals(), 'result_dist_bays09_kappa_T25_sum', 'Kappa T=2/5')

        # Distance em fits T = 1
        plot_scatter(locals(), 'result_dist_bays09_emmixt_T1_sum', 'em fits T=1')

        # Distance em fits T = 2...5
        plot_scatter(locals(), 'result_dist_bays09_emmixt_T25_sum', 'em fits T=2/5')



    # if plots_per_T:
    #     for T in T_space:
    #         currT_indices = result_parameters_flat[:, 2] == T

    #         utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat[currT_indices][..., :2], result_fitexperiments_bic_avg[currT_indices], xlabel='Ratio_conj', ylabel='sigma x', title='BIC, T %d' % T, interpolation_numpoints=200, interpolation_method='nearest', log_scale=False)

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
        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='fit_mixturemodels')


    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Fitting of experimental data. All experiments. Random sampling of parameter space. Perhaps too big, be careful...',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['M', 'ratio_conj', 'sigmax'],
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

