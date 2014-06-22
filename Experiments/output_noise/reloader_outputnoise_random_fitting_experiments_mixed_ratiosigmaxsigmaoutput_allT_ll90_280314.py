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

from experimentlauncher import *
from dataio import *

import re
import inspect

import utils
# import submitpbs

from mpl_toolkits.mplot3d import Axes3D

# Commit @2042319 +


def plots_fitting_experiments_random(data_pbs, generator_module=None):
    '''
        Reload 2D volume runs from PBS and plot them

    '''

    #### SETUP
    #
    savefigs = True
    savedata = True
    savemovies = True

    plots_per_T = True
    scatter3d_all_T = True

    nb_best_points = 20
    nb_best_points_per_T = nb_best_points/6
    size_normal_points = 8
    size_best_points = 50
    downsampling = 2


    # do_relaunch_bestparams_pbs = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()
    # parameters: ratio_conj, sigmax, T

    # Extract data
    result_fitexperiments_flat = np.array(data_pbs.dict_arrays['result_fitexperiments']['results_flat'])
    result_fitexperiments_all_flat = np.array(data_pbs.dict_arrays['result_fitexperiments_all']['results_flat'])
    result_fitexperiments_noiseconv_flat = np.array(data_pbs.dict_arrays['result_fitexperiments_noiseconv']['results_flat'])
    result_fitexperiments_noiseconv_all_flat = np.array(data_pbs.dict_arrays['result_fitexperiments_noiseconv_all']['results_flat'])
    result_parameters_flat = np.array(data_pbs.dict_arrays['result_fitexperiments']['parameters_flat'])

    # Extract order of datasets
    experiment_ids = data_pbs.loaded_data['datasets_list'][0]['fitexperiment_parameters']['experiment_ids']
    parameter_names_sorted = data_pbs.dataset_infos['parameters']

    T_space = data_pbs.loaded_data['datasets_list'][0]['T_space']

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    # Compute some stuff
    result_fitexperiments_noiseconv_bic_avg_allT = utils.nanmean(result_fitexperiments_noiseconv_flat, axis=-1)[..., 0]

    # Summed T
    result_fitexperiments_noiseconv_bic_avg_summedT = np.nansum(result_fitexperiments_noiseconv_bic_avg_allT, axis=-1)
    result_fitexperiments_noiseconv_bic_avg_summedT_masked = mask_outliers(result_fitexperiments_noiseconv_bic_avg_summedT)



    def best_points_allT(result_dist_to_use):
        '''
            Best points for all T
        '''
        return np.argsort(result_dist_to_use)[:nb_best_points]

    def plot_scatter(all_vars, result_dist_to_use_name, title='', downsampling=1, label_file=''):
        fig = plt.figure()
        ax = Axes3D(fig)

        result_dist_to_use = all_vars[result_dist_to_use_name]
        utils.scatter3d(result_parameters_flat[:, 0][::downsampling], result_parameters_flat[:, 1][::downsampling], result_parameters_flat[:, 2][::downsampling], s=size_normal_points, c=np.log(result_dist_to_use[::downsampling]), xlabel=parameter_names_sorted[0], ylabel=parameter_names_sorted[1], zlabel=parameter_names_sorted[2], title=title, ax_handle=ax)
        best_points_result_dist_to_use = best_points_allT(result_dist_to_use[::downsampling])
        utils.scatter3d(result_parameters_flat[::downsampling][best_points_result_dist_to_use, 0], result_parameters_flat[::downsampling][best_points_result_dist_to_use, 1], result_parameters_flat[::downsampling][best_points_result_dist_to_use, 2], c='r', s=size_best_points, ax_handle=ax)
        print "Best points, %s:" % title
        print '\n'.join(['%s %.3f, %s %.3f, %s %.3f:  %f' % (parameter_names_sorted[0], result_parameters_flat[::downsampling][i, 0],
            parameter_names_sorted[1], result_parameters_flat[::downsampling][i, 1],
            parameter_names_sorted[2], result_parameters_flat[::downsampling][i, 2], result_dist_to_use[i]) for i in best_points_result_dist_to_use])

        if savefigs:
            dataio.save_current_figure('scatter3d_%s%s_{label}_{unique_id}.pdf' % (result_dist_to_use_name, label_file))

            if savemovies:
                try:
                    utils.rotate_plot3d(ax, dataio.create_formatted_filename('scatter3d_%s%s_{label}_{unique_id}.mp4' % (result_dist_to_use_name, label_file)), bitrate=8000, min_duration=8)
                    utils.rotate_plot3d(ax, dataio.create_formatted_filename('scatter3d_%s%s_{label}_{unique_id}.gif' % (result_dist_to_use_name, label_file)), nb_frames=30, min_duration=8)
                except Exception:
                    # Most likely wrong aggregator...
                    print "failed when creating movies for ", result_dist_to_use_name

            ax.view_init(azim=-90, elev=10)
            dataio.save_current_figure('scatter3d_view2_%s%s_{label}_{unique_id}.pdf' % (result_dist_to_use_name, label_file))

        return ax

    if scatter3d_all_T:

        # plot_scatter(locals(), 'result_fitexperiments_noiseconv_bic_avg_summedT', 'BIC all T')
        plot_scatter(locals(), 'result_fitexperiments_noiseconv_bic_avg_summedT', 'BIC all T', downsampling=3)
        plot_scatter(locals(), 'result_fitexperiments_noiseconv_bic_avg_summedT_masked', 'BIC all T masked', downsampling=5)


    if plots_per_T:
        for T_i, T in enumerate(T_space):

            result_fitexperiments_noiseconv_bic_avg_currT =  result_fitexperiments_noiseconv_bic_avg_allT[..., T_i]
            result_fitexperiments_noiseconv_bic_avg_currT_masked = mask_outliers(result_fitexperiments_noiseconv_bic_avg_currT)

            # plot_scatter(locals(), 'result_fitexperiments_noiseconv_bic_avg_currT', 'BIC T %d' % T, downsampling=5)

            plot_scatter(locals(), 'result_fitexperiments_noiseconv_bic_avg_currT_masked', 'BIC T %d masked' % T, downsampling=5, label_file="T{}".format(T))




    # if plot_per_ratio:
    #     # Plot the evolution of loglike as a function of sigmax, with std shown
    #     for ratio_conj_i, ratio_conj in enumerate(ratio_space):
    #         ax = utils.plot_mean_std_area(sigmax_space, result_log_posterior_mean[ratio_conj_i], result_log_posterior_std[ratio_conj_i])

    #         ax.get_figure().canvas.draw()

    #         if savefigs:
    #             dataio.save_current_figure('results_fitexp_%s_loglike_ratioconj%.2f_{label}_global_{unique_id}.pdf' % (exp_dataset, ratio_conj))



    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['experiment_ids', 'parameter_names_sorted', 'T_space']

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
dataset_infos = dict(label='Random sampling. Fitting of experimental data. Use automatic parameter setting for rcscale and rcscale2, and vary ratio_conj, sigmax and sigma_output. Should fit following datasets: Bays09, Dualrecall, Gorgo11. Compute and store LL and LL90%. Uses new output noise scheme, see if this has an effect',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['ratio_conj', 'sigmax', 'sigma_output'],
                     variables_to_load=['result_fitexperiments', 'result_fitexperiments_all', 'result_fitexperiments_noiseconv', 'result_fitexperiments_noiseconv_all'],
                     variables_description=['Fit experiments summed accross experiments', 'Fit experiments per experiment', 'Fit experiments with noise convolved posterior summed accross experiments', 'Fit experiments with noise convolved posterior'],
                     post_processing=plots_fitting_experiments_random,
                     save_output_filename='plots_fitexp_random_outputnoise'
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

