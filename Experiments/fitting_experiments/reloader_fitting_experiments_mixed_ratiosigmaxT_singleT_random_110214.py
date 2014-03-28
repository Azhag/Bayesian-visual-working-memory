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
    result_parameters_flat = np.array(data_pbs.dict_arrays['result_fitexperiments']['parameters_flat'])

    # Extract order of datasets
    experiment_ids = data_pbs.loaded_data['datasets_list'][0]['fitexperiment_parameters']['experiment_ids']
    parameter_names_sorted = data_pbs.dataset_infos['parameters']


    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    # Compute some stuff
    result_fitexperiments_bic_avg = utils.nanmean(result_fitexperiments_flat[:,0], axis=-1)
    T_space = np.unique(result_parameters_flat[:, 2])


    if plots_per_T:
        for T in T_space:
            currT_indices = result_parameters_flat[:, 2] == T

            utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat[currT_indices][..., :2], result_fitexperiments_bic_avg[currT_indices], xlabel='Ratio_conj', ylabel='sigma x', title='BIC, T %d' % T, interpolation_numpoints=200, interpolation_method='nearest', log_scale=False)

            if savefigs:
                dataio.save_current_figure('contourf_fittingexp_bic_random_ratiosigma_T%d_{label}_{unique_id}.pdf' % (T))


    if scatter3d_all_T:
        nb_best_points_per_T = 2
        size_normal_points = 8
        size_best_points = 50

        # Get the best points for a specific T
        def best_points_T(result_dist_to_use, T):
            currT_indices = result_parameters_flat[:, 2] == T
            best_points_indices_currT = np.argsort(result_dist_to_use[currT_indices])[:nb_best_points_per_T]

            # Indirect it all
            return np.nonzero(currT_indices)[0][best_points_indices_currT]

        def best_points_allT(result_dist_to_use):
            return np.array(utils.flatten_list([best_points_T(result_dist_to_use, T) for T in T_space]))

        def plot_scatter(all_vars, result_dist_to_use_name, title=''):
            fig = plt.figure()
            ax = Axes3D(fig)

            result_dist_to_use = all_vars[result_dist_to_use_name]
            utils.scatter3d(result_parameters_flat[:, 0], result_parameters_flat[:, 1], result_parameters_flat[:, 2], s=size_normal_points, c=np.log(result_dist_to_use), xlabel=parameter_names_sorted[0], ylabel=parameter_names_sorted[1], zlabel=parameter_names_sorted[2], title=title, ax_handle=ax)
            best_points_result_dist_to_use = best_points_allT(result_dist_to_use)
            utils.scatter3d(result_parameters_flat[best_points_result_dist_to_use, 0], result_parameters_flat[best_points_result_dist_to_use, 1], result_parameters_flat[best_points_result_dist_to_use, 2], c='r', s=size_best_points, ax_handle=ax)
            print "Best points, %s:" % title
            print '\n'.join(['ratio %.2f, sigmax %.2f, T %d:  %f' % (result_parameters_flat[i, 0], result_parameters_flat[i, 1], result_parameters_flat[i, 2], result_dist_to_use[i]) for i in best_points_result_dist_to_use])

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
                dataio.save_current_figure('scatter3d_view2_%s_{label}_{unique_id}.pdf' % result_dist_to_use_name)

            return ax

        plot_scatter(locals(), 'result_fitexperiments_bic_avg', 'BIC all T')



    # if plot_per_ratio:
    #     # Plot the evolution of loglike as a function of sigmax, with std shown
    #     for ratio_conj_i, ratio_conj in enumerate(ratio_space):
    #         ax = utils.plot_mean_std_area(sigmax_space, result_log_posterior_mean[ratio_conj_i], result_log_posterior_std[ratio_conj_i])

    #         ax.get_figure().canvas.draw()

    #         if savefigs:
    #             dataio.save_current_figure('results_fitexp_%s_loglike_ratioconj%.2f_{label}_global_{unique_id}.pdf' % (exp_dataset, ratio_conj))



    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['experiment_ids', 'parameter_names_sorted']

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)
        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='fitting_experiments')


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
                     parameters=['ratio_conj', 'sigmax', 'T'],
                     variables_to_load=['result_fitexperiments', 'result_fitexperiments_all'],
                     variables_description=['Fit experiments sum', 'Fit experiments per experiment'],
                     post_processing=plots_fitting_experiments_random,
                     save_output_filename='plots_fitexp_random_mixed'
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

