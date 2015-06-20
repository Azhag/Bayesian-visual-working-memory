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
import launchers_memorycurves_marginal_fi
import plots_experimental_data

# import matplotlib.animation as plt_anim
from mpl_toolkits.mplot3d import Axes3D

import re
import inspect
import imp

import utils
# import submitpbs
import load_experimental_data



def plots_fit_collapsedmixturemodels_random(data_pbs, generator_module=None):
    '''
        Reload runs from PBS

        Sequential data analysis.
    '''

    #### SETUP
    #
    plots_bestfits = True
    plots_scatter3d = False

    savefigs = True
    savedata = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", data_pbs.dataset_infos['parameters']
    # parameters: M, ratio_conj, sigmax

    # Extract data
    T_space = data_pbs.loaded_data['datasets_list'][0]['T_space']

    result_em_fits_collapsed_tr_flat = np.array(data_pbs.dict_arrays['result_em_fits_collapsed_tr']['results_flat'])
    result_em_fits_collapsed_summary_flat = np.array(data_pbs.dict_arrays['result_em_fits_collapsed_summary']['results_flat'])
    result_dist_gorgo11_sequ_collapsed_flat = np.array(data_pbs.dict_arrays['result_dist_gorgo11_sequ_collapsed']['results_flat'])
    result_dist_gorgo11_sequ_collapsed_emmixt_KL_flat = np.array(data_pbs.dict_arrays['result_dist_gorgo11_sequ_collapsed_emmixt_KL']['results_flat'])

    result_parameters_flat = np.array(data_pbs.dict_arrays['result_em_fits_collapsed_tr']['parameters_flat'])
    all_repeats_completed = data_pbs.dict_arrays['result_em_fits_collapsed_tr']['repeats_completed']

    all_args_arr = np.array(data_pbs.loaded_data['args_list'])

    M_space = data_pbs.loaded_data['parameters_uniques']['M']
    ratio_conj_space = data_pbs.loaded_data['parameters_uniques']['ratio_conj']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    alpha_space = data_pbs.loaded_data['parameters_uniques']['alpha']

    num_repetitions = generator_module.num_repetitions
    parameter_names_sorted = data_pbs.dataset_infos['parameters']

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    # Load ground truth
    data_gorgo11_sequ = load_experimental_data.load_data_gorgo11_sequential(fit_mixture_model=True)

    ## Filter everything with repeats_completed == num_repet
    filter_data = all_repeats_completed == num_repetitions - 1
    result_parameters_flat = result_parameters_flat[filter_data]

    result_em_fits_collapsed_tr_flat = result_em_fits_collapsed_tr_flat[filter_data]
    result_em_fits_collapsed_summary_flat = result_em_fits_collapsed_summary_flat[filter_data]
    result_dist_gorgo11_sequ_collapsed_flat = result_dist_gorgo11_sequ_collapsed_flat[filter_data]
    result_dist_gorgo11_sequ_collapsed_emmixt_KL_flat = result_dist_gorgo11_sequ_collapsed_emmixt_KL_flat[filter_data]

    all_args_arr = all_args_arr[filter_data]
    all_repeats_completed = all_repeats_completed[filter_data]

    print "Size post-filter: ", result_parameters_flat.shape[0]

    # Compute lots of averages over the repetitions
    result_em_fits_collapsed_tr_flat_avg = utils.nanmean(result_em_fits_collapsed_tr_flat, axis=-1)
    result_em_fits_collapsed_summary_flat_avg = utils.nanmean(result_em_fits_collapsed_summary_flat, axis=-1)
    result_dist_gorgo11_sequ_collapsed_flat_avg = utils.nanmean(result_dist_gorgo11_sequ_collapsed_flat, axis=-1)
    result_dist_gorgo11_sequ_collapsed_emmixt_KL_flat_avg = utils.nanmean(result_dist_gorgo11_sequ_collapsed_emmixt_KL_flat, axis=-1)

    result_dist_gorgo11_sequ_collapsed_flat_avg_overall = np.nansum(np.nansum(np.nansum(result_dist_gorgo11_sequ_collapsed_flat_avg, axis=-1), axis=-1), axis=-1)
    # We will now grid some of the parameters, to have a 2D/3D surface back.
    # Let's fix the ratio_conj, as we know that the other models need around
    # ratio =0.8 to fit data well.

    def str_best_params(best_i, result_dist_to_use):
        return ' '.join(["%s %.4f" % (parameter_names_sorted[param_i], result_parameters_flat[best_i, param_i]) for param_i in xrange(len(parameter_names_sorted))]) + ' >> %f' % result_dist_to_use[best_i]

    ###### Best fitting points
    if plots_bestfits:
        nb_best_points = 5

        def plot_collapsed_modelfits(T_space, curr_result_emfits_collapsed_tr, labelplot='', dataio=None):
            f, ax = plt.subplots()
            for nitems_i, nitems in enumerate(T_space):
                ax = plots_experimental_data.plot_kappa_mean_error(T_space[:nitems], curr_result_emfits_collapsed_tr[..., 0][nitems_i, :nitems], 0.0*curr_result_emfits_collapsed_tr[..., 0][nitems_i, :nitems], title='model fit fig7 %s' % labelplot , ax=ax, label='%d items' % nitems, xlabel='T_recall')

            if dataio is not None:
                dataio.save_current_figure('bestfit_doublepowerlaw_%s_kappa_{label}_{unique_id}.pdf' % labelplot)

            _, ax_target = plt.subplots()
            _, ax_nontarget = plt.subplots()
            _, ax_random = plt.subplots()
            for nitems_i, nitems in enumerate(T_space):
                ax_target = plots_experimental_data.plot_emmixture_mean_error(T_space[:nitems], curr_result_emfits_collapsed_tr[..., 1][nitems_i, :nitems], curr_result_emfits_collapsed_tr[..., 1][nitems_i, :nitems]*0.0, title='Target model fit %s' % labelplot, ax=ax_target, label='%d items' % nitems, xlabel='T_recall')
                ax_nontarget = plots_experimental_data.plot_emmixture_mean_error(T_space[:nitems], curr_result_emfits_collapsed_tr[..., 2][nitems_i, :nitems], curr_result_emfits_collapsed_tr[..., 2][nitems_i, :nitems]*0.0, title='Nontarget model fit %s' % labelplot, ax=ax_nontarget, label='%d items' % nitems, xlabel='T_recall')
                ax_random = plots_experimental_data.plot_emmixture_mean_error(T_space[:nitems], curr_result_emfits_collapsed_tr[..., 3][nitems_i, :nitems], curr_result_emfits_collapsed_tr[..., 3][nitems_i, :nitems]*0.0, title='Random model fit %s' % labelplot, ax=ax_random, label='%d items' % nitems, xlabel='T_recall')

            if dataio is not None:
                plt.figure(ax_target.get_figure().number)
                dataio.save_current_figure('bestfit_doublepowerlaw_%s_mixttarget_{label}_{unique_id}.pdf' % labelplot)

                plt.figure(ax_nontarget.get_figure().number)
                dataio.save_current_figure('bestfit_doublepowerlaw_%s_mixtnontarget_{label}_{unique_id}.pdf' % labelplot)

                plt.figure(ax_random.get_figure().number)
                dataio.save_current_figure('bestfit_doublepowerlaw_%s_mixtrandom_{label}_{unique_id}.pdf' % labelplot)


        best_points_result_dist_gorgo11seq_all = np.argsort(result_dist_gorgo11_sequ_collapsed_flat_avg_overall)[:nb_best_points]

        for best_point_i in best_points_result_dist_gorgo11seq_all:
            plot_collapsed_modelfits(T_space, result_em_fits_collapsed_tr_flat_avg[best_point_i], labelplot='%.1f' % result_dist_gorgo11_sequ_collapsed_flat_avg_overall[best_point_i], dataio=dataio)


    ###### 3D scatter plots
    if plots_scatter3d:
        nb_best_points = 30
        size_normal_points = 8
        size_best_points = 50

        def plot_scatter(all_vars, result_dist_to_use_name, title='', log_color=True, downsampling=1, label_file=''):

            result_dist_to_use = all_vars[result_dist_to_use_name]
            result_parameters_flat_3d = all_vars['result_parameters_flat_3d']

            # Filter if downsampling
            filter_downsampling = np.arange(0, result_dist_to_use.size, downsampling)
            result_dist_to_use = result_dist_to_use[filter_downsampling]
            result_parameters_flat_3d = result_parameters_flat_3d[filter_downsampling]

            best_points_result_dist_to_use = np.argsort(result_dist_to_use)[:nb_best_points]

            # Construct all permutations of 3 parameters, for 3D scatters
            params_permutations = set([tuple(np.sort(np.random.choice(result_parameters_flat_3d.shape[-1], 3, replace=False)).tolist()) for i in xrange(1000)])

            for param_permut in params_permutations:
                fig = plt.figure()
                ax = Axes3D(fig)

                # One plot per parameter permutation
                if log_color:
                    color_points = np.log(result_dist_to_use)
                else:
                    color_points = result_dist_to_use

                utils.scatter3d(result_parameters_flat_3d[:, param_permut[0]], result_parameters_flat_3d[:, param_permut[1]], result_parameters_flat_3d[:, param_permut[2]], s=size_normal_points, c=color_points, xlabel=parameter_names_sorted[param_permut[0]], ylabel=parameter_names_sorted[param_permut[1]], zlabel=parameter_names_sorted[param_permut[2]], title=title, ax_handle=ax)

                utils.scatter3d(result_parameters_flat_3d[best_points_result_dist_to_use, param_permut[0]], result_parameters_flat_3d[best_points_result_dist_to_use, param_permut[1]], result_parameters_flat_3d[best_points_result_dist_to_use, param_permut[2]], c='r', s=size_best_points, ax_handle=ax)

                if savefigs:
                    dataio.save_current_figure('scatter3d_%s_%s%s_{label}_{unique_id}.pdf' % (result_dist_to_use_name, '_'.join([parameter_names_sorted[i] for i in param_permut]), label_file))

                if savemovies:
                    try:
                        utils.rotate_plot3d(ax, dataio.create_formatted_filename('scatter3d_%s_%s%s_{label}_{unique_id}.mp4' % (result_dist_to_use_name, '_'.join([parameter_names_sorted[i] for i in param_permut]), label_file)), bitrate=8000, min_duration=8)
                        utils.rotate_plot3d(ax, dataio.create_formatted_filename('scatter3d_%s_%s%s_{label}_{unique_id}.gif' % (result_dist_to_use_name, '_'.join([parameter_names_sorted[i] for i in param_permut]), label_file)), nb_frames=30, min_duration=8)
                    except Exception:
                        # Most likely wrong aggregator...
                        print "failed when creating movies for ", result_dist_to_use_name


                if False and savefigs:
                    ax.view_init(azim=90, elev=10)
                    dataio.save_current_figure('scatter3d_view2_%s_%s%s_{label}_{unique_id}.pdf' % (result_dist_to_use_name, '_'.join([parameter_names_sorted[i] for i in param_permut]), label_file))

                # plt.close('all')

            print "Parameters: %s" % ', '.join(parameter_names_sorted)
            print "Best points, %s:" % title
            print '\n'.join([str_best_params(best_i, result_dist_to_use) for best_i in best_points_result_dist_to_use])


    # all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['parameter_names_sorted', 'all_args_arr', 'all_repeats_completed', 'filter_data']

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)
        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='gorgo11_sequential_fitmixturemodel')


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
                     construct_multidimension_npyarr=False,
                     # limit_max_files=1,
                     parameters=['M', 'ratio_conj', 'sigmax', 'alpha'],
                     variables_to_load=['result_em_fits_collapsed_tr', 'result_em_fits_collapsed_summary', 'result_dist_gorgo11_sequ_collapsed', 'result_dist_gorgo11_sequ_collapsed_emmixt_KL'],
                     variables_description=['Collapsed em fits, all trecall', 'Collapsed em fits, summary stats', 'Fit to gorgo11 seq collapsed em fits', 'Fit to gorgo11 seq collapsed emmixt KL fit'],
                     post_processing=None,
                     save_output_filename='plots_sequentialgorgo11_fitmixturemodel_Mratiosigmaxalphatrecall'
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
