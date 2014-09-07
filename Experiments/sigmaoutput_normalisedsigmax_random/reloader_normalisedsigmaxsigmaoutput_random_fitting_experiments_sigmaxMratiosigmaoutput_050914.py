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
    savefigs = False
    savedata = True
    savemovies = False

    do_bays09 = True
    do_gorgo11 = True

    scatter3d_sumT = False
    plots_flat_sorted_performance = False
    plots_memorycurves_fits_best = True

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

    all_repeats_completed = data_pbs.dict_arrays['result_fitexperiments']['repeats_completed']
    all_args = data_pbs.loaded_data['args_list']
    all_args_arr = np.array(all_args)
    num_repetitions = generator_module.num_repetitions

    # Extract order of datasets
    experiment_ids = data_pbs.loaded_data['datasets_list'][0]['fitexperiment_parameters']['experiment_ids']
    parameter_names_sorted = data_pbs.dataset_infos['parameters']

    T_space = data_pbs.loaded_data['datasets_list'][0]['T_space']

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    # filter_data = (result_parameters_flat[:, -1] < 1.0) & (all_repeats_completed == num_repetitions - 1)
    # filter_data = (all_repeats_completed == num_repetitions - 1)
    # result_fitexperiments_flat = result_fitexperiments_flat[filter_data]
    # result_fitexperiments_all_flat = result_fitexperiments_all_flat[filter_data]
    # result_fitexperiments_noiseconv_flat = result_fitexperiments_noiseconv_flat[filter_data]
    # result_fitexperiments_noiseconv_all_flat = result_fitexperiments_noiseconv_all_flat[filter_data]
    # result_parameters_flat = result_parameters_flat[filter_data]

    # Compute some stuff
    # Data is summed over all experiments for _flat, contains bic, ll and ll90.
    # for _all_flat, contains bic, ll and ll90 per experiment. Given that Gorgo11 and Bays09 are incompatible, shouldn't really use the combined version directly!
    result_fitexperiments_noiseconv_bic_avg_allT = utils.nanmean(result_fitexperiments_noiseconv_flat, axis=-1)[..., 0]
    result_fitexperiments_noiseconv_allexp_bic_avg_allT = utils.nanmean(result_fitexperiments_noiseconv_all_flat, axis=-1)[:, :, 0]
    result_fitexperiments_noiseconv_allexp_ll90_avg_allT = -utils.nanmean(result_fitexperiments_noiseconv_all_flat, axis=-1)[:, :, -1]

    ### BIC
    # result_fitexperiments_noiseconv_allexp_bic_avg_allT: N x T x exp
    result_fitexperiments_noiseconv_bays09_bic_avg_allT = result_fitexperiments_noiseconv_allexp_bic_avg_allT[..., 0]
    result_fitexperiments_noiseconv_gorgo11_bic_avg_allT = result_fitexperiments_noiseconv_allexp_bic_avg_allT[..., 1]
    result_fitexperiments_noiseconv_dualrecall_bic_avg_allT = result_fitexperiments_noiseconv_allexp_bic_avg_allT[..., 2]
    # Summed T
    result_fitexperiments_noiseconv_bays09_bic_avg_sumT = np.nansum(result_fitexperiments_noiseconv_bays09_bic_avg_allT, axis=-1)
    result_fitexperiments_noiseconv_gorgo11_bic_avg_sumT = np.nansum(result_fitexperiments_noiseconv_gorgo11_bic_avg_allT, axis=-1)
    result_fitexperiments_noiseconv_dualrecall_bic_avg_sumT = np.nansum(result_fitexperiments_noiseconv_dualrecall_bic_avg_allT, axis=-1)

    ### LL90
    # N x T x exp
    result_fitexperiments_noiseconv_bays09_ll90_avg_allT = result_fitexperiments_noiseconv_allexp_ll90_avg_allT[..., 0]
    result_fitexperiments_noiseconv_gorgo11_ll90_avg_allT = result_fitexperiments_noiseconv_allexp_ll90_avg_allT[..., 1]
    result_fitexperiments_noiseconv_dualrecall_ll90_avg_allT = result_fitexperiments_noiseconv_allexp_ll90_avg_allT[..., 2]
    # Summed T
    result_fitexperiments_noiseconv_bays09_ll90_avg_sumT = np.nansum(result_fitexperiments_noiseconv_bays09_ll90_avg_allT, axis=-1)
    result_fitexperiments_noiseconv_gorgo11_ll90_avg_sumT = np.nansum(result_fitexperiments_noiseconv_gorgo11_ll90_avg_allT, axis=-1)
    result_fitexperiments_noiseconv_dualrecall_ll90_avg_sumT = np.nansum(result_fitexperiments_noiseconv_dualrecall_ll90_avg_allT, axis=-1)

    def mask_outliers_array(result_dist_to_use, sigma_outlier=3):
        '''
            Mask outlier datapoints.
            Compute the mean of the results and assume that points with:
              result > mean + sigma_outlier*std
            are outliers.

            As we want the minimum values, do not mask small values
        '''
        return np.ma.masked_greater(result_dist_to_use, np.mean(result_dist_to_use) + sigma_outlier*np.std(result_dist_to_use))

    def best_points_allT(result_dist_to_use):
        '''
            Best points for all T
        '''
        return np.argsort(result_dist_to_use)[:nb_best_points]

    def str_best_params(best_i, result_dist_to_use):
        return ' '.join(["%s %.4f" % (parameter_names_sorted[param_i], result_parameters_flat[best_i, param_i]) for param_i in xrange(len(parameter_names_sorted))]) + ' >> %f' % result_dist_to_use[best_i]

    def plot_scatter(all_vars, result_dist_to_use_name, title='', log_color=True, downsampling=1, label_file='', mask_outliers=True):

        result_dist_to_use = all_vars[result_dist_to_use_name]
        result_parameters_flat = all_vars['result_parameters_flat']

        # Filter if downsampling
        filter_downsampling = np.arange(0, result_dist_to_use.size, downsampling)
        result_dist_to_use = result_dist_to_use[filter_downsampling]
        result_parameters_flat = result_parameters_flat[filter_downsampling]

        if mask_outliers:
            result_dist_to_use = mask_outliers_array(result_dist_to_use)

        best_points_result_dist_to_use = np.argsort(result_dist_to_use)[:nb_best_points]

        # Construct all permutations of 3 parameters, for 3D scatters
        params_permutations = set([tuple(np.sort(np.random.choice(result_parameters_flat.shape[-1], 3, replace=False)).tolist()) for i in xrange(1000)])

        for param_permut in params_permutations:
            fig = plt.figure()
            ax = Axes3D(fig)

            # One plot per parameter permutation
            if log_color:
                color_points = np.log(result_dist_to_use)
            else:
                color_points = result_dist_to_use

            utils.scatter3d(result_parameters_flat[:, param_permut[0]], result_parameters_flat[:, param_permut[1]], result_parameters_flat[:, param_permut[2]], s=size_normal_points, c=color_points, xlabel=parameter_names_sorted[param_permut[0]], ylabel=parameter_names_sorted[param_permut[1]], zlabel=parameter_names_sorted[param_permut[2]], title=title, ax_handle=ax)

            utils.scatter3d(result_parameters_flat[best_points_result_dist_to_use, param_permut[0]], result_parameters_flat[best_points_result_dist_to_use, param_permut[1]], result_parameters_flat[best_points_result_dist_to_use, param_permut[2]], c='r', s=size_best_points, ax_handle=ax)

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



    if scatter3d_sumT:

        plot_scatter(locals(), 'result_fitexperiments_noiseconv_bays09_bic_avg_sumT', 'BIC Bays09')
        plot_scatter(locals(), 'result_fitexperiments_noiseconv_bays09_ll90_avg_sumT', 'LL90 Bays09')

        plot_scatter(locals(), 'result_fitexperiments_noiseconv_gorgo11_bic_avg_sumT', 'BIC Gorgo11')
        plot_scatter(locals(), 'result_fitexperiments_noiseconv_gorgo11_ll90_avg_sumT', 'LL90 Gorgo11')

        plot_scatter(locals(), 'result_fitexperiments_noiseconv_dualrecall_bic_avg_sumT', 'BIC Dual recall')
        plot_scatter(locals(), 'result_fitexperiments_noiseconv_dualrecall_ll90_avg_sumT', 'LL90 Dual recall')


    if plots_flat_sorted_performance:
        result_dist_to_try = []
        if do_bays09:
            result_dist_to_try.extend(['result_fitexperiments_noiseconv_bays09_bic_avg_sumT', 'result_fitexperiments_noiseconv_bays09_ll90_avg_sumT'])
        if do_gorgo11:
            result_dist_to_try.extend(['result_fitexperiments_noiseconv_gorgo11_bic_avg_sumT', 'result_fitexperiments_noiseconv_gorgo11_ll90_avg_sumT'])

        for result_dist in result_dist_to_try:
            order_indices = np.argsort(locals()[result_dist])[::-1]

            f, axes = plt.subplots(2, 1)
            axes[0].plot(np.arange(4) + result_parameters_flat[order_indices]/np.max(result_parameters_flat[order_indices], axis=0))
            axes[0].legend(parameter_names_sorted, loc='upper left')
            axes[0].set_ylabel('Parameters')
            axes[1].plot(locals()[result_dist][order_indices])
            axes[1].set_ylabel(result_dist.split('result_dist_')[-1])
            axes[0].set_title('Distance ordered ' + result_dist.split('result_dist_')[-1])
            f.canvas.draw()

            if savefigs:
                dataio.save_current_figure('plot_sortedperf_full_%s_{label}_{unique_id}.pdf' % (result_dist))

    if plots_memorycurves_fits_best:
        # Alright, will actually reload the data from another set of runs, and find the closest parameter set to the ones found here.
        data = utils.load_npy('normalisedsigmaxsigmaoutput_random_fitmixturemodels_sigmaxMratiosigmaoutput_repetitions3_280814/outputs/global_plots_fitmixtmodel_random_sigmaoutsigmaxnormMratio-plots_fit_mixturemodels_random-75eb9c74-72e0-4165-8014-92c1ef446f0a.npy')
        result_em_fits_flat_fitmixture = data['result_em_fits_flat']
        result_parameters_flat_fitmixture = data['result_parameters_flat']
        all_args_arr_fitmixture = data['all_args_arr']

        data_dir = None
        if not os.environ.get('WORKDIR_DROP'):
            data_dir = '../experimental_data/'

        plotting_parameters = launchers_memorycurves_marginal_fi.load_prepare_datasets(data_dir = data_dir)

        def plot_memorycurves_fits_fromexternal(all_vars, result_dist_to_use_name, nb_best_points=10):
            result_dist_to_use = all_vars[result_dist_to_use_name]

            result_em_fits_flat_fitmixture = all_vars['result_em_fits_flat_fitmixture']
            result_parameters_flat_fitmixture = all_vars['result_parameters_flat_fitmixture']
            all_args_arr_fitmixture = all_vars['all_args_arr_fitmixture']

            best_point_indices_result_dist = np.argsort(result_dist_to_use)[:nb_best_points]

            for best_point_index in best_point_indices_result_dist:
                print "extended plot desired for: " + str_best_params(best_point_index, result_dist_to_use)

                dist_best_points_fitmixture = np.abs(result_parameters_flat_fitmixture - result_parameters_flat[best_point_index])
                dist_best_points_fitmixture -= np.min(dist_best_points_fitmixture, axis=0)
                dist_best_points_fitmixture /= np.max(dist_best_points_fitmixture, axis=0)

                best_point_index_fitmixture = np.argmax(np.prod(1-dist_best_points_fitmixture, axis=-1))

                print "found closest: " + ' '.join(["%s %.4f" % (parameter_names_sorted[param_i], result_parameters_flat_fitmixture[best_point_index_fitmixture, param_i]) for param_i in xrange(len(parameter_names_sorted))])

                # Update arguments
                all_args_arr_fitmixture[best_point_index_fitmixture].update(dict(zip(parameter_names_sorted, result_parameters_flat_fitmixture[best_point_index_fitmixture])))
                packed_data = dict(T_space=T_space, result_em_fits=result_em_fits_flat_fitmixture[best_point_index_fitmixture], all_parameters=all_args_arr_fitmixture[best_point_index_fitmixture])

                plotting_parameters['suptitle'] = result_dist_to_use_name
                plotting_parameters['reuse_axes'] = False
                if savefigs:
                    packed_data['dataio'] = dataio

                launchers_memorycurves_marginal_fi.do_memory_plots(packed_data, plotting_parameters)


        plot_memorycurves_fits_fromexternal(locals(), 'result_fitexperiments_noiseconv_bays09_ll90_avg_sumT', nb_best_points=3)

        plot_memorycurves_fits_fromexternal(locals(), 'result_fitexperiments_noiseconv_gorgo11_ll90_avg_sumT', nb_best_points=3)

        plot_memorycurves_fits_fromexternal(locals(), 'result_fitexperiments_noiseconv_dualrecall_ll90_avg_sumT', nb_best_points=3)



    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['experiment_ids', 'parameter_names_sorted', 'T_space', 'all_args_arr', 'all_repeats_completed', 'filter_data']

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)
        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='sigmaoutput_normalisedsigmax_random')


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
                     parameters=['M', 'ratio_conj', 'sigmax', 'sigma_output'],
                     variables_to_load=['result_fitexperiments', 'result_fitexperiments_all', 'result_fitexperiments_noiseconv', 'result_fitexperiments_noiseconv_all'],
                     variables_description=['Fit experiments summed accross experiments', 'Fit experiments per experiment', 'Fit experiments with noise convolved posterior summed accross experiments', 'Fit experiments with noise convolved posterior'],
                     post_processing=plots_fitting_experiments_random,
                     save_output_filename='plots_fitexp_random_4d'
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

