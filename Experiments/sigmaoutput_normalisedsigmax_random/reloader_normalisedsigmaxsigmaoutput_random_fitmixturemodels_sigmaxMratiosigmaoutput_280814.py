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

# import matplotlib.animation as plt_anim
from mpl_toolkits.mplot3d import Axes3D

import re
import inspect
import imp

import utils
# import submitpbs
import load_experimental_data


def plots_fit_mixturemodels_random(data_pbs, generator_module=None):
    '''
        Reload runs from PBS
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True
    savemovies = False

    do_bays09 = True
    do_gorgo11 = True

    plots_scatter3d = True
    plots_scatter_per_T = False
    plots_flat_sorted_performance = False
    plots_memorycurves_fits_best = True

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
    all_repeats_completed = data_pbs.dict_arrays['result_em_fits']['repeats_completed']

    all_args = data_pbs.loaded_data['args_list']
    all_args_arr = np.array(all_args)

    sigmaoutput_space = data_pbs.loaded_data['parameters_uniques']['sigma_output']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    ratio_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    num_repetitions = generator_module.num_repetitions
    parameter_names_sorted = data_pbs.dataset_infos['parameters']

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    # Load bays09
    # data_bays09 = load_experimental_data.load_data_bays09(fit_mixture_model=True)
    # bays09_nitems = data_bays09['data_to_fit']['n_items']
    # bays09_em_target = np.nan*np.empty((bays09_nitems.max(), 4))  #kappa, prob_target, prob_nontarget, prob_random
    # bays09_em_target[bays09_nitems - 1] = data_bays09['em_fits_nitems_arrays']['mean'].T
    # bays09_emmixt_target = bays09_em_target[:, 1:]

    ## Filter everything with sigma_output > 1.0 and repeats_completed == num_repet
    filter_data = (result_parameters_flat[:, -1] < 1.0) & (all_repeats_completed == num_repetitions - 1)
    result_em_fits_flat = result_em_fits_flat[filter_data]
    result_precisions_flat = result_precisions_flat[filter_data]
    result_dist_bays09_flat = result_dist_bays09_flat[filter_data]
    result_dist_gorgo11_flat = result_dist_gorgo11_flat[filter_data]
    result_dist_bays09_emmixt_KL = result_dist_bays09_emmixt_KL[filter_data]
    result_dist_gorgo11_emmixt_KL = result_dist_gorgo11_emmixt_KL[filter_data]
    result_parameters_flat = result_parameters_flat[filter_data]
    all_args_arr = all_args_arr[filter_data]
    all_repeats_completed = all_repeats_completed[filter_data]

    # for _result_dist in ['result_em_fits_flat', 'result_precisions_flat', 'result_dist_bays09_flat', 'result_dist_gorgo11_flat', 'result_dist_bays09_emmixt_KL', 'result_dist_gorgo11_emmixt_KL', 'result_parameters_flat']:
    #     locals()[key] = locals()[key][filter_sigmaout]
    #     # exec("%s = %s[%s]" % (_result_dist, _result_dist, 'filter_sigmaout'))

    ## Compute some stuff

    result_em_fits_all_avg = utils.nanmean(result_em_fits_flat, axis=-1)
    result_em_kappa_allT = result_em_fits_all_avg[..., 0]
    result_em_emmixt_allT = result_em_fits_all_avg[..., 1:4]

    result_precisions_all_avg = utils.nanmean(result_precisions_flat, axis=-1)

    ##### Distance to Bays09
    result_dist_bays09_allT_avg = utils.nanmean(result_dist_bays09_flat, axis=-1)
    result_dist_bays09_emmixt_KL_allT_avg = utils.nanmean(result_dist_bays09_emmixt_KL, axis=-1)
    result_dist_bays09_kappa_allT = result_dist_bays09_allT_avg[..., 0]

    result_dist_bays09_kappa_sumT = np.nansum(result_dist_bays09_kappa_allT, axis=-1)
    result_dist_bays09_logkappa_sumT = np.log(result_dist_bays09_kappa_sumT)
    result_dist_bays09_emmixt_KL_sumT = np.nansum(result_dist_bays09_emmixt_KL_allT_avg, axis=-1)

    # combined versions
    result_dist_bays09_both_normalised = result_dist_bays09_emmixt_KL_sumT/np.max(result_dist_bays09_emmixt_KL_sumT) + result_dist_bays09_kappa_sumT/np.max(result_dist_bays09_kappa_sumT)
    result_dist_bays09_logkappamixtKL = result_dist_bays09_logkappa_sumT + result_dist_bays09_emmixt_KL_sumT
    result_dist_bays09_logkappamixtKL_normalised = result_dist_bays09_logkappa_sumT/np.max(result_dist_bays09_logkappa_sumT) + result_dist_bays09_emmixt_KL_sumT/np.max(result_dist_bays09_emmixt_KL_sumT)

    result_dist_bays09_logkappa_sumT_forand = result_dist_bays09_logkappa_sumT - np.min(result_dist_bays09_logkappa_sumT)*np.sign(np.min(result_dist_bays09_logkappa_sumT))
    result_dist_bays09_logkappa_sumT_forand /= np.max(result_dist_bays09_logkappa_sumT_forand)

    result_dist_bays09_emmixt_KL_sumT_forand = result_dist_bays09_emmixt_KL_sumT - np.min(result_dist_bays09_emmixt_KL_sumT)*np.sign(np.min(result_dist_bays09_emmixt_KL_sumT))
    result_dist_bays09_emmixt_KL_sumT_forand /= np.max(result_dist_bays09_emmixt_KL_sumT_forand)

    result_dist_bays09_logkappamixtKL_AND = 1. - (1. - result_dist_bays09_logkappa_sumT_forand)*(1. - result_dist_bays09_emmixt_KL_sumT_forand)

    # Mask kappa for bad performance
    # result_dist_bays09_kappa_sumT_masked = np.ma.masked_greater(result_dist_bays09_kappa_sumT, 2*np.median(result_dist_bays09_kappa_sumT))
    # result_dist_bays09_logkappa_sumT_masked = np.ma.masked_greater(result_dist_bays09_logkappa_sumT, 2*np.median(result_dist_bays09_logkappa_sumT))
    # result_dist_bays09_emmixt_KL_sumT_masked = np.ma.masked_greater(result_dist_bays09_emmixt_KL_sumT, 2*np.median(result_dist_bays09_emmixt_KL_sumT))
    # result_dist_bays09_both_normalised_mult_masked = 1-(1. - result_dist_bays09_emmixt_KL_sumT_masked/np.max(result_dist_bays09_emmixt_KL_sumT_masked))*(1. - result_dist_bays09_kappa_sumT_masked/np.max(result_dist_bays09_kappa_sumT_masked))

    ##### Distance to Gorgo11
    result_dist_gorgo11_allT_avg = utils.nanmean(result_dist_gorgo11_flat, axis=-1)
    result_dist_gorgo11_emmixt_KL_allT_avg = utils.nanmean(result_dist_gorgo11_emmixt_KL, axis=-1)
    result_dist_gorgo11_kappa_allT = result_dist_gorgo11_allT_avg[..., 0]

    result_dist_gorgo11_kappa_sumT = np.nansum(result_dist_gorgo11_kappa_allT, axis=-1)
    result_dist_gorgo11_logkappa_sumT = np.log(result_dist_gorgo11_kappa_sumT)
    result_dist_gorgo11_emmixt_KL_sumT = np.nansum(result_dist_gorgo11_emmixt_KL_allT_avg, axis=-1)
    result_dist_gorgo11_emmixt_KL_sumT25 = np.nansum(result_dist_gorgo11_emmixt_KL_allT_avg[:, 1:], axis=-1)
    result_dist_gorgo11_logkappa_sumT25 = np.log(np.nansum(result_dist_gorgo11_kappa_allT[..., 1:], axis=-1))

    # combined versions
    result_dist_gorgo11_both_normalised = result_dist_gorgo11_emmixt_KL_sumT/np.max(result_dist_gorgo11_emmixt_KL_sumT) + result_dist_gorgo11_kappa_sumT/np.max(result_dist_gorgo11_kappa_sumT)
    result_dist_gorgo11_logkappamixtKL = result_dist_gorgo11_logkappa_sumT + result_dist_gorgo11_emmixt_KL_sumT
    result_dist_gorgo11_logkappamixtKL_normalised = result_dist_gorgo11_logkappa_sumT/np.max(result_dist_gorgo11_logkappa_sumT) + result_dist_gorgo11_emmixt_KL_sumT/np.max(result_dist_gorgo11_emmixt_KL_sumT)

    result_dist_gorgo11_logkappa_sumT_forand = result_dist_gorgo11_logkappa_sumT - np.min(result_dist_gorgo11_logkappa_sumT)*np.sign(np.min(result_dist_gorgo11_logkappa_sumT))
    result_dist_gorgo11_logkappa_sumT_forand /= np.max(result_dist_gorgo11_logkappa_sumT_forand)


    result_dist_gorgo11_logkappa_sumT25_forand = result_dist_gorgo11_logkappa_sumT25 - np.min(result_dist_gorgo11_logkappa_sumT25)*np.sign(np.min(result_dist_gorgo11_logkappa_sumT25))
    result_dist_gorgo11_logkappa_sumT25_forand /= np.max(result_dist_gorgo11_logkappa_sumT25_forand)

    result_dist_gorgo11_emmixt_KL_sumT_forand = result_dist_gorgo11_emmixt_KL_sumT - np.min(result_dist_gorgo11_emmixt_KL_sumT)*np.sign(np.min(result_dist_gorgo11_emmixt_KL_sumT))
    result_dist_gorgo11_emmixt_KL_sumT_forand /= np.max(result_dist_gorgo11_emmixt_KL_sumT_forand)

    result_dist_gorgo11_emmixt_KL_sumT25_forand = result_dist_gorgo11_emmixt_KL_sumT25 - np.min(result_dist_gorgo11_emmixt_KL_sumT25)*np.sign(np.min(result_dist_gorgo11_emmixt_KL_sumT25))
    result_dist_gorgo11_emmixt_KL_sumT25_forand /= np.max(result_dist_gorgo11_emmixt_KL_sumT25_forand)

    result_dist_gorgo11_logkappamixtKL_AND = 1. - (1. - result_dist_gorgo11_logkappa_sumT_forand)*(1. - result_dist_gorgo11_emmixt_KL_sumT_forand)

    result_dist_gorgo11_logkappa25mixtKL_AND = 1. - (1. - result_dist_gorgo11_logkappa_sumT25_forand)*(1. - result_dist_gorgo11_emmixt_KL_sumT25_forand)

    def str_best_params(best_i, result_dist_to_use):
        return ' '.join(["%s %.4f" % (parameter_names_sorted[param_i], result_parameters_flat[best_i, param_i]) for param_i in xrange(len(parameter_names_sorted))]) + ' >> %f' % result_dist_to_use[best_i]

    if plots_scatter3d:
        nb_best_points = 30
        size_normal_points = 8
        size_best_points = 50

        def plot_scatter(all_vars, result_dist_to_use_name, title='', log_color=True, downsampling=1, label_file=''):

            result_dist_to_use = all_vars[result_dist_to_use_name]
            result_parameters_flat = all_vars['result_parameters_flat']

            # Filter if downsampling
            filter_downsampling = np.arange(0, result_dist_to_use.size, downsampling)
            result_dist_to_use = result_dist_to_use[filter_downsampling]
            result_parameters_flat = result_parameters_flat[filter_downsampling]

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



        #### BAYS 09
        if do_bays09:
            # Distance for log kappa, all T
            plot_scatter(locals(), 'result_dist_bays09_logkappa_sumT', 'Bays09 kappa all T', log_color=False)

            # # Distance for em fits, all T, KL distance
            plot_scatter(locals(), 'result_dist_bays09_emmixt_KL_sumT', 'Bays09 em fits, sum T, KL', log_color=False)

            # Distance for product of normalised em fits KL + normalised log kappa, all T
            plot_scatter(locals(), 'result_dist_bays09_logkappamixtKL', 'Bays09 em fits KL, log kappa')

            # Distance for AND normalised em fits KL + log kappa
            plot_scatter(locals(), 'result_dist_bays09_logkappamixtKL_AND', 'Bays09 em fits KL AND log kappa')

        #### Gorgo 11
        if do_gorgo11:
            # Distance for product of normalised em fits KL + normalised log kappa, all T
            plot_scatter(locals(), 'result_dist_gorgo11_logkappamixtKL', 'Gorgo11 em fits KL, log kappa')

            # Distance for AND normalised em fits KL + log kappa
            plot_scatter(locals(), 'result_dist_gorgo11_logkappamixtKL_AND', 'Gorgo11 em fits KL AND log kappa')

            # Distance for logkappa
            plot_scatter(locals(), 'result_dist_gorgo11_logkappa_sumT', 'Gorgo11 log kappa all T', log_color=False)

            # Distance for EM mixture proportions
            plot_scatter(locals(), 'result_dist_gorgo11_emmixt_KL_sumT', 'Gorgo11 em fits, sum T, KL', log_color=False)



    if plots_flat_sorted_performance:
        result_dist_to_try = []
        if do_bays09:
            result_dist_to_try.extend(['result_dist_bays09_logkappamixtKL_AND', 'result_dist_bays09_logkappamixtKL'])
        if do_gorgo11:
            result_dist_to_try.extend(['result_dist_gorgo11_logkappamixtKL_AND', 'result_dist_gorgo11_logkappamixtKL'])

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

        ## Extra plot for logkappamixtKL_AND, it seems well behaved

        def plot_flat_best(all_vars, result_name, order_indices_filter, filter_goodAND, ordering='fitness'):
            f = plt.figure()
            axp1 = plt.subplot2grid((3, 2), (0, 0))
            axp2 = plt.subplot2grid((3, 2), (0, 1))
            axp3 = plt.subplot2grid((3, 2), (1, 0))
            axp4 = plt.subplot2grid((3, 2), (1, 1))
            axfit = plt.subplot2grid((3, 2), (2, 0), colspan=2)

            axp1.plot(result_parameters_flat[filter_goodAND][order_indices_filter, 0])
            axp1.set_title(parameter_names_sorted[0])
            axp2.plot(result_parameters_flat[filter_goodAND][order_indices_filter, 1], 'g')
            axp2.set_title(parameter_names_sorted[1])
            axp3.plot(result_parameters_flat[filter_goodAND][order_indices_filter, 2], 'r')
            axp3.set_title(parameter_names_sorted[2])
            axp4.plot(result_parameters_flat[filter_goodAND][order_indices_filter, 3], 'k')
            axp4.set_title(parameter_names_sorted[3])

            axfit.plot(all_vars[result_name][filter_goodAND][order_indices_filter])
            axfit.set_ylabel('bays09_logkappamixtKL_AND')
            plt.suptitle('Distance ordered bays09_logkappamixtKL_AND')

            if savefigs:
                dataio.save_current_figure('plot_sortedperf_best_%s_%s_{label}_{unique_id}.pdf' % (result_name, ordering))

        if do_bays09:
            filter_goodAND = result_dist_bays09_logkappamixtKL_AND < 0.2

            # First order them by fitness
            order_indices_filter = np.argsort(result_dist_bays09_logkappamixtKL_AND[filter_goodAND])[::-1]
            plot_flat_best(locals(), 'result_dist_bays09_logkappamixtKL_AND', order_indices_filter, filter_goodAND, 'fitness')

            # Then by M, to see if there is some structure
            order_indices_filter = np.argsort(result_parameters_flat[filter_goodAND, 0])
            plot_flat_best(locals(), 'result_dist_bays09_logkappamixtKL_AND', order_indices_filter, filter_goodAND, 'M')

        if do_gorgo11:
            filter_goodAND = result_dist_gorgo11_logkappamixtKL_AND < 0.5

            # First order them by fitness
            order_indices_filter = np.argsort(result_dist_gorgo11_logkappamixtKL_AND[filter_goodAND])[::-1]
            plot_flat_best(locals(), 'result_dist_gorgo11_logkappamixtKL_AND', order_indices_filter, filter_goodAND, 'fitness')

            # Then by M, to see if there is some structure
            order_indices_filter = np.argsort(result_parameters_flat[filter_goodAND, 0])
            plot_flat_best(locals(), 'result_dist_gorgo11_logkappamixtKL_AND', order_indices_filter, filter_goodAND, 'M')

            # dist_cmaes_result = np.sum((result_parameters_flat - np.array([75, 1.0, 0.1537, 0.2724]))**2., axis=-1)
            # filter_close_cmaes_result = np.argsort(dist_cmaes_result)[:20]
            # order_indices_filter = np.argsort(result_dist_gorgo11_logkappamixtKL_AND[filter_close_cmaes_result])[::-1]
            # plot_flat_best(locals(), 'result_dist_gorgo11_logkappamixtKL_AND', order_indices_filter, filter_close_cmaes_result, 'Like current CMA/ES run')


    if plots_scatter_per_T:
        for T_i, T in enumerate(T_space):

            # Kappa per T, fit to Bays09
            result_dist_bays09_kappa_currT = result_dist_bays09_kappa_allT[:, T_i]
            result_dist_bays09_kappa_currT_masked = mask_outliers(result_dist_bays09_kappa_currT)

            plot_scatter(locals(), 'result_dist_bays09_kappa_currT_masked', 'kappa T %d masked' % T, label_file="T{}".format(T))

            # EM Mixt per T, fit to Bays09
            result_dist_bays09_emmixt_KL_currT = result_dist_bays09_emmixt_KL_allT_avg[:, T_i]
            result_dist_bays09_emmixt_KL_currT_masked = mask_outliers(result_dist_bays09_emmixt_KL_currT)

            plot_scatter(locals(), 'result_dist_bays09_emmixt_KL_currT_masked', 'KL EM mixt T %d masked' % T, label_file="T{}".format(T), log_color=False)


    if plots_memorycurves_fits_best:

        data_dir = None
        if not os.environ.get('WORKDIR_DROP'):
            data_dir = '../experimental_data/'

        plotting_parameters = launchers_memorycurves_marginal_fi.load_prepare_datasets(data_dir = data_dir)

        def plot_memorycurves_fits(all_vars, result_dist_to_use_name, nb_best_points=10):
            result_dist_to_use = all_vars[result_dist_to_use_name]

            best_points_result_dist_to_use = np.argsort(result_dist_to_use)[:nb_best_points]

            for best_point_index in best_points_result_dist_to_use:
                print "extended plot for: " + str_best_params(best_point_index, result_dist_to_use)

                # Update arguments
                all_args_arr[best_point_index].update(dict(zip(parameter_names_sorted, result_parameters_flat[best_point_index])))
                packed_data = dict(T_space=T_space, result_em_fits=result_em_fits_flat[best_point_index], all_parameters=all_args_arr[best_point_index])

                plotting_parameters['suptitle'] = result_dist_to_use_name
                plotting_parameters['reuse_axes'] = False
                if savefigs:
                    packed_data['dataio'] = dataio

                launchers_memorycurves_marginal_fi.do_memory_plots(packed_data, plotting_parameters)


        plot_memorycurves_fits(locals(), 'result_dist_bays09_logkappamixtKL_AND', nb_best_points=3)

        plot_memorycurves_fits(locals(), 'result_dist_gorgo11_logkappamixtKL_AND', nb_best_points=3)
        # plot_memorycurves_fits(locals(), 'result_dist_gorgo11_logkappamixtKL', nb_best_points=3)

        plot_memorycurves_fits(locals(), 'result_dist_gorgo11_logkappa25mixtKL_AND', nb_best_points=3)

        # plot_memorycurves_fits(locals(), 'result_dist_gorgo11_logkappa_sumT', nb_best_points=3)




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



    # all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['parameter_names_sorted', 'all_args_arr', 'all_repeats_completed', 'filter_data']

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
dataset_infos = dict(label='Fitting of experimental data. All experiments. Random sampling of parameter space. Perhaps too big, be careful...',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['M', 'ratio_conj', 'sigmax', 'sigma_output'],
                     variables_to_load=['result_em_fits', 'result_em_fits_allnontargets', 'result_dist_bays09', 'result_dist_gorgo11', 'result_dist_bays09_emmixt_KL', 'result_dist_gorgo11_emmixt_KL', 'result_all_precisions'],
                     variables_description=['EM fits, using all nontargets (may be wrong)', 'EM fits, with nontargets mixtures', 'Distance of EM fits to Bays09 dataset (kappa, mixtures)', 'Distance to Gorgo11 dataset (kappa, mixtures)', 'KL bays09 fits', 'KL gorgo11 fits'],
                     post_processing=plots_fit_mixturemodels_random,
                     save_output_filename='plots_fitmixtmodel_random_sigmaoutsigmaxnormMratio'
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
