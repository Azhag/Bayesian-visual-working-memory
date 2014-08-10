"""
    ExperimentDescriptor to fit Memory curves using a Mixed population code

    Uses the new Marginal Inverse Fisher Information, and some new code altogether.
"""

import os
import numpy as np
import experimentlauncher
import dataio as DataIO
import utils
import re
import imp
import matplotlib.pyplot as plt
import matplotlib
import launchers
import cPickle as pickle

import em_circularmixture
import em_circularmixture_allitems_uniquekappa
import em_circularmixture_allitems_kappafi

import inspect

import load_experimental_data

import scipy.interpolate as spint

# Commit @8c49507 +


def plots_ratioMscaling(data_pbs, generator_module=None):
    '''
        Reload and plot precision/fits of a Mixed code.
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plots_pcolor_all = False
    plots_effect_M_target_kappa = False

    plots_kappa_fi_comparison = False
    plots_multiple_fisherinfo = False
    specific_plot_effect_R = False
    specific_plot_effect_ratio_M = True

    convert_M_realsizes = True

    plots_pcolor_realsizes_Msubs = True
    plots_pcolor_realsizes_Mtot = True


    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    # interpolation_method = 'linear'
    interpolation_method = 'nearest'
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_all_precisions_mean = (utils.nanmean(data_pbs.dict_arrays['result_all_precisions']['results'], axis=-1))
    result_all_precisions_std = (utils.nanstd(data_pbs.dict_arrays['result_all_precisions']['results'], axis=-1))
    result_em_fits_mean = (utils.nanmean(data_pbs.dict_arrays['result_em_fits']['results'], axis=-1))
    result_em_fits_std = (utils.nanstd(data_pbs.dict_arrays['result_em_fits']['results'], axis=-1))
    result_fisherinfo_mean = (utils.nanmean(data_pbs.dict_arrays['result_fisher_info']['results'], axis=-1))
    result_fisherinfo_std = (utils.nanstd(data_pbs.dict_arrays['result_fisher_info']['results'], axis=-1))

    all_args = data_pbs.loaded_data['args_list']

    result_em_fits_kappa = result_em_fits_mean[..., 0]
    result_em_fits_target = result_em_fits_mean[..., 1]
    result_em_fits_kappa_valid = np.ma.masked_where(result_em_fits_target < 0.8, result_em_fits_kappa)

    # flat versions
    result_parameters_flat = np.array(data_pbs.dict_arrays['result_all_precisions']['parameters_flat'])
    result_all_precisions_mean_flat = np.mean(np.array(data_pbs.dict_arrays['result_all_precisions']['results_flat']), axis=-1)
    result_em_fits_mean_flat = np.mean(np.array(data_pbs.dict_arrays['result_em_fits']['results_flat']), axis=-1)
    result_fisherinfor_mean_flat = np.mean(np.array(data_pbs.dict_arrays['result_fisher_info']['results_flat']), axis=-1)
    result_em_fits_kappa_flat = result_em_fits_mean_flat[..., 0]
    result_em_fits_target_flat = result_em_fits_mean_flat[..., 1]
    result_em_fits_kappa_valid_flat = np.ma.masked_where(result_em_fits_target_flat < 0.8, result_em_fits_kappa_flat)



    M_space = data_pbs.loaded_data['parameters_uniques']['M'].astype(int)
    ratio_space = data_pbs.loaded_data['parameters_uniques']['ratio_conj']
    R_space = data_pbs.loaded_data['parameters_uniques']['R'].astype(int)
    num_repetitions = generator_module.num_repetitions
    T = generator_module.T

    print M_space
    print ratio_space
    print R_space
    print result_all_precisions_mean.shape, result_em_fits_mean.shape

    dataio = DataIO.DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    MAX_DISTANCE = 100.

    if convert_M_realsizes:
        # alright, currently M*ratio_conj gives the conjunctive subpopulation,
        # but only floor(M_conj**1/R) neurons are really used. So we should
        # convert to M_conj_real and M_feat_real instead of M and ratio
        result_parameters_flat_subM_converted = []
        result_parameters_flat_Mtot_converted = []

        for params in result_parameters_flat:
            M = params[0]; ratio_conj = params[1]; R = int(params[2])

            M_conj_prior = int(M*ratio_conj)
            M_conj_true = int(np.floor(M_conj_prior**(1./R))**R)
            M_feat_true = int(np.floor((M-M_conj_prior)/R)*R)

            # result_parameters_flat_subM_converted contains (M_conj, M_feat, R)
            result_parameters_flat_subM_converted.append(np.array([M_conj_true, M_feat_true, R]))
            # result_parameters_flat_Mtot_converted contains (M_tot, ratio_conj, R)
            result_parameters_flat_Mtot_converted.append(np.array([float(M_conj_true+M_feat_true), float(M_conj_true)/float(M_conj_true+M_feat_true), R]))

        result_parameters_flat_subM_converted = np.array(result_parameters_flat_subM_converted)
        result_parameters_flat_Mtot_converted = np.array(result_parameters_flat_Mtot_converted)

    if plots_pcolor_all:
        if convert_M_realsizes:
            def plot_interp(points, data, currR_indices, title='', points_label='', xlabel='', ylabel=''):
                utils.contourf_interpolate_data_interactive_maxvalue(points[currR_indices][..., :2], data[currR_indices], xlabel=xlabel, ylabel=ylabel, title='%s, R=%d' % (title, R), interpolation_numpoints=200, interpolation_method=interpolation_method, log_scale=False)

                if savefigs:
                    dataio.save_current_figure('pcolortrueM%s_%s_R%d_log_%s_{label}_{unique_id}.pdf' % (points_label, title, R, interpolation_method))

            all_datas = [dict(name='precision', data=result_all_precisions_mean_flat), dict(name='kappa', data=result_em_fits_kappa_flat), dict(name='kappavalid', data=result_em_fits_kappa_valid_flat), dict(name='target', data=result_em_fits_target_flat), dict(name='fisherinfo', data=result_fisherinfor_mean_flat)]
            all_points = []
            if plots_pcolor_realsizes_Msubs:
                all_points.append(dict(name='sub', data=result_parameters_flat_subM_converted, xlabel='M_conj', ylabel='M_feat'))
            if plots_pcolor_realsizes_Mtot:
                all_points.append(dict(name='tot', data=result_parameters_flat_Mtot_converted, xlabel='Mtot', ylabel='ratio_conj'))

            for curr_points in all_points:
                for curr_data in all_datas:
                    for R_i, R in enumerate(R_space):
                        currR_indices = curr_points['data'][:, 2] == R

                        plot_interp(curr_points['data'], curr_data['data'], currR_indices, title=curr_data['name'], points_label=curr_points['name'], xlabel=curr_points['xlabel'], ylabel=curr_points['ylabel'])


        else:
            # Do one pcolor for M and ratio per R
            for R_i, R in enumerate(R_space):
                # Check evolution of precision given M and ratio
                utils.pcolor_2d_data(result_all_precisions_mean[..., R_i], log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='precision, R=%d' % R)
                if savefigs:
                    dataio.save_current_figure('pcolor_precision_R%d_log_{label}_{unique_id}.pdf' % R)

                # Show kappa
                try:
                    utils.pcolor_2d_data(result_em_fits_kappa_valid[..., R_i], log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='kappa, R=%d' % R)
                    if savefigs:
                        dataio.save_current_figure('pcolor_kappa_R%d_log_{label}_{unique_id}.pdf' % R)
                except ValueError:
                    pass

                # Show probability on target
                utils.pcolor_2d_data(result_em_fits_target[..., R_i], log_scale=False, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='target, R=%d' % R)
                if savefigs:
                    dataio.save_current_figure('pcolor_target_R%d_{label}_{unique_id}.pdf' % R)

                # # Show Fisher info
                utils.pcolor_2d_data(result_fisherinfo_mean[..., R_i], log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='fisher info, R=%d' % R)
                if savefigs:
                    dataio.save_current_figure('pcolor_fisherinfo_R%d_log_{label}_{unique_id}.pdf' % R)

                plt.close('all')

    if plots_effect_M_target_kappa:
        def plot_ratio_target_kappa(ratio_target_kappa_given_M, target_kappa, R):
            f, ax = plt.subplots()
            ax.plot(M_space, ratio_target_kappa_given_M)
            ax.set_xlabel('M')
            ax.set_ylabel('Optimal ratio')
            ax.set_title('Optimal Ratio for kappa %d, R=%d' % (target_kappa, R))

            if savefigs:
                dataio.save_current_figure('optratio_M_targetkappa%d_R%d_{label}_{unique_id}.pdf' % (target_kappa, R))

        target_kappas = np.array([100, 200, 300, 500, 1000, 3000])
        for R_i, R in enumerate(R_space):
            for target_kappa in target_kappas:
                dist_to_target_kappa = (result_em_fits_kappa[..., R_i] - target_kappa)**2.
                best_dist_to_target_kappa = np.argmin(dist_to_target_kappa, axis=1)
                ratio_target_kappa_given_M = np.ma.masked_where(dist_to_target_kappa[np.arange(dist_to_target_kappa.shape[0]), best_dist_to_target_kappa] > MAX_DISTANCE, ratio_space[best_dist_to_target_kappa])

                # replot
                plot_ratio_target_kappa(ratio_target_kappa_given_M, target_kappa, R)

            plt.close('all')


    if plots_kappa_fi_comparison:

        # result_em_fits_kappa and fisher info
        if True:
            for R_i, R in enumerate(R_space):
                for M_tot_selected_i, M_tot_selected in enumerate(M_space[::2]):

                    # M_conj_space = ((1.-ratio_space)*M_tot_selected).astype(int)
                    # M_feat_space = M_tot_selected - M_conj_space

                    f, axes = plt.subplots(2, 1)
                    axes[0].plot(ratio_space, result_em_fits_kappa[2*M_tot_selected_i, ..., R_i])
                    axes[0].set_xlabel('ratio')
                    axes[0].set_title('Fitted kappa')

                    axes[1].plot(ratio_space, utils.stddev_to_kappa(1./result_fisherinfo_mean[2*M_tot_selected_i, ..., R_i]**0.5))
                    axes[1].set_xlabel('ratio')
                    axes[1].set_title('kappa_FI')

                    f.suptitle('M_tot %d' % M_tot_selected, fontsize=15)
                    f.set_tight_layout(True)

                    if savefigs:
                        dataio.save_current_figure('comparison_kappa_fisher_R%d_M%d_{label}_{unique_id}.pdf' % (R, M_tot_selected))

                    plt.close(f)

        if plots_multiple_fisherinfo:
            target_fisherinfos = np.array([100, 200, 300, 500, 1000])
            for R_i, R in enumerate(R_space):
                for target_fisherinfo in target_fisherinfos:
                    dist_to_target_fisherinfo = (result_fisherinfo_mean[..., R_i] - target_fisherinfo)**2.

                    utils.pcolor_2d_data(dist_to_target_fisherinfo, log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='Fisher info, R=%d' % R)
                    if savefigs:
                        dataio.save_current_figure('pcolor_distfi%d_R%d_log_{label}_{unique_id}.pdf' % (target_fisherinfo, R))

                plt.close('all')

    if specific_plot_effect_R:
        M_target = 356
        if convert_M_realsizes:
            M_tot_target = M_target
            delta_around_target = 30

            filter_points_totM_indices = np.abs(result_parameters_flat_Mtot_converted[:, 0] - M_tot_target) < delta_around_target

            # first check landscape
            utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat_Mtot_converted[filter_points_totM_indices][..., 1:], result_em_fits_kappa_flat[filter_points_totM_indices], xlabel='ratio_conj', ylabel='R', title='kappa, M_target=%d +- %d' % (M_tot_target, delta_around_target), interpolation_numpoints=200, interpolation_method=interpolation_method, log_scale=False, show_slider=False)

            if savefigs:
                dataio.save_current_figure('specific_pcolortrueMtot_kappa_M%d_log_%s_{label}_{unique_id}.pdf' % (M_tot_target, interpolation_method))

            # Then plot distance to specific kappa
            target_kappa = 1.2e3
            dist_target_kappa_flat = np.abs(result_em_fits_kappa_flat - target_kappa)
            mask_greater_than = 5e3

            utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat_Mtot_converted[filter_points_totM_indices][..., 1:], dist_target_kappa_flat[filter_points_totM_indices], xlabel='ratio_conj', ylabel='R', title='dist kappa %d, M_target=%d +- %d' % (target_kappa, M_tot_target, delta_around_target), interpolation_numpoints=200, interpolation_method=interpolation_method, log_scale=False, mask_greater_than=mask_greater_than, mask_smaller_than=0)

            if savefigs:
                dataio.save_current_figure('specific_pcolortrueMtot_distkappa%d_M%d_log_%s_{label}_{unique_id}.pdf' % (target_kappa, M_tot_target, interpolation_method))

        else:
            # Choose a M, find which ratio gives best fit to a given kappa
            M_target_i = np.argmin(np.abs(M_space - M_target))

            utils.pcolor_2d_data(result_em_fits_kappa[M_target_i], log_scale=True, x=ratio_space, y=R_space, xlabel='ratio', ylabel='R', ylabel_format="%d", title='Kappa, M %d' % (M_target))
            plt.gcf().set_tight_layout(True)
            plt.gcf().canvas.draw()
            if savefigs:
                dataio.save_current_figure('specific_Reffect_pcolor_kappa_M%dT%d_log_{label}_{unique_id}.pdf' % (M_target, T))
            # target_kappa = np.ma.mean(result_em_fits_kappa_valid[M_target_i])
            # target_kappa = 5*1e3
            target_kappa = 1.2e3

            # dist_target_kappa = np.abs(result_em_fits_kappa_valid[M_target_i] - target_kappa)
            dist_target_kappa = result_em_fits_kappa[M_target_i]/target_kappa
            dist_target_kappa[dist_target_kappa > 2.0] = 2.0
            # dist_target_kappa[dist_target_kappa < 0.5] = 0.5

            utils.pcolor_2d_data(dist_target_kappa, log_scale=False, x=ratio_space, y=R_space, xlabel='ratio', ylabel='R', ylabel_format="%d", title='Kappa dist %.2f, M %d' % (target_kappa, M_target), cmap='RdBu_r')
            plt.gcf().set_tight_layout(True)
            plt.gcf().canvas.draw()
            if savefigs:
                dataio.save_current_figure('specific_Reffect_pcolor_distkappa%d_M%dT%d_log_{label}_{unique_id}.pdf' % (target_kappa, M_target, T))

            # Plot the probability of being on-target
            utils.pcolor_2d_data(result_em_fits_target[M_target_i], log_scale=False, x=ratio_space, y=R_space, xlabel='ratio', ylabel='R', ylabel_format="%d", title='target mixture proportion, M %d' % (M_target), vmin=0.0, vmax=1.0, cmap='Greys')
            plt.gcf().set_tight_layout(True)
            plt.gcf().canvas.draw()
            if savefigs:
                dataio.save_current_figure('specific_Reffect_pcolor_target_M%dT%d_{label}_{unique_id}.pdf' % (M_target, T))



    if specific_plot_effect_ratio_M:
        # try to do the same plots as in ratio_scaling_M but with the current data
        R_target = 2
        interpolation_method = 'cubic'
        mask_greater_than = 1e3

        if convert_M_realsizes:
            # filter_points_targetR_indices = np.abs(result_parameters_flat_Mtot_converted[:, -1] - R_target) == 0
            filter_points_targetR_indices = np.abs(result_parameters_flat_subM_converted[:, -1] - R_target) == 0

            # utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., :-1], result_em_fits_kappa_flat[filter_points_targetR_indices], xlabel='M', ylabel='ratio_conj', title='kappa, R=%d' % (R_target), interpolation_numpoints=200, interpolation_method=interpolation_method, log_scale=False)
            utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat_subM_converted[filter_points_targetR_indices][..., :-1], result_em_fits_kappa_flat[filter_points_targetR_indices], xlabel='M_conj', ylabel='M_feat', title='kappa, R=%d' % (R_target), interpolation_numpoints=200, interpolation_method=interpolation_method, log_scale=False, show_slider=False)

            if savefigs:
                dataio.save_current_figure('specific_ratioM_pcolorsubM_kappa_R%d_log_%s_{label}_{unique_id}.pdf' % (R_target, interpolation_method))

            # Then plot distance to specific kappa
            target_kappa = 580
            # target_kappa = 2000
            dist_target_kappa_flat = np.abs(result_em_fits_kappa_flat - target_kappa)
            dist_target_kappa_flat = result_em_fits_kappa_flat/target_kappa
            dist_target_kappa_flat[dist_target_kappa_flat > 1.45] = 1.45
            dist_target_kappa_flat[dist_target_kappa_flat < 0.5] = 0.5

            # utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., :-1], dist_target_kappa_flat[filter_points_targetR_indices], xlabel='M', ylabel='ratio_conj', title='kappa, R=%d' % (R_target), interpolation_numpoints=200, interpolation_method=interpolation_method, log_scale=False, mask_smaller_than=0, show_slider=False, mask_greater_than =mask_greater_than)

            # utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat[filter_points_targetR_indices][..., :-1], dist_target_kappa_flat[filter_points_targetR_indices], xlabel='M', ylabel='ratio_conj', title='kappa, R=%d' % (R_target), interpolation_numpoints=200, interpolation_method=interpolation_method, log_scale=False, mask_smaller_than=0, show_slider=False, mask_greater_than =mask_greater_than)

            utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat_subM_converted[filter_points_targetR_indices][..., :-1], dist_target_kappa_flat[filter_points_targetR_indices], xlabel='M_conj', ylabel='M_feat', title='Kappa dist %.2f, R=%d' % (target_kappa, R_target), interpolation_numpoints=200, interpolation_method=interpolation_method, log_scale=False, mask_greater_than=mask_greater_than, mask_smaller_than=0, show_slider=False)

            if savefigs:
                dataio.save_current_figure('specific_ratioM_pcolorsubM_distkappa%d_R%d_log_%s_{label}_{unique_id}.pdf' % (target_kappa, R_target, interpolation_method))


            ## Scatter plot, works better...

            f, ax = plt.subplots()
            ss = ax.scatter(result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 0], result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 1], 500, c=(dist_target_kappa_flat[filter_points_targetR_indices]),
                norm=matplotlib.colors.LogNorm())
            plt.colorbar(ss)
            ax.set_xlabel('M')
            ax.set_ylabel('ratio')
            ax.set_xlim((result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 0].min()*0.8, result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 0].max()*1.03))
            ax.set_ylim((-0.05, result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 1].max()*1.05))
            ax.set_title('Kappa dist %.2f, R=%d' % (target_kappa, R_target))

            if savefigs:
                dataio.save_current_figure('specific_ratioM_scattertotM_distkappa%d_R%d_log_%s_{label}_{unique_id}.pdf' % (target_kappa, R_target, interpolation_method))


            ## Spline interpolation
            distkappa_spline_int_params = spint.bisplrep(result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 0], result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 1], np.log(dist_target_kappa_flat[filter_points_targetR_indices]), kx=3, ky=3, s=1)
            if True:
                M_interp_space = np.linspace(100, 740, 100)
                ratio_interp_space = np.linspace(0.0, 1.0, 100)
                # utils.pcolor_2d_data(spint.bisplev(M_interp_space, ratio_interp_space, distkappa_spline_int_params), y=ratio_interp_space, x=M_interp_space, ylabel='ratio', xlabel='M', xlabel_format="%d", title='Kappa dist %.2f, R %d' % (target_kappa, R_target), ticks_interpolate=11)
                utils.pcolor_2d_data(np.exp(spint.bisplev(M_interp_space, ratio_interp_space, distkappa_spline_int_params)), y=ratio_interp_space, x=M_interp_space, ylabel='ratio conjunctivity', xlabel='M', xlabel_format="%d", title='Ratio kappa/%.2f, R %d' % (target_kappa, R_target), ticks_interpolate=11, log_scale=False, cmap='RdBu_r')
                # plt.scatter(result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 0], result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 1], marker='o', c='b', s=5)
                # plt.scatter(np.argmin(np.abs(M_interp_space[:, np.newaxis] - result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 0]), axis=0), np.argmin(np.abs(ratio_interp_space[:, np.newaxis] - result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 1]), axis=0), marker='o', c='b', s=5)
            else:
                M_interp_space = M_space
                ratio_interp_space = ratio_space
                utils.pcolor_2d_data(spint.bisplev(M_interp_space, ratio_interp_space, distkappa_spline_int_params), y=ratio_interp_space, x=M_interp_space, ylabel='ratio', xlabel='M', xlabel_format="%d", title='Kappa dist %.2f, R %d' % (target_kappa, R_target))


            plt.gcf().set_tight_layout(True)
            plt.gcf().canvas.draw()
            if savefigs:
                dataio.save_current_figure('specific_ratioM_pcolorsplinetotM_distkappa%d_R%d_log_%s_{label}_{unique_id}.pdf' % (target_kappa, R_target, interpolation_method))

            ### Distance to Fisher Info

            target_fi = 2*target_kappa
            dist_target_fi_flat = np.abs(result_fisherinfor_mean_flat - target_fi)

            dist_target_fi_flat = result_fisherinfor_mean_flat/target_fi
            dist_target_fi_flat[dist_target_fi_flat > 1.45] = 1.45
            dist_target_fi_flat[dist_target_fi_flat < 0.5] = 0.5

            # utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., :-1], dist_target_kappa_flat[filter_points_targetR_indices], xlabel='M', ylabel='ratio_conj', title='kappa, R=%d' % (R_target), interpolation_numpoints=200, interpolation_method=interpolation_method, log_scale=False)
            utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat_subM_converted[filter_points_targetR_indices][..., :-1], dist_target_fi_flat[filter_points_targetR_indices], xlabel='M_conj', ylabel='M_feat', title='FI dist %.2f, R=%d' % (target_fi, R_target), interpolation_numpoints=200, interpolation_method=interpolation_method, log_scale=False, mask_greater_than=mask_greater_than, show_slider=False)

            if savefigs:
                dataio.save_current_figure('specific_ratioM_pcolorsubM_distfi%d_R%d_log_%s_{label}_{unique_id}.pdf' % (target_fi, R_target, interpolation_method))


            distFI_spline_int_params = spint.bisplrep(result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 0], result_parameters_flat_Mtot_converted[filter_points_targetR_indices][..., 1], np.log(dist_target_fi_flat[filter_points_targetR_indices]), kx=3, ky=3, s=1)

            M_interp_space = np.linspace(100, 740, 100)
            ratio_interp_space = np.linspace(0.0, 1.0, 100)
            # utils.pcolor_2d_data(spint.bisplev(M_interp_space, ratio_interp_space, spline_int_params), y=ratio_interp_space, x=M_interp_space, ylabel='ratio', xlabel='M', xlabel_format="%d", title='Kappa dist %.2f, R %d' % (target_kappa, R_target), ticks_interpolate=11)
            utils.pcolor_2d_data(np.exp(spint.bisplev(M_interp_space, ratio_interp_space, distFI_spline_int_params)), y=ratio_interp_space, x=M_interp_space, ylabel='ratio conjunctivity', xlabel='M', xlabel_format="%d", title='Ratio FI/%.2f, R %d' % (target_fi, R_target), ticks_interpolate=11, log_scale=False, cmap='RdBu_r')
            plt.gcf().set_tight_layout(True)
            plt.gcf().canvas.draw()
            if savefigs:
                dataio.save_current_figure('specific_ratioM_pcolorsplinetotM_distfi%d_R%d_log_%s_{label}_{unique_id}.pdf' % (target_fi, R_target, interpolation_method))



        else:
            R_target_i = np.argmin(np.abs(R_space - R_target))

            utils.pcolor_2d_data(result_em_fits_kappa_valid[..., R_target_i], log_scale=True, y=ratio_space, x=M_space, ylabel='ratio', xlabel='M', xlabel_format="%d", title='Kappa, R %d' % (R_target))
            if savefigs:
                dataio.save_current_figure('specific_ratioM_pcolor_kappa_R%d_log_{label}_{unique_id}.pdf' % (R_target))
            # target_kappa = np.ma.mean(result_em_fits_kappa_valid[R_target_i])
            # target_kappa = 5*1e3
            target_kappa = 580

            dist_target_kappa = np.ma.masked_greater(np.abs(result_em_fits_kappa_valid[..., R_target_i] - target_kappa), mask_greater_than*5)

            utils.pcolor_2d_data(dist_target_kappa, log_scale=True, y=ratio_space, x=M_space, ylabel='ratio', xlabel='M', xlabel_format="%d", title='Kappa dist %.2f, R %d' % (target_kappa, R_target))
            if savefigs:
                dataio.save_current_figure('specific_ratioM_pcolor_distkappa%d_R%d_log_{label}_{unique_id}.pdf' % (target_kappa, R_target))






    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = []

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='higher_dimensions_R')

    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Runs and collect precision and mixture model fits for varying M, ratio_conj and R. Should check the effect of R',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['M', 'ratio_conj', 'R'],
                     variables_to_load=['result_all_precisions', 'result_em_fits', 'result_fisher_info'],
                     variables_description=['Precision of recall', 'Fits mixture model', 'Fisher infor'],
                     post_processing=plots_ratioMscaling,
                     save_output_filename='plots_ratioMscaling',
                     concatenate_multiple_datasets=True
                     )




if __name__ == '__main__':

    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio', 'post_processing_outputs', 'fit_exp']

    if 'variables_to_save' in experiment_launcher.all_vars:
        # Also reinstantiate the variables we saved
        variables_to_reinstantiate.extend(experiment_launcher.all_vars['variables_to_save'])

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]

    for var_reinst in post_processing_outputs:
        vars()[var_reinst] = post_processing_outputs[var_reinst]

