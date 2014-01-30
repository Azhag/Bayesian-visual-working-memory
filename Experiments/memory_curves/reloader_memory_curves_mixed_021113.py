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

import inspect

import load_experimental_data

# Commit @8c49507 +


def plots_memory_curves(data_pbs, generator_module=None):
    '''
        Reload and plot memory curve of a Mixed code.
        Can use Marginal Fisher Information and fitted Mixture Model as well
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plot_pcolor_fit_precision_to_fisherinfo = False
    plot_selected_memory_curves = False
    plot_best_memory_curves = False
    plot_subplots_persigmax = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_all_precisions_mean = utils.nanmean(np.squeeze(data_pbs.dict_arrays['result_all_precisions']['results']), axis=-1)
    result_all_precisions_std = utils.nanstd(np.squeeze(data_pbs.dict_arrays['result_all_precisions']['results']), axis=-1)
    # ratio_space, sigmax_space, T, 5
    result_em_fits_mean = utils.nanmean(np.squeeze(data_pbs.dict_arrays['result_em_fits']['results']), axis=-1)
    result_em_fits_std = utils.nanstd(np.squeeze(data_pbs.dict_arrays['result_em_fits']['results']), axis=-1)
    result_marginal_inv_fi_mean = utils.nanmean(np.squeeze(data_pbs.dict_arrays['result_marginal_inv_fi']['results']), axis=-1)
    result_marginal_inv_fi_std = utils.nanstd(np.squeeze(data_pbs.dict_arrays['result_marginal_inv_fi']['results']), axis=-1)
    result_marginal_fi_mean = utils.nanmean(np.squeeze(1./data_pbs.dict_arrays['result_marginal_inv_fi']['results']), axis=-1)
    result_marginal_fi_std = utils.nanstd(np.squeeze(1./data_pbs.dict_arrays['result_marginal_inv_fi']['results']), axis=-1)

    ratioconj_space = data_pbs.loaded_data['parameters_uniques']['ratio_conj']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    T_space = data_pbs.loaded_data['datasets_list'][0]['T_space']

    print ratioconj_space
    print sigmax_space
    print T_space
    print result_all_precisions_mean.shape, result_em_fits_mean.shape, result_marginal_inv_fi_mean.shape

    dataio = DataIO.DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    ## Load Experimental data
    data_simult = load_experimental_data.load_data_simult(data_dir=os.path.normpath(os.path.join(os.path.split(load_experimental_data.__file__)[0], '../../experimental_data/')), fit_mixture_model=True)
    memory_experimental_precision = data_simult['precision_nitems_theo']
    memory_experimental_kappa = np.array([data['kappa'] for _, data in data_simult['em_fits_nitems']['mean'].items()])
    memory_experimental_kappa_std = np.array([data['kappa'] for _, data in data_simult['em_fits_nitems']['std'].items()])

    experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(load_experimental_data.__file__)[0])
    data_bays2009 = load_experimental_data.load_data_bays2009(data_dir=os.path.normpath(os.path.join(experim_datadir, '../../experimental_data/')), fit_mixture_model=True)
    bays09_experimental_mixtures_mean = data_bays2009['em_fits_nitems_arrays']['mean']
    # add interpolated points for 3 and 5 items
    bays3 = (bays09_experimental_mixtures_mean[:, 2] + bays09_experimental_mixtures_mean[:, 1])/2.
    bays5 = (bays09_experimental_mixtures_mean[:, -1] + bays09_experimental_mixtures_mean[:, -2])/2.
    bays09_experimental_mixtures_mean_compatible = np.c_[bays09_experimental_mixtures_mean[:,:2], bays3, bays09_experimental_mixtures_mean[:, 2], bays5]

    # Boost non-targets
    # bays09_experimental_mixtures_mean_compatible[1] *= 1.5
    # bays09_experimental_mixtures_mean_compatible[2] /= 1.5
    # bays09_experimental_mixtures_mean_compatible /= np.sum(bays09_experimental_mixtures_mean_compatible, axis=0)

    # Compute some landscapes of fit!
    dist_diff_precision_margfi = np.sum(np.abs(result_all_precisions_mean*2. - result_marginal_fi_mean[..., 0])**2., axis=-1)
    dist_ratio_precision_margfi = np.sum(np.abs((result_all_precisions_mean*2.)/result_marginal_fi_mean[..., 0] - 1.0)**2., axis=-1)
    dist_diff_emkappa_margfi = np.sum(np.abs(result_em_fits_mean[..., 0]*2. - result_marginal_fi_mean[..., 0])**2., axis=-1)
    dist_ratio_emkappa_margfi = np.sum(np.abs((result_em_fits_mean[..., 0]*2.)/result_marginal_fi_mean[..., 0] - 1.0)**2., axis=-1)

    dist_diff_precision_experim = np.sum(np.abs(result_all_precisions_mean - memory_experimental_precision)**2., axis=-1)
    dist_diff_emkappa_experim = np.sum(np.abs(result_em_fits_mean[..., 0] - memory_experimental_kappa)**2., axis=-1)

    dist_diff_precision_experim_1item = np.abs(result_all_precisions_mean[..., 0] - memory_experimental_precision[0])**2.
    dist_diff_precision_experim_2item = np.abs(result_all_precisions_mean[..., 1] - memory_experimental_precision[1])**2.
    dist_diff_precision_margfi_1item = np.abs(result_all_precisions_mean[..., 0]*2. - result_marginal_fi_mean[..., 0, 0])**2.
    dist_diff_emkappa_experim_1item = np.abs(result_em_fits_mean[..., 0, 0] - memory_experimental_kappa[0])**2.
    dist_diff_margfi_experim_1item = np.abs(result_marginal_fi_mean[..., 0, 0] - memory_experimental_precision[0])**2.

    dist_diff_emkappa_mixtures_bays09 = np.sum(np.sum((result_em_fits_mean[..., 1:4] - bays09_experimental_mixtures_mean_compatible[1:].T)**2., axis=-1), axis=-1)
    dist_diff_modelfits_experfits_bays09 = np.sum(np.sum((result_em_fits_mean[..., :4] - bays09_experimental_mixtures_mean_compatible.T)**2., axis=-1), axis=-1)


    if plot_pcolor_fit_precision_to_fisherinfo:
        # Check fit between precision and fisher info
        utils.pcolor_2d_data(dist_diff_precision_margfi, log_scale=True, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax')

        if savefigs:
            dataio.save_current_figure('match_precision_margfi_log_pcolor_{label}_{unique_id}.pdf')

        # utils.pcolor_2d_data(dist_diff_precision_margfi, x=ratioconj_space, y=sigmax_space[2:], xlabel='ratio conj', ylabel='sigmax')
        # if savefigs:
        #    dataio.save_current_figure('match_precision_margfi_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_ratio_precision_margfi, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_ratio_precision_margfi_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_emkappa_margfi, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_diff_emkappa_margfi_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_ratio_emkappa_margfi, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_ratio_emkappa_margfi_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_precision_experim, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_diff_precision_experim_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_emkappa_experim, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_diff_emkappa_experim_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_precision_margfi*dist_diff_emkappa_margfi*dist_diff_precision_experim*dist_diff_emkappa_experim, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_bigmultiplication_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_precision_margfi_1item, log_scale=True, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_precision_margfi_1item_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_precision_experim_1item, log_scale=True, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_precision_experim_1item_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_emkappa_experim_1item, log_scale=True, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_emkappa_experim_1item_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_margfi_experim_1item, log_scale=True, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_margfi_experim_1item_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_precision_experim_2item, log_scale=True, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_precision_experim_2item_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_emkappa_mixtures_bays09, log_scale=True, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_mixtures_experbays09_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_modelfits_experfits_bays09, log_scale=True, x=ratioconj_space, y=sigmax_space, xlabel='ratio conj', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_emfits_experbays09_pcolor_{label}_{unique_id}.pdf')

    # Macro plot
    def mem_plot_precision(sigmax_i, ratioconj_i):
        ax = utils.plot_mean_std_area(T_space, memory_experimental_precision, np.zeros(T_space.size), linewidth=3, fmt='o-', markersize=8, label='Experimental data')

        ax = utils.plot_mean_std_area(T_space, result_all_precisions_mean[ratioconj_i, sigmax_i], result_all_precisions_std[ratioconj_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Precision of samples')

        # ax = utils.plot_mean_std_area(T_space, 0.5*result_marginal_fi_mean[..., 0][ratioconj_i, sigmax_i], 0.5*result_marginal_fi_std[..., 0][ratioconj_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Marginal Fisher Information')

        # ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][ratioconj_i, sigmax_i], result_em_fits_std[..., 0][ratioconj_i, sigmax_i], ax_handle=ax, xlabel='Number of items', ylabel="Inverse variance $[rad^{-2}]$", linewidth=3, fmt='o-', markersize=8, label='Fitted kappa')

        ax.set_title('ratio_conj %.2f, sigmax %.2f' % (ratioconj_space[ratioconj_i], sigmax_space[sigmax_i]))
        ax.legend()
        ax.set_xlim([0.9, 5.1])
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(range(1, 6))

        if savefigs:
            dataio.save_current_figure('memorycurves_precision_ratioconj%.2fsigmax%.2f_{label}_{unique_id}.pdf' % (ratioconj_space[ratioconj_i], sigmax_space[sigmax_i]))

    def mem_plot_kappa(sigmax_i, ratioconj_i):
        ax = utils.plot_mean_std_area(T_space, memory_experimental_kappa, memory_experimental_kappa_std, linewidth=3, fmt='o-', markersize=8, label='Experimental data')

        ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][ratioconj_i, sigmax_i], result_em_fits_std[..., 0][ratioconj_i, sigmax_i], xlabel='Number of items', ylabel="Memory error $[rad^{-2}]$", ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Fitted kappa')

        # ax = utils.plot_mean_std_area(T_space, 0.5*result_marginal_fi_mean[..., 0][ratioconj_i, sigmax_i], 0.5*result_marginal_fi_std[..., 0][ratioconj_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Marginal Fisher Information')

        ax.set_title('ratio_conj %.2f, sigmax %.2f' % (ratioconj_space[ratioconj_i], sigmax_space[sigmax_i]))
        ax.legend()
        ax.set_xlim([0.9, 5.1])
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(range(1, 6))

        if savefigs:
            dataio.save_current_figure('memorycurves_kappa_ratioconj%.2fsigmax%.2f_{label}_{unique_id}.pdf' % (ratioconj_space[ratioconj_i], sigmax_space[sigmax_i]))

    def em_plot(sigmax_i, ratioconj_i):
        # TODO finish checking this up.
        f, ax = plt.subplots()
        ax2 = ax.twinx()

        # left axis, kappa
        ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][ratioconj_i, sigmax_i], result_em_fits_std[..., 0][ratioconj_i, sigmax_i], xlabel='Number of items', ylabel="Inverse variance $[rad^{-2}]$", ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Fitted kappa', color='k')

        # Right axis, mixture probabilities
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 1][ratioconj_i, sigmax_i], result_em_fits_std[..., 1][ratioconj_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Target')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 2][ratioconj_i, sigmax_i], result_em_fits_std[..., 2][ratioconj_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Nontarget')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 3][ratioconj_i, sigmax_i], result_em_fits_std[..., 3][ratioconj_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Random')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)

        ax.set_title('ratio_conj %.2f, sigmax %.2f' % (ratioconj_space[ratioconj_i], sigmax_space[sigmax_i]))
        ax.set_xlim([0.9, 5.1])
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(range(1, 6))

        f.canvas.draw()

        if savefigs:
            dataio.save_current_figure('memorycurves_emfits_ratioconj%.2fsigmax%.2f_{label}_{unique_id}.pdf' % (ratioconj_space[ratioconj_i], sigmax_space[sigmax_i]))

    def em_plot_paper(sigmax_i, ratioconj_i):
        f, ax = plt.subplots()

        # Right axis, mixture probabilities
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 1][ratioconj_i, sigmax_i], result_em_fits_std[..., 1][ratioconj_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Target')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 2][ratioconj_i, sigmax_i], result_em_fits_std[..., 2][ratioconj_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Nontarget')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 3][ratioconj_i, sigmax_i], result_em_fits_std[..., 3][ratioconj_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Random')

        ax.legend(prop={'size':15})

        ax.set_title('ratio_conj %.2f, sigmax %.2f' % (ratioconj_space[ratioconj_i], sigmax_space[sigmax_i]))
        ax.set_xlim([1.0, 5.0])
        ax.set_ylim([0.0, 1.1])
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(range(1, 6))

        f.canvas.draw()

        if savefigs:
            dataio.save_current_figure('memorycurves_emfits_paper_ratioconj%.2fsigmax%.2f_{label}_{unique_id}.pdf' % (ratioconj_space[ratioconj_i], sigmax_space[sigmax_i]))


    if plot_selected_memory_curves:
        selected_values = [[0.84, 0.23], [0.84, 0.19]]

        for current_values in selected_values:
            # Find the indices
            ratioconj_i     = np.argmin(np.abs(current_values[0] - ratioconj_space))
            sigmax_i        = np.argmin(np.abs(current_values[1] - sigmax_space))


            mem_plot_precision(sigmax_i, ratioconj_i)
            mem_plot_kappa(sigmax_i, ratioconj_i)

    if plot_best_memory_curves:
        # Best precision fit
        best_axis2_i_all = np.argmin(dist_diff_precision_experim, axis=1)

        for axis1_i, best_axis2_i in enumerate(best_axis2_i_all):
            mem_plot_precision(best_axis2_i, axis1_i)

        # Best kappa fit
        best_axis2_i_all = np.argmin(dist_diff_emkappa_experim, axis=1)

        for axis1_i, best_axis2_i in enumerate(best_axis2_i_all):
            mem_plot_kappa(best_axis2_i, axis1_i)
            em_plot(best_axis2_i, axis1_i)

        # Best em parameters fit to Bays09
        best_axis2_i_all = np.argmin(dist_diff_modelfits_experfits_bays09, axis=1)

        for axis1_i, best_axis2_i in enumerate(best_axis2_i_all):
            mem_plot_kappa(best_axis2_i, axis1_i)
            # em_plot(best_axis2_i, axis1_i)
            em_plot_paper(best_axis2_i, axis1_i)

    if plot_subplots_persigmax:
        # Do subplots with ratio_conj on x, one plot per T and different figures per sigmax. Looks a bit like reloader_hierarchical_constantMMlower_maxlik_allresponses_221213.py

        sigmax_selected_indices = np.array([np.argmin((sigmax_space - sigmax_select)**2.) for sigmax_select in [0.1, 0.2, 0.3, 0.4, 0.5]])

        for sigmax_select_i in sigmax_selected_indices:
            # two plots per sigmax.

            f, axes = plt.subplots(nrows=T_space[-1], ncols=1, sharex=True, figsize=(10, 12))
            # all_lines_bis = []

            for T_i, T in enumerate(T_space):
                # Plot EM mixtures
                utils.plot_mean_std_area(ratioconj_space, result_em_fits_mean[:, sigmax_select_i, T_i, 1], result_em_fits_std[:, sigmax_select_i, T_i, 1], ax_handle=axes[T_i], linewidth=3, fmt='o-', markersize=5)
                utils.plot_mean_std_area(ratioconj_space, result_em_fits_mean[:, sigmax_select_i, T_i, 2], result_em_fits_std[:, sigmax_select_i, T_i, 2], ax_handle=axes[T_i], linewidth=3, fmt='o-', markersize=5)
                utils.plot_mean_std_area(ratioconj_space, result_em_fits_mean[:, sigmax_select_i, T_i, 3], result_em_fits_std[:, sigmax_select_i, T_i, 3], ax_handle=axes[T_i], linewidth=3, fmt='o-', markersize=5)

                # ratio_MMlower_space, result_emfits_filtered[:, i, 1:4], 0*result_emfits_filtered[:, i, 1:4], ax_handle=axes[T_i], linewidth=2) #, color=all_lines[T_i].get_color())
                # curr_lines = axes[T_i].plot(ratio_MMlower_space, results_precision_filtered[:, i], linewidth=2, color=all_lines[T_i].get_color())
                axes[T_i].grid()
                axes[T_i].set_xticks(np.linspace(0., 1.0, 5))
                axes[T_i].set_xlim((0.0, 1.0))
                # axes[T_i].set_yticks([])
                # axes[T_i].set_ylim((np.min(result_emfits_filtered[:, i, 0]), result_emfits_filtered[max_ind, i, 0]*1.1))
                axes[T_i].set_ylim((0.0, 1.05))
                axes[T_i].locator_params(axis='y', tight=True, nbins=4)
                # all_lines_bis.extend(curr_lines)

            axes[0].set_title('Sigmax: %.3f' % sigmax_space[sigmax_select_i])
            axes[-1].set_xlabel('Proportion of conjunctive units')

            if savefigs:
                dataio.save_current_figure('results_subplots_emmixtures_sigmax%.2f_{label}_global_{unique_id}.pdf' % sigmax_space[sigmax_select_i])

            f, axes = plt.subplots(nrows=T_space[-1], ncols=1, sharex=True, figsize=(10, 12))

            # Kappa kappa
            for T_i, T in enumerate(T_space):

                # Plot kappa mixture
                utils.plot_mean_std_area(ratioconj_space, result_em_fits_mean[:, sigmax_select_i, T_i, 0], result_em_fits_std[:, sigmax_select_i, T_i, 0], ax_handle=axes[T_i], linewidth=3, fmt='o-', markersize=5)
                # utils.plot_mean_std_area(ratio_MMlower_space, result_emfits_filtered[:, i, 0], 0*result_emfits_filtered[:, i, 0], ax_handle=axes[T_i], linewidth=2) #, color=all_lines[T_i].get_color())
                # curr_lines = axes[T_i].plot(ratio_MMlower_space, results_precision_filtered[:, i], linewidth=2, color=all_lines[T_i].get_color())
                axes[T_i].grid()
                axes[T_i].set_xticks(np.linspace(0., 1.0, 5))
                axes[T_i].set_xlim((0.0, 1.0))
                # axes[T_i].set_yticks([])
                # axes[T_i].set_ylim((np.min(result_emfits_filtered[:, i, 0]), result_emfits_filtered[max_ind, i, 0]*1.1))
                # axes[T_i].set_ylim((0.0, 1.0))
                axes[T_i].locator_params(axis='y', tight=True, nbins=4)
                # all_lines_bis.extend(curr_lines)

            axes[0].set_title('Sigmax: %.3f' % sigmax_space[sigmax_select_i])
            axes[-1].set_xlabel('Proportion of conjunctive units')

            # f.subplots_adjust(right=0.75)
            # plt.figlegend(all_lines_bis, ['%d item' % i + 's'*(i>1) for i in xrange(1, T+1)], loc='right', bbox_to_anchor=(1.0, 0.5))

            if savefigs:
                dataio.save_current_figure('results_subplots_emkappa_sigmax%.2f_{label}_global_{unique_id}.pdf' % sigmax_space[sigmax_select_i])



    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['memory_experimental_precision', 'memory_experimental_kappa', 'bays09_experimental_mixtures_mean_compatible']

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='memory_curves')

    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Fit Memory curves using the new code (october 2013). Compute marginal inverse fisher information, which is slightly better at capturing items interactions effects. Also fit Mixture models directly. Uses a Mixed population now. Maybe not the best for FI fit, but we may remove them anyway.',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['ratio_conj', 'sigmax'],
                     variables_to_load=['result_all_precisions', 'result_em_fits', 'result_marginal_inv_fi'],
                     variables_description=['Precision of recall', 'Fits mixture model', 'Marginal Inverse Fisher Information'],
                     post_processing=plots_memory_curves,
                     save_output_filename='plots_memory_curves',
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

