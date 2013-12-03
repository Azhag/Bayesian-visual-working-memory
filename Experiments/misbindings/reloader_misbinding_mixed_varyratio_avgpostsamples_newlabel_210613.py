"""
    ExperimentDescriptor for Misbinding effect for Mixed population code.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


import pypr.clustering.gmm as pygmm

from experimentlauncher import *
from dataio import *
# from smooth import *
import inspect
import em_circularmixture
import em_circularmixture_allitems_uniquekappa

import utils

# # Commit @2042319 +

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')


def plots_misbinding_logposterior(data_pbs, generator_module=None):
    '''
        Reload 3D volume runs from PBS and plot them

    '''


    #### SETUP
    #
    savefigs = True

    plot_logpost = False
    plot_error = False
    plot_mixtmodel = True

    use_allitems_mixturesmodel = True

    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_all_log_posterior = np.squeeze(data_pbs.dict_arrays['result_all_log_posterior']['results'])
    result_all_thetas = np.squeeze(data_pbs.dict_arrays['result_all_thetas']['results'])

    ratio_space = data_pbs.loaded_data['parameters_uniques']['ratio_conj']

    print ratio_space
    print result_all_log_posterior.shape

    N = result_all_thetas.shape[-1]

    result_prob_wrong = np.zeros((ratio_space.size, N))
    result_em_fits = np.empty((ratio_space.size, 6))*np.nan


    fixed_means = [-np.pi*0.6, np.pi*0.6]
    all_angles = np.linspace(-np.pi, np.pi, result_all_log_posterior.shape[-1])

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])


    plt.rcParams['font.size'] = 18



    if plot_logpost:
        for ratio_conj_i, ratio_conj in enumerate(ratio_space):
            # ax = utils.plot_mean_std_area(all_angles, nanmean(result_all_log_posterior[ratio_conj_i], axis=0), nanstd(result_all_log_posterior[ratio_conj_i], axis=0))

            # ax.set_xlim((-np.pi, np.pi))
            # ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
            # ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
            # ax.set_yticks(())

            # ax.get_figure().canvas.draw()

            # if savefigs:
            #     dataio.save_current_figure('results_misbinding_logpost_ratioconj%.2f_{label}_global_{unique_id}.pdf' % ratio_conj)


            # Compute the probability of answering wrongly (from fitting mixture distrib onto posterior)
            for n in xrange(result_all_log_posterior.shape[1]):
                result_prob_wrong[ratio_conj_i, n], _, _ = utils.fit_gaussian_mixture_fixedmeans(all_angles, np.exp(result_all_log_posterior[ratio_conj_i, n]), fixed_means=fixed_means, normalise=True, return_fitted_data=False, should_plot=False)

        # ax = utils.plot_mean_std_area(ratio_space, nanmean(result_prob_wrong, axis=-1), nanstd(result_prob_wrong, axis=-1))
        plt.figure()
        plt.plot(ratio_space, utils.nanmean(result_prob_wrong, axis=-1))

        # ax.get_figure().canvas.draw()
        if savefigs:
            dataio.save_current_figure('results_misbinding_probwrongpost_allratioconj_{label}_global_{unique_id}.pdf')

    if plot_error:

        ## Compute Standard deviation/precision from samples and plot it as a function of ratio_conj
        stats = utils.compute_mean_std_circular_data(utils.wrap_angles(result_all_thetas - fixed_means[1]).T)

        f = plt.figure()
        plt.plot(ratio_space, stats['std'])
        plt.ylabel('Standard deviation [rad]')

        if savefigs:
            dataio.save_current_figure('results_misbinding_stddev_allratioconj_{label}_global_{unique_id}.pdf')

        f = plt.figure()
        plt.plot(ratio_space, utils.compute_angle_precision_from_std(stats['std'], square_precision=False), linewidth=2)
        plt.ylabel('Precision [$1/rad$]')
        plt.xlabel('Proportion of conjunctive units')
        plt.grid()

        if savefigs:
            dataio.save_current_figure('results_misbinding_precision_allratioconj_{label}_global_{unique_id}.pdf')

        ## Compute the probability of misbinding
        # 1) Just count samples < 0 / samples tot
        # 2) Fit a mixture model, average over mixture probabilities
        prob_smaller0 = np.sum(result_all_thetas <= 1, axis=1)/float(result_all_thetas.shape[1])

        em_centers = np.zeros((ratio_space.size, 2))
        em_covs = np.zeros((ratio_space.size, 2))
        em_pk = np.zeros((ratio_space.size, 2))
        em_ll = np.zeros(ratio_space.size)
        for ratio_conj_i, ratio_conj in enumerate(ratio_space):
            cen_lst, cov_lst, em_pk[ratio_conj_i], em_ll[ratio_conj_i] = pygmm.em(result_all_thetas[ratio_conj_i, np.newaxis].T, K = 2, max_iter = 400, init_kw={'cluster_init':'fixed', 'fixed_means': fixed_means})

            em_centers[ratio_conj_i] = np.array(cen_lst).flatten()
            em_covs[ratio_conj_i] = np.array(cov_lst).flatten()

        # print em_centers
        # print em_covs
        # print em_pk

        f = plt.figure()
        plt.plot(ratio_space, prob_smaller0)
        plt.ylabel('Misbound proportion')
        if savefigs:
            dataio.save_current_figure('results_misbinding_countsmaller0_allratioconj_{label}_global_{unique_id}.pdf')

        f = plt.figure()
        plt.plot(ratio_space, np.max(em_pk, axis=-1), 'g', linewidth=2)
        plt.ylabel('Mixture proportion, correct')
        plt.xlabel('Proportion of conjunctive units')
        plt.grid()
        if savefigs:
            dataio.save_current_figure('results_misbinding_emmixture_allratioconj_{label}_global_{unique_id}.pdf')


        # Put everything on one figure
        f = plt.figure(figsize=(10, 6))
        norm_for_plot = lambda x: (x - np.min(x))/np.max((x - np.min(x)))
        plt.plot(ratio_space, norm_for_plot(stats['std']), ratio_space, norm_for_plot(utils.compute_angle_precision_from_std(stats['std'], square_precision=False)), ratio_space, norm_for_plot(prob_smaller0), ratio_space, norm_for_plot(em_pk[:, 1]), ratio_space, norm_for_plot(em_pk[:, 0]))
        plt.legend(('Std dev', 'Precision', 'Prob smaller 1', 'Mixture proportion correct', 'Mixture proportion misbinding'))
        # plt.plot(ratio_space, norm_for_plot(compute_angle_precision_from_std(stats['std'], square_precision=False)), ratio_space, norm_for_plot(em_pk[:, 1]), linewidth=2)
        # plt.legend(('Precision', 'Mixture proportion correct'), loc='best')
        plt.grid()
        if savefigs:
            dataio.save_current_figure('results_misbinding_allmetrics_allratioconj_{label}_global_{unique_id}.pdf')


    if plot_mixtmodel:
        # Fit Paul's model
        target_angle = np.ones(N)*fixed_means[1]
        nontarget_angles = np.ones((N, 1))*fixed_means[0]

        for ratio_conj_i, ratio_conj in enumerate(ratio_space):
            print "Ratio: ", ratio_conj

            responses = result_all_thetas[ratio_conj_i]

            if not use_allitems_mixturesmodel:
                curr_params_fit = em_circularmixture.fit(responses, target_angle, nontarget_angles)
                result_em_fits[ratio_conj_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random', 'train_LL')]
            else:
                curr_params_fit = em_circularmixture_allitems_uniquekappa.fit(responses, target_angle, nontarget_angles)
                print curr_params_fit
                result_em_fits[ratio_conj_i] = [curr_params_fit['kappa'], curr_params_fit['mixt_target']] + curr_params_fit['mixt_nontargets'].tolist() + [curr_params_fit[key] for key in ('mixt_random', 'train_LL', 'bic')]

            print curr_params_fit

        if False:
            f, ax = plt.subplots()
            ax2 = ax.twinx()

            # left axis, kappa
            ax = utils.plot_mean_std_area(ratio_space, result_em_fits[:, 0], 0*result_em_fits[:, 0], xlabel='Proportion of conjunctive units', ylabel="Inverse variance $[rad^{-2}]$", ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Fitted kappa', color='k')

            # Right axis, mixture probabilities
            utils.plot_mean_std_area(ratio_space, result_em_fits[:, 1], 0*result_em_fits[:, 1], xlabel='Proportion of conjunctive units', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Target')
            utils.plot_mean_std_area(ratio_space, result_em_fits[:, 2], 0*result_em_fits[:, 2], xlabel='Proportion of conjunctive units', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Nontarget')
            utils.plot_mean_std_area(ratio_space, result_em_fits[:, 3], 0*result_em_fits[:, 3], xlabel='Proportion of conjunctive units', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Random')

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, fontsize=12, loc='right')

            # ax.set_xlim([0.9, 5.1])
            # ax.set_xticks(range(1, 6))
            # ax.set_xticklabels(range(1, 6))
            plt.grid()

            f.canvas.draw()

        if True:
            # Mixture probabilities
            ax = utils.plot_mean_std_area(ratio_space, result_em_fits[:, 1], 0*result_em_fits[:, 1], xlabel='Proportion of conjunctive units', ylabel="Mixture probabilities", linewidth=3, fmt='-', markersize=8, label='Target')
            utils.plot_mean_std_area(ratio_space, result_em_fits[:, 2], 0*result_em_fits[:, 2], xlabel='Proportion of conjunctive units', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='-', markersize=8, label='Nontarget')
            utils.plot_mean_std_area(ratio_space, result_em_fits[:, 3], 0*result_em_fits[:, 3], xlabel='Proportion of conjunctive units', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='-', markersize=8, label='Random')

            ax.legend(loc='right')

            # ax.set_xlim([0.9, 5.1])
            # ax.set_xticks(range(1, 6))
            # ax.set_xticklabels(range(1, 6))
            plt.grid()

        if savefigs:
            dataio.save_current_figure('results_misbinding_emmixture_allratioconj_{label}_global_{unique_id}.pdf')


    # plt.figure()
    # plt.plot(ratio_MMlower, results_filtered_smoothed/np.max(results_filtered_smoothed, axis=0), linewidth=2)
    # plt.plot(ratio_MMlower[np.argmax(results_filtered_smoothed, axis=0)], np.ones(results_filtered_smoothed.shape[-1]), 'ro', markersize=10)
    # plt.grid()
    # plt.ylim((0., 1.1))
    # plt.subplots_adjust(right=0.8)
    # plt.legend(['%d item' % i + 's'*(i>1) for i in xrange(1, T+1)], loc='center right', bbox_to_anchor=(1.3, 0.5))
    # plt.xticks(np.linspace(0, 1.0, 5))

    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['result_all_log_posterior', 'ratio_space', 'all_args']

    if savefigs:
        dataio.save_variables(variables_to_save, locals())


    plt.show()

    return locals()




generator_script='generator_misbinding_mixed_varyratio_avgpostsamples_newlabel_210613.py'

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Study misbindings, by computing an average posterior for fixed stimuli. Check the distribution of errors as well. Uses a Mixed population, vary ratio_conj to see what happens. Limit to squared ratio_subpop_nb only. Plots already done before, but lets try to redo them',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['ratio_conj'],
                     variables_to_load=['result_all_log_posterior', 'result_all_thetas'],
                     variables_description=['Log posterior for multiple ratio_conj'],
                     post_processing=plots_misbinding_logposterior,
                     save_output_filename='plots_misbinding_logposterior'
                     )




if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio', 'post_processing_outputs']

    if 'variables_to_save' in experiment_launcher.all_vars:
        # Also reinstantiate the variables we saved
        variables_to_reinstantiate.extend(experiment_launcher.all_vars['variables_to_save'])

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]

    for var_reinst in post_processing_outputs:
        vars()[var_reinst] = post_processing_outputs[var_reinst]

