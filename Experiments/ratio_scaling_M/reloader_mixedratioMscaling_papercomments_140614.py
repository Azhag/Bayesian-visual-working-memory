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

    plots_pcolor_all = True
    plots_effect_M_target_precision = True
    plots_multiple_precisions = True

    plots_effect_M_target_kappa = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_all_precisions_mean = (utils.nanmean(data_pbs.dict_arrays['result_all_precisions']['results'], axis=-1))
    result_all_precisions_std = (utils.nanstd(data_pbs.dict_arrays['result_all_precisions']['results'], axis=-1))
    result_em_fits_mean = (utils.nanmean(data_pbs.dict_arrays['result_em_fits']['results'], axis=-1))
    result_em_fits_std = (utils.nanstd(data_pbs.dict_arrays['result_em_fits']['results'], axis=-1))

    M_space = data_pbs.loaded_data['parameters_uniques']['M'].astype(int)
    ratio_space = data_pbs.loaded_data['parameters_uniques']['ratio_conj']
    num_repetitions = generator_module.num_repetitions

    print M_space
    print ratio_space
    print result_all_precisions_mean.shape, result_em_fits_mean.shape

    dataio = DataIO.DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    target_precision = 100.
    dist_to_target_precision = (result_all_precisions_mean - target_precision)**2.
    ratio_target_precision_given_M = ratio_space[np.argmin(dist_to_target_precision, axis=1)]

    if plots_pcolor_all:
        # Check evolution of precision given M and ratio
        utils.pcolor_2d_data(result_all_precisions_mean, log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='precision wrt M / ratio')
        if savefigs:
            dataio.save_current_figure('precision_log_pcolor_{label}_{unique_id}.pdf')

        # See distance to target precision evolution
        utils.pcolor_2d_data(dist_to_target_precision, log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='Dist to target precision')
        if savefigs:
            dataio.save_current_figure('dist_targetprecision_log_pcolor_{label}_{unique_id}.pdf')


        # Show kappa
        utils.pcolor_2d_data(result_em_fits_mean[..., 0], log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='kappa wrt M / ratio')
        if savefigs:
            dataio.save_current_figure('kappa_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data((result_em_fits_mean[..., 0] - 200)**2., log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='dist to kappa')
        if savefigs:
            dataio.save_current_figure('dist_kappa_log_pcolor_{label}_{unique_id}.pdf')


    if plots_effect_M_target_precision:
        def plot_ratio_target_precision(ratio_target_precision_given_M, target_precision):
            f, ax = plt.subplots()
            ax.plot(M_space, ratio_target_precision_given_M)
            ax.set_xlabel('M')
            ax.set_ylabel('Optimal ratio')
            ax.set_title('Optimal Ratio for precison %d' % target_precision)

            if savefigs:
                dataio.save_current_figure('effect_ratio_M_targetprecision%d_{label}_{unique_id}.pdf' % target_precision)

        plot_ratio_target_precision(ratio_target_precision_given_M, target_precision)

        if plots_multiple_precisions:
            target_precisions = np.array([100, 200, 300, 500, 1000])
            for target_precision in target_precisions:
                dist_to_target_precision = (result_all_precisions_mean - target_precision)**2.
                ratio_target_precision_given_M = ratio_space[np.argmin(dist_to_target_precision, axis=1)]

                # replot
                plot_ratio_target_precision(ratio_target_precision_given_M, target_precision)

    if plots_effect_M_target_kappa:
        def plot_ratio_target_kappa(ratio_target_kappa_given_M, target_kappa):
            f, ax = plt.subplots()
            ax.plot(M_space, ratio_target_kappa_given_M)
            ax.set_xlabel('M')
            ax.set_ylabel('Optimal ratio')
            ax.set_title('Optimal Ratio for precison %d' % target_kappa)

            if savefigs:
                dataio.save_current_figure('effect_ratio_M_targetkappa%d_{label}_{unique_id}.pdf' % target_kappa)

        target_kappa = np.array([100, 200, 300, 500, 1000, 3000])
        for target_kappa in target_kappa:
            dist_to_target_kappa = (result_em_fits_mean[..., 0] - target_kappa)**2.
            ratio_target_kappa_given_M = ratio_space[np.argmin(dist_to_target_kappa, axis=1)]

            # replot
            plot_ratio_target_kappa(ratio_target_kappa_given_M, target_kappa)

    # Macro plot
    # def mem_plot_precision(sigmax_i, M_i, mem_exp_prec):
    #     ax = utils.plot_mean_std_area(T_space[:mem_exp_prec.size], mem_exp_prec, np.zeros(mem_exp_prec.size), linewidth=3, fmt='o-', markersize=8, label='Experimental data')

    #     ax = utils.plot_mean_std_area(T_space[:mem_exp_prec.size], result_all_precisions_mean[M_i, sigmax_i, :mem_exp_prec.size], result_all_precisions_std[M_i, sigmax_i, :mem_exp_prec.size], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Precision of samples')

    #     # ax = utils.plot_mean_std_area(T_space, 0.5*result_marginal_fi_mean[..., 0][M_i, sigmax_i], 0.5*result_marginal_fi_std[..., 0][M_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Marginal Fisher Information')

    #     # ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][M_i, sigmax_i], result_em_fits_std[..., 0][M_i, sigmax_i], ax_handle=ax, xlabel='Number of items', ylabel="Inverse variance $[rad^{-2}]$", linewidth=3, fmt='o-', markersize=8, label='Fitted kappa')

    #     ax.set_title('M %d, sigmax %.2f' % (M_space[M_i], sigmax_space[sigmax_i]))
    #     ax.legend()
    #     ax.set_xlim([0.9, mem_exp_prec.size + 0.1])
    #     ax.set_xticks(range(1, mem_exp_prec.size + 1))
    #     ax.set_xticklabels(range(1, mem_exp_prec.size + 1))

    #     if savefigs:
    #         dataio.save_current_figure('memorycurves_precision_M%dsigmax%.2f_{label}_{unique_id}.pdf' % (M_space[M_i], sigmax_space[sigmax_i]))

    # def mem_plot_kappa(sigmax_i, M_i, exp_kappa_mean, exp_kappa_std=None):
    #     ax = utils.plot_mean_std_area(T_space[:exp_kappa_mean.size], exp_kappa_mean, exp_kappa_std, linewidth=3, fmt='o-', markersize=8, label='Experimental data')

    #     ax = utils.plot_mean_std_area(T_space[:exp_kappa_mean.size], result_em_fits_mean[..., :exp_kappa_mean.size, 0][M_i, sigmax_i], result_em_fits_std[..., :exp_kappa_mean.size, 0][M_i, sigmax_i], xlabel='Number of items', ylabel="Memory error $[rad^{-2}]$", linewidth=3, fmt='o-', markersize=8, label='Fitted kappa', ax_handle=ax)

    #     # ax = utils.plot_mean_std_area(T_space, 0.5*result_marginal_fi_mean[..., 0][M_i, sigmax_i], 0.5*result_marginal_fi_std[..., 0][M_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Marginal Fisher Information')

    #     ax.set_title('M %d, sigmax %.2f' % (M_space[M_i], sigmax_space[sigmax_i]))
    #     ax.legend()
    #     ax.set_xlim([0.9, exp_kappa_mean.size+0.1])
    #     ax.set_xticks(range(1, exp_kappa_mean.size+1))
    #     ax.set_xticklabels(range(1, exp_kappa_mean.size+1))

    #     ax.get_figure().canvas.draw()

    #     if savefigs:
    #         dataio.save_current_figure('memorycurves_kappa_M%dsigmax%.2f_{label}_{unique_id}.pdf' % (M_space[M_i], sigmax_space[sigmax_i]))

    # def em_plot(sigmax_i, M_i):
    #     f, ax = plt.subplots()
    #     ax2 = ax.twinx()

    #     # left axis, kappa
    #     ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][M_i, sigmax_i], result_em_fits_std[..., 0][M_i, sigmax_i], xlabel='Number of items', ylabel="Inverse variance $[rad^{-2}]$", ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Fitted kappa', color='k')

    #     # Right axis, mixture probabilities
    #     utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 1][M_i, sigmax_i], result_em_fits_std[..., 1][M_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Target')
    #     utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 2][M_i, sigmax_i], result_em_fits_std[..., 2][M_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Nontarget')
    #     utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 3][M_i, sigmax_i], result_em_fits_std[..., 3][M_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Random')

    #     lines, labels = ax.get_legend_handles_labels()
    #     lines2, labels2 = ax2.get_legend_handles_labels()
    #     ax.legend(lines + lines2, labels + labels2)

    #     ax.set_title('M %d, sigmax %.2f' % (M_space[M_i], sigmax_space[sigmax_i]))
    #     ax.set_xlim([0.9, T_space.size])
    #     ax.set_xticks(range(1, T_space.size+1))
    #     ax.set_xticklabels(range(1, T_space.size+1))

    #     f.canvas.draw()

    #     if savefigs:
    #         dataio.save_current_figure('memorycurves_emfits_M%dsigmax%.2f_{label}_{unique_id}.pdf' % (M_space[M_i], sigmax_space[sigmax_i]))

    # def em_plot_paper(sigmax_i, M_i):
    #     f, ax = plt.subplots()

    #     # Right axis, mixture probabilities
    #     utils.plot_mean_std_area(T_space_bays09, result_em_fits_mean[..., 1][M_i, sigmax_i][:T_space_bays09.size], result_em_fits_std[..., 1][M_i, sigmax_i][:T_space_bays09.size], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Target')
    #     utils.plot_mean_std_area(T_space_bays09, result_em_fits_mean[..., 2][M_i, sigmax_i][:T_space_bays09.size], result_em_fits_std[..., 2][M_i, sigmax_i][:T_space_bays09.size], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Nontarget')
    #     utils.plot_mean_std_area(T_space_bays09, result_em_fits_mean[..., 3][M_i, sigmax_i][:T_space_bays09.size], result_em_fits_std[..., 3][M_i, sigmax_i][:T_space_bays09.size], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Random')

    #     ax.legend(prop={'size':15})

    #     ax.set_title('M %d, sigmax %.2f' % (M_space[M_i], sigmax_space[sigmax_i]))
    #     ax.set_xlim([1.0, T_space_bays09.size])
    #     ax.set_ylim([0.0, 1.1])
    #     ax.set_xticks(range(1, T_space_bays09.size+1))
    #     ax.set_xticklabels(range(1, T_space_bays09.size+1))

    #     f.canvas.draw()

    #     if savefigs:
    #         dataio.save_current_figure('memorycurves_emfits_paper_M%dsigmax%.2f_{label}_{unique_id}.pdf' % (M_space[M_i], sigmax_space[sigmax_i]))



    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = []

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='ratio_scaling_M')

    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Runs and collect precision and mixture model fits for varying M and ratio_conj. Should then look for a specific precision/kappa and see how ratio_conj evolves with M',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['M', 'ratio_conj'],
                     variables_to_load=['result_all_precisions', 'result_em_fits'],
                     variables_description=['Precision of recall', 'Fits mixture model'],
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

