"""
    ExperimentDescriptor to fit Memory curves using a Hierarchical population code

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

    plot_pcolor_fit_precision_to_fisherinfo = True
    plot_selected_memory_curves = False
    plot_best_memory_curves = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_all_precisions_mean = utils.nanmean(np.squeeze(data_pbs.dict_arrays['result_all_precisions']['results']), axis=-1)
    result_all_precisions_std = utils.nanstd(np.squeeze(data_pbs.dict_arrays['result_all_precisions']['results']), axis=-1)
    result_em_fits_mean = utils.nanmean(np.squeeze(data_pbs.dict_arrays['result_em_fits']['results']), axis=-1)
    result_em_fits_std = utils.nanstd(np.squeeze(data_pbs.dict_arrays['result_em_fits']['results']), axis=-1)

    ratiohier_space = data_pbs.loaded_data['parameters_uniques']['ratio_hierarchical']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    T_space = data_pbs.loaded_data['datasets_list'][0]['T_space']

    print ratiohier_space
    print sigmax_space
    print T_space
    print result_all_precisions_mean.shape, result_em_fits_mean.shape

    dataio = DataIO.DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    ## Load Experimental data
    data_simult = load_experimental_data.load_data_simult(data_dir=os.path.normpath(os.path.join(os.path.split(load_experimental_data.__file__)[0], '../../experimental_data/')), fit_mixturemodel=True)
    memory_experimental_precision = data_simult['precision_nitems_theo']
    memory_experimental_kappa = np.array([data['kappa'] for _, data in data_simult['em_fits_nitems']['mean'].items()])

    # Compute some landscapes of fit!
    dist_diff_precision_experim = np.sum(np.abs(result_all_precisions_mean - memory_experimental_precision)**2., axis=-1)
    dist_diff_emkappa_experim = np.sum(np.abs(result_em_fits_mean[..., 0] - memory_experimental_kappa)**2., axis=-1)

    dist_diff_precision_experim_1item = np.abs(result_all_precisions_mean[..., 0] - memory_experimental_precision[0])**2.
    dist_diff_precision_experim_2item = np.abs(result_all_precisions_mean[..., 1] - memory_experimental_precision[1])**2.
    dist_diff_emkappa_experim_1item = np.abs(result_em_fits_mean[..., 0, 0] - memory_experimental_kappa[0])**2.


    if plot_pcolor_fit_precision_to_fisherinfo:
        utils.pcolor_2d_data(dist_diff_precision_experim, x=ratiohier_space, y=sigmax_space, xlabel='ratio hier', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_diff_precision_experim_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_emkappa_experim, x=ratiohier_space, y=sigmax_space, xlabel='ratio hier', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_diff_emkappa_experim_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_precision_experim*dist_diff_emkappa_experim, x=ratiohier_space, y=sigmax_space, xlabel='ratio hier', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_bigmultiplication_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_precision_experim_1item, log_scale=True, x=ratiohier_space, y=sigmax_space, xlabel='ratio hier', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_precision_experim_1item_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_emkappa_experim_1item, log_scale=True, x=ratiohier_space, y=sigmax_space, xlabel='ratio hier', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_emkappa_experim_1item_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_precision_experim_2item, log_scale=True, x=ratiohier_space, y=sigmax_space, xlabel='ratio hier', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_precision_experim_2item_log_pcolor_{label}_{unique_id}.pdf')

    # Macro plot
    def mem_plot_precision(sigmax_i, ratiohier_i):
        ax = utils.plot_mean_std_area(T_space, memory_experimental_precision, np.zeros(T_space.size), linewidth=3, fmt='o-', markersize=8, label='Experimental data')

        ax = utils.plot_mean_std_area(T_space, result_all_precisions_mean[ratiohier_i, sigmax_i], result_all_precisions_std[ratiohier_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Precision of samples')

        # ax = utils.plot_mean_std_area(T_space, 0.5*result_marginal_fi_mean[..., 0][ratiohier_i, sigmax_i], 0.5*result_marginal_fi_std[..., 0][ratiohier_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Marginal Fisher Information')

        # ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][ratiohier_i, sigmax_i], result_em_fits_std[..., 0][ratiohier_i, sigmax_i], ax_handle=ax, xlabel='Number of items', ylabel="Inverse variance $[rad^{-2}]$", linewidth=3, fmt='o-', markersize=8, label='Fitted kappa')

        ax.set_title('ratio_hier %.2f, sigmax %.2f' % (ratiohier_space[ratiohier_i], sigmax_space[sigmax_i]))
        ax.legend()
        ax.set_xlim([0.9, 5.1])
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(range(1, 6))

        if savefigs:
            dataio.save_current_figure('memorycurves_precision_ratiohier%.2fsigmax%.2f_{label}_{unique_id}.pdf' % (ratiohier_space[ratiohier_i], sigmax_space[sigmax_i]))

    def mem_plot_kappa(sigmax_i, ratiohier_i):
        ax = utils.plot_mean_std_area(T_space, memory_experimental_kappa, np.zeros(T_space.size), linewidth=3, fmt='o-', markersize=8, label='Experimental data')

        ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][ratiohier_i, sigmax_i], result_em_fits_std[..., 0][ratiohier_i, sigmax_i], xlabel='Number of items', ylabel="Inverse variance $[rad^{-2}]$", ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Fitted kappa')

        ax.set_title('ratio_hier %.2f, sigmax %.2f' % (ratiohier_space[ratiohier_i], sigmax_space[sigmax_i]))
        ax.legend()
        ax.set_xlim([0.9, 5.1])
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(range(1, 6))

        if savefigs:
            dataio.save_current_figure('memorycurves_kappa_ratiohier%.2fsigmax%.2f_{label}_{unique_id}.pdf' % (ratiohier_space[ratiohier_i], sigmax_space[sigmax_i]))

    def em_plot(sigmax_i, ratiohier_i):
        # TODO finish checking this up.
        f, ax = plt.subplots()
        ax2 = ax.twinx()

        # left axis, kappa
        ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][ratiohier_i, sigmax_i], result_em_fits_std[..., 0][ratiohier_i, sigmax_i], xlabel='Number of items', ylabel="Inverse variance $[rad^{-2}]$", ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Fitted kappa', color='k')

        # Right axis, mixture probabilities
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 1][ratiohier_i, sigmax_i], result_em_fits_std[..., 1][ratiohier_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Target')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 2][ratiohier_i, sigmax_i], result_em_fits_std[..., 2][ratiohier_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Nontarget')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 3][ratiohier_i, sigmax_i], result_em_fits_std[..., 3][ratiohier_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax2, linewidth=3, fmt='o-', markersize=8, label='Random')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)

        ax.set_title('ratio_hier %.2f, sigmax %.2f' % (ratiohier_space[ratiohier_i], sigmax_space[sigmax_i]))
        ax.set_xlim([0.9, 5.1])
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(range(1, 6))

        f.canvas.draw()

        if savefigs:
            dataio.save_current_figure('memorycurves_emfits_ratiohier%.2fsigmax%.2f_{label}_{unique_id}.pdf' % (ratiohier_space[ratiohier_i], sigmax_space[sigmax_i]))


    if plot_selected_memory_curves:
        selected_values = [[0.84, 0.23], [0.84, 0.19]]

        for current_values in selected_values:
            # Find the indices
            ratiohier_i     = np.argmin(np.abs(current_values[0] - ratiohier_space))
            sigmax_i        = np.argmin(np.abs(current_values[1] - sigmax_space))


            mem_plot_precision(sigmax_i, ratiohier_i)
            mem_plot_kappa(sigmax_i, ratiohier_i)

    if plot_best_memory_curves:

        # # Best precision fit
        # best_axis2_i_all = np.argmin(dist_diff_precision_experim, axis=1)
        # for axis1_i, best_axis2_i in enumerate(best_axis2_i_all):
        #     mem_plot_precision(best_axis2_i, axis1_i)

        # Best kappa fit
        best_axis2_i_all = np.argmin(dist_diff_emkappa_experim, axis=1)
        for axis1_i, best_axis2_i in enumerate(best_axis2_i_all):
            mem_plot_kappa(best_axis2_i, axis1_i)
            em_plot(best_axis2_i, axis1_i)

    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['memory_experimental_precision', 'memory_experimental_kappa']

    if savedata:
        dataio.save_variables_default(locals(), additional_variables=variables_to_save)

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='memory_curves')

    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

# generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]
generator_script = 'generator_memory_curves_hierarchical_newruns_141113.py'

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Fit Memory curves using the new code (october 2013). Compute marginal inverse fisher information, which is slightly better at capturing items interactions effects. Also fit Mixture models directly. Uses a Hierarchical population now. No Fisher information in this setup, not available.',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['ratio_hierarchical', 'sigmax'],
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

