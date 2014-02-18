"""
    ExperimentDescriptor to fit Memory curves using a Conjunctive population code

    Uses the new Marginal Inverse Fisher Information, and some new code altogether.
    Precisions do fit nicely, given a factor of 2.
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
# Commit @e055373 +


def plots_memory_curves(data_pbs, generator_module=None):
    '''
        Reload and plot memory curve of a conjunctive code.
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
    result_marginal_inv_fi_mean = utils.nanmean(np.squeeze(data_pbs.dict_arrays['result_marginal_inv_fi']['results']), axis=-1)
    result_marginal_inv_fi_std = utils.nanstd(np.squeeze(data_pbs.dict_arrays['result_marginal_inv_fi']['results']), axis=-1)
    result_marginal_fi_mean = utils.nanmean(1./np.squeeze(data_pbs.dict_arrays['result_marginal_inv_fi']['results']), axis=-1)
    result_marginal_fi_std = utils.nanstd(1./np.squeeze(data_pbs.dict_arrays['result_marginal_inv_fi']['results']), axis=-1)

    M_space = data_pbs.loaded_data['parameters_uniques']['M']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    T_space = data_pbs.loaded_data['datasets_list'][0]['T_space']

    print M_space
    print sigmax_space
    print T_space
    print result_all_precisions_mean.shape, result_em_fits_mean.shape, result_marginal_inv_fi_mean.shape

    dataio = DataIO.DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    ## Load Experimental data
    experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(load_experimental_data.__file__)[0])

    data_simult = load_experimental_data.load_data_simult(data_dir=os.path.normpath(os.path.join(experim_datadir, '../../experimental_data/')), fit_mixture_model=True)
    gorgo11_experimental_precision = data_simult['precision_nitems_theo']
    gorgo11_experimental_kappa = np.array([data_s['kappa'] for _, data_s in data_simult['em_fits_nitems']['mean'].items()])
    gorgo11_experimental_emfits_mean = np.array([[data[key] for _, data in data_simult['em_fits_nitems']['mean'].items()] for key in ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random']])
    gorgo11_experimental_emfits_std = np.array([[data[key] for _, data in data_simult['em_fits_nitems']['std'].items()] for key in ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random']])
    gorgo11_experimental_emfits_sem = gorgo11_experimental_emfits_std/np.sqrt(np.unique(data_simult['subject']).size)


    data_bays2009 = load_experimental_data.load_data_bays2009(data_dir=os.path.normpath(os.path.join(experim_datadir, '../../experimental_data/')), fit_mixture_model=True)
    # add interpolated points for 3 and 5 items
    emfit_mean_intpfct = spint.interp1d(np.unique(data_bays2009['n_items']), data_bays2009['em_fits_nitems_arrays']['mean'])
    bays09_experimental_mixtures_mean_compatible = emfit_mean_intpfct(np.arange(1, 6))
    emfit_std_intpfct = spint.interp1d(np.unique(data_bays2009['n_items']), data_bays2009['em_fits_nitems_arrays']['std'])
    bays09_experimental_mixtures_std_compatible = emfit_std_intpfct(np.arange(1, 6))

    # Boost non-targets
    # bays09_experimental_mixtures_mean_compatible[1] *= 1.5
    # bays09_experimental_mixtures_mean_compatible[2] /= 1.5
    # bays09_experimental_mixtures_mean_compatible /= np.sum(bays09_experimental_mixtures_mean_compatible, axis=0)

    # Force non target em fit mixture to be zero and not nan
    result_em_fits_mean[..., 0, 2] = 0
    result_em_fits_std[..., 0, 2] = 0

    # Compute some landscapes of fit!
    dist_diff_precision_margfi = np.sum(np.abs(result_all_precisions_mean*2. - result_marginal_fi_mean[..., 0])**2., axis=-1)
    dist_ratio_precision_margfi = np.sum(np.abs((result_all_precisions_mean*2.)/result_marginal_fi_mean[..., 0] - 1.0)**2., axis=-1)
    dist_diff_emkappa_margfi = np.sum(np.abs(result_em_fits_mean[..., 0]*2. - result_marginal_fi_mean[..., 0])**2., axis=-1)
    dist_ratio_emkappa_margfi = np.sum(np.abs((result_em_fits_mean[..., 0]*2.)/result_marginal_fi_mean[..., 0] - 1.0)**2., axis=-1)

    dist_diff_precision_experim = np.sum(np.abs(result_all_precisions_mean - gorgo11_experimental_precision)**2., axis=-1)
    dist_diff_emkappa_experim = np.sum(np.abs(result_em_fits_mean[..., 0] - gorgo11_experimental_kappa)**2., axis=-1)

    dist_diff_precision_experim_1item = np.abs(result_all_precisions_mean[..., 0] - gorgo11_experimental_precision[0])**2.
    dist_diff_precision_experim_2item = np.abs(result_all_precisions_mean[..., 1] - gorgo11_experimental_precision[1])**2.
    dist_diff_precision_margfi_1item = np.abs(result_all_precisions_mean[..., 0]*2. - result_marginal_fi_mean[..., 0, 0])**2.
    dist_diff_emkappa_experim_1item = np.abs(result_em_fits_mean[..., 0, 0] - gorgo11_experimental_kappa[0])**2.
    dist_diff_margfi_experim_1item = np.abs(result_marginal_fi_mean[..., 0, 0] - gorgo11_experimental_precision[0])**2.

    dist_diff_emkappa_mixtures_bays09 = np.sum(np.sum((result_em_fits_mean[..., 1:4] - bays09_experimental_mixtures_mean_compatible[1:].T)**2., axis=-1), axis=-1)

    if plot_pcolor_fit_precision_to_fisherinfo:
        # Check fit between precision and fisher info
        utils.pcolor_2d_data(dist_diff_precision_margfi, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')

        if savefigs:
            dataio.save_current_figure('match_precision_margfi_log_pcolor_{label}_{unique_id}.pdf')

        # utils.pcolor_2d_data(dist_diff_precision_margfi, x=M_space, y=sigmax_space[2:], xlabel='M', ylabel='sigmax')
        # if savefigs:
        #    dataio.save_current_figure('match_precision_margfi_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_ratio_precision_margfi, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_ratio_precision_margfi_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_emkappa_margfi, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_diff_emkappa_margfi_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_ratio_emkappa_margfi, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_ratio_emkappa_margfi_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_precision_experim, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_diff_precision_experim_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_emkappa_experim, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_diff_emkappa_experim_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_precision_margfi*dist_diff_emkappa_margfi*dist_diff_precision_experim*dist_diff_emkappa_experim, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax', log_scale=True)
        if savefigs:
            dataio.save_current_figure('match_bigmultiplication_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_precision_margfi_1item, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_precision_margfi_1item_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_precision_experim_1item, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_precision_experim_1item_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_emkappa_experim_1item, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_emkappa_experim_1item_log_pcolor_{label}_{unique_id}.pdf')


        utils.pcolor_2d_data(dist_diff_margfi_experim_1item, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_margfi_experim_1item_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_emkappa_mixtures_bays09, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')
        if savefigs:
            dataio.save_current_figure('match_diff_mixtures_experbays09_pcolor_{label}_{unique_id}.pdf')


    # Macro plot
    def mem_plot_precision(sigmax_i, M_i):
        ax = utils.plot_mean_std_area(T_space, gorgo11_experimental_precision, np.zeros(T_space.size), linewidth=3, fmt='o-', markersize=8, label='Experimental data')

        ax = utils.plot_mean_std_area(T_space, result_all_precisions_mean[M_i, sigmax_i], result_all_precisions_std[M_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Precision of samples')

        ax = utils.plot_mean_std_area(T_space, 0.5*result_marginal_fi_mean[..., 0][M_i, sigmax_i], 0.5*result_marginal_fi_std[..., 0][M_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Marginal Fisher Information')

        # ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][M_i, sigmax_i], result_em_fits_std[..., 0][M_i, sigmax_i], ax_handle=ax, xlabel='Number of items', ylabel="Inverse variance $[rad^{-2}]$", linewidth=3, fmt='o-', markersize=8, label='Fitted kappa')

        ax.set_title('M %d, sigmax %.2f' % (M_space[M_i], sigmax_space[sigmax_i]))
        ax.legend()
        ax.set_xlim([0.9, 5.1])
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(range(1, 6))

        ax.get_figure().canvas.draw()

        if savefigs:
            dataio.save_current_figure('memorycurves_precision_M%dsigmax%.2f_{label}_{unique_id}.pdf' % (M_space[M_i], sigmax_space[sigmax_i]))


    def mem_plot_kappa(sigmax_i, M_i, experim_data_mean, experim_data_std=None):
        ax = utils.plot_mean_std_area(T_space, experim_data_mean, experim_data_std, linewidth=3, fmt='o-', markersize=8, label='Experimental data')

        ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][M_i, sigmax_i], result_em_fits_std[..., 0][M_i, sigmax_i], xlabel='Number of items', ylabel="Inverse variance $[rad^{-2}]$", ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Fitted kappa')

        # ax = utils.plot_mean_std_area(T_space, 0.5*result_marginal_fi_mean[..., 0][M_i, sigmax_i], 0.5*result_marginal_fi_std[..., 0][M_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8, label='Marginal Fisher Information')

        ax.set_title('M %d, sigmax %.2f' % (M_space[M_i], sigmax_space[sigmax_i]))
        ax.legend()
        ax.set_xlim([0.9, 5.1])
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(range(1, 6))

        ax.get_figure().canvas.draw()

        if savefigs:
            dataio.save_current_figure('memorycurves_kappa_M%dsigmax%.2f_{label}_{unique_id}.pdf' % (M_space[M_i], sigmax_space[sigmax_i]))

    def em_plot_paper(sigmax_i, M_i):
        f, ax = plt.subplots()

        # Right axis, mixture probabilities
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 1][M_i, sigmax_i], result_em_fits_std[..., 1][M_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Target')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 2][M_i, sigmax_i], result_em_fits_std[..., 2][M_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Nontarget')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 3][M_i, sigmax_i], result_em_fits_std[..., 3][M_i, sigmax_i], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Random')

        ax.legend(prop={'size':15})

        ax.set_title('M %d, sigmax %.2f' % (M_space[M_i], sigmax_space[sigmax_i]))
        ax.set_xlim([1.0, 5.0])
        ax.set_ylim([0.0, 1.1])
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(range(1, 6))

        f.canvas.draw()

        if savefigs:
            dataio.save_current_figure('memorycurves_emfits_paper_M%.2fsigmax%.2f_{label}_{unique_id}.pdf' % (M_space[M_i], sigmax_space[sigmax_i]))


    if plot_selected_memory_curves:
        selected_values = [[100, 0.1], [196, 0.28], [64, 0.05]]

        for current_values in selected_values:
            # Find the indices
            M_i         = np.argmin(np.abs(current_values[0] - M_space))
            sigmax_i    = np.argmin(np.abs(current_values[1] - sigmax_space))

            mem_plot_precision(sigmax_i, M_i)

            mem_plot_kappa(sigmax_i, M_i)


    if plot_best_memory_curves:
        # Best precision fit
        best_axis2_i_all = np.argmin(dist_diff_precision_experim, axis=1)

        for axis1_i, best_axis2_i in enumerate(best_axis2_i_all):
            mem_plot_precision(best_axis2_i, axis1_i)

        # Best kappa fit
        best_axis2_i_all = np.argmin(dist_diff_emkappa_experim, axis=1)

        for axis1_i, best_axis2_i in enumerate(best_axis2_i_all):
            mem_plot_kappa(best_axis2_i, axis1_i, gorgo11_experimental_emfits_mean[0], gorgo11_experimental_emfits_std[0])

            mem_plot_precision(best_axis2_i, axis1_i)

        # Best mixtures fit for Bays09
        best_axis2_i_all = np.argmin(dist_diff_emkappa_mixtures_bays09, axis=1)

        for axis1_i, best_axis2_i in enumerate(best_axis2_i_all):
            em_plot_paper(best_axis2_i, axis1_i)



    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['gorgo11_experimental_precision', 'gorgo11_experimental_kappa']

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
dataset_infos = dict(label='Fit Memory curves using the new code (october 2013). Compute marginal inverse fisher information, which is slightly better at capturing items interactions effects. Also fit Mixture models directly.',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['M', 'sigmax'],
                     variables_to_load=['result_all_precisions', 'result_em_fits', 'result_marginal_inv_fi'],
                     variables_description=['Precision of recall', 'Fits mixture model', 'Marginal Inverse Fisher Information'],
                     post_processing=plots_memory_curves,
                     save_output_filename='plots_memory_curves'
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

