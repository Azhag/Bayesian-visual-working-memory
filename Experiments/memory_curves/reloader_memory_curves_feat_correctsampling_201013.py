"""
    ExperimentDescriptor to fit Memory curves using a Feature population code

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

# Commit @e055373 +


def plots_memory_curves(data_pbs, generator_module=None):
    '''
        Reload and plot memory curve of a feature code.
        Can use Marginal Fisher Information and fitted Mixture Model as well
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plot_pcolor_fit_precision_to_fisherinfo = True
    plot_selected_memory_curves = True
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
    data_simult = load_experimental_data.load_data_simult(data_dir=os.path.normpath(os.path.join(os.path.split(load_experimental_data.__file__)[0], '../../experimental_data/')))
    memory_experimental = data_simult['precision_nitems_theo']

    data_bays2009 = load_experimental_data.load_data_bays2009(data_dir=os.path.normpath(os.path.join(os.path.split(load_experimental_data.__file__)[0], '../../experimental_data/')), fit_mixture_model=True)
    bays09_experimental_mixtures_mean = data_bays2009['em_fits_nitems_arrays']['mean'][1:]
    # add interpolated points for 3 and 5 items
    bays3 = (bays09_experimental_mixtures_mean[:, 2] + bays09_experimental_mixtures_mean[:, 1])/2.
    bays5 = (bays09_experimental_mixtures_mean[:, -1] + bays09_experimental_mixtures_mean[:, -2])/2.
    bays09_experimental_mixtures_mean_compatible = c_[bays09_experimental_mixtures_mean[:,:2], bays3, bays09_experimental_mixtures_mean[:, 2], bays5]

    # Boost non-targets
    bays09_experimental_mixtures_mean_compatible[1] *= 1.5
    bays09_experimental_mixtures_mean_compatible[2] /= 1.5
    bays09_experimental_mixtures_mean_compatible /= np.sum(bays09_experimental_mixtures_mean_compatible, axis=0)

    # Force non target em fit mixture to be zero and not nan
    result_em_fits_mean[..., 0, 2] = 0
    result_em_fits_std[..., 0, 2] = 0

    # Compute some landscapes of fit!
    dist_diff_precision_margfi = np.sum(np.abs(result_all_precisions_mean*2. - result_marginal_fi_mean[..., 0])**2., axis=-1)
    dist_diff_precision_margfi_1item = np.abs(result_all_precisions_mean[..., 0]*2. - result_marginal_fi_mean[..., 0, 0])**2.


    dist_diff_emkappa_margfi = np.sum(np.abs(result_em_fits_mean[..., 0]*2. - result_marginal_fi_mean[..., 0])**2., axis=-1)
    dist_ratio_emkappa_margfi = np.sum(np.abs((result_em_fits_mean[..., 0]*2.)/result_marginal_fi_mean[..., 0] - 1.0)**2., axis=-1)

    dist_diff_precision_experim = np.sum(np.abs(result_all_precisions_mean - memory_experimental)**2., axis=-1)
    dist_diff_precision_experim_1item = np.abs(result_all_precisions_mean[..., 0] - memory_experimental[0])**2.

    dist_diff_emkappa_experim = np.sum(np.abs(result_em_fits_mean[..., 0] - memory_experimental)**2., axis=-1)
    dist_diff_emkappa_experim_1item = np.abs(result_em_fits_mean[..., 0, 0] - memory_experimental[0])**2.

    dist_diff_margfi_experim_1item = np.abs(result_marginal_fi_mean[..., 0, 0] - memory_experimental[0])**2.

    dist_diff_emkappa_mixtures_bays09 = np.sum(np.sum((result_em_fits_mean[..., 1:4] - bays09_experimental_mixtures_mean_compatible.T)**2., axis=-1), axis=-1)


    if plot_pcolor_fit_precision_to_fisherinfo:
        # Check fit between precision and fisher info
        utils.pcolor_2d_data(dist_diff_precision_margfi, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')

        if savefigs:
           dataio.save_current_figure('match_precision_margfi_log_pcolor_{label}_{unique_id}.pdf')

        # utils.pcolor_2d_data(dist_diff_precision_margfi, x=M_space, y=sigmax_space[2:], xlabel='M', ylabel='sigmax')
        # if savefigs:
        #    dataio.save_current_figure('match_precision_margfi_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data(dist_diff_precision_experim, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax', log_scale=True)

        utils.pcolor_2d_data(dist_diff_emkappa_experim, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax', log_scale=True)

        utils.pcolor_2d_data(dist_diff_precision_margfi*dist_diff_emkappa_margfi*dist_diff_precision_experim*dist_diff_emkappa_experim, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax', log_scale=True)

        utils.pcolor_2d_data(dist_diff_precision_margfi_1item, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')

        utils.pcolor_2d_data(dist_diff_precision_experim_1item, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')

        utils.pcolor_2d_data(dist_diff_emkappa_experim_1item, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')

        utils.pcolor_2d_data(dist_diff_margfi_experim_1item, log_scale=True, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')

        utils.pcolor_2d_data(dist_diff_emkappa_mixtures_bays09, log_scale=False, x=M_space, y=sigmax_space, xlabel='M', ylabel='sigmax')



    if plot_selected_memory_curves:
        selected_values = [[100, 0.8], [200, 0.27], [100, 0.1], [200, 0.8], [100, 0.17], [100, 0.08], [50, 0.21]]

        for current_values in selected_values:
            # Find the indices
            M_i         = np.argmin(np.abs(current_values[0] - M_space))
            sigmax_i    = np.argmin(np.abs(current_values[1] - sigmax_space))

            ax = utils.plot_mean_std_area(T_space, memory_experimental, np.zeros(T_space.size), linewidth=3, fmt='o-', markersize=8)

            ax = utils.plot_mean_std_area(T_space, result_all_precisions_mean[M_i, sigmax_i], result_all_precisions_std[M_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8)

            ax = utils.plot_mean_std_area(T_space, 0.5*result_marginal_fi_mean[..., 0][M_i, sigmax_i], 0.5*result_marginal_fi_std[..., 0][M_i, sigmax_i], ax_handle=ax, linewidth=3, fmt='o-', markersize=8)

            ax = utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 0][M_i, sigmax_i], result_em_fits_std[..., 0][M_i, sigmax_i], ax_handle=ax, xlabel='Number of items', ylabel="Inverse variance $[rad^{-2}]$", linewidth=3, fmt='o-', markersize=8)

            # ax.set_title('M %d, sigmax %.2f' % (M_space[M_i], sigmax_space[sigmax_i]))
            plt.legend(['Experimental data', 'Precision of samples', 'Marginal Fisher Information', 'Fitted kappa'])

            ax.set_xlim([0.9, 5.1])
            ax.set_xticks(range(1, 6))
            ax.set_xticklabels(range(1, 6))

            if savefigs:
                dataio.save_current_figure('memorycurves_M%dsigmax%.2f_{label}_{unique_id}.pdf' % (M_space[M_i], sigmax_space[sigmax_i]))

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

    if plot_best_memory_curves:

        # Best mixtures fit
        best_axis2_i_all = np.argmin(dist_diff_emkappa_mixtures_bays09, axis=1)
        for axis1_i, best_axis2_i in enumerate(best_axis2_i_all):
            em_plot_paper(best_axis2_i, axis1_i)


    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['result_all_precisions_mean', 'result_em_fits_mean', 'result_marginal_inv_fi_mean', 'result_all_precisions_std', 'result_em_fits_std', 'result_marginal_inv_fi_std', 'result_marginal_fi_mean', 'result_marginal_fi_std', 'M_space', 'sigmax_space', 'T_space', 'all_args']

    if savedata:
        dataio.save_variables(variables_to_save, locals())


    # Make link to Dropbox
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

