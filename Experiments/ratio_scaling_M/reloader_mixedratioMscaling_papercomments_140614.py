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

    plots_subpopulations_effects = True

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

    if plots_subpopulations_effects:
        # result_all_precisions_mean
        for M_tot_selected_i, M_tot_selected in enumerate(M_space[::2]):

            M_conj_space = ((1.-ratio_space)*M_tot_selected).astype(int)
            M_feat_space = M_tot_selected - M_conj_space

            f, axes = plt.subplots(2, 2)
            axes[0, 0].plot(ratio_space, result_all_precisions_mean[M_tot_selected_i])
            axes[0, 0].set_xlabel('ratio')
            axes[0, 0].set_title('Measured precision')

            axes[1, 0].plot(ratio_space, M_conj_space**2*M_feat_space)
            axes[1, 0].set_xlabel('M_feat_size')
            axes[1, 0].set_title('M_c**2*M_f')

            axes[0, 1].plot(ratio_space, M_conj_space**2.)
            axes[0, 1].set_xlabel('M')
            axes[0, 1].set_title('M_c**2')

            axes[1, 1].plot(ratio_space, M_feat_space)
            axes[1, 1].set_xlabel('M')
            axes[1, 1].set_title('M_f')

            f.suptitle('M_tot %d' % M_tot_selected, fontsize=15)
            f.set_tight_layout(True)

            if savefigs:
                dataio.save_current_figure('scaling_subpop_Mtot%d_{label}_{unique_id}.pdf' % M_tot_selected)

            plt.close(f)





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

